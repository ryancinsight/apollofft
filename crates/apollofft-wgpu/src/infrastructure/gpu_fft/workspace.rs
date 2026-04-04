//! Buffer gathering, scatter operations, and forward/inverse routing for GPU FFT.

use crate::infrastructure::gpu_fft::pipeline::GpuFft3d;
use crate::infrastructure::gpu_fft::strategy::{Axis, AxisStrategy};
use ndarray::Array3;

impl GpuFft3d {
    /// Forward transform of a real field into an interleaved complex buffer.
    #[must_use]
    pub fn forward(&self, field: &Array3<f64>) -> Vec<f32> {
        let n = self.nx * self.ny * self.nz;
        let mut out = vec![0.0_f32; 2 * n];
        self.forward_into(field, &mut out);
        out
    }

    /// Forward transform of a real field into a caller-owned interleaved complex buffer.
    pub fn forward_into(&self, field: &Array3<f64>, out: &mut [f32]) {
        assert_eq!(field.dim(), (self.nx, self.ny, self.nz));
        assert_eq!(out.len(), 2 * self.nx * self.ny * self.nz);
        let n = self.nx * self.ny * self.nz;
        let mut re_data = vec![0.0_f32; n];
        let im_data = vec![0.0_f32; n];
        re_data
            .iter_mut()
            .zip(field.iter().copied())
            .for_each(|(dst, src)| *dst = src as f32);

        self.queue
            .write_buffer(&self.full_re_buf, 0, bytemuck::cast_slice(&re_data));
        self.queue
            .write_buffer(&self.full_im_buf, 0, bytemuck::cast_slice(&im_data));

        self.run_device_axis_fft(Axis::Z, false);
        self.run_device_axis_fft(Axis::Y, false);
        self.run_device_axis_fft(Axis::X, false);

        let (re_out, im_out) = self.read_back_full_buffers();

        for ((re, im), pair) in re_out
            .iter()
            .zip(im_out.iter())
            .zip(out.chunks_exact_mut(2))
        {
            pair[0] = *re;
            pair[1] = *im;
        }
    }

    /// Inverse transform of an interleaved complex buffer into a real field.
    pub fn inverse(&self, field_hat: &[f32], out: &mut Array3<f64>) {
        assert_eq!(field_hat.len(), 2 * self.nx * self.ny * self.nz);
        assert_eq!(out.dim(), (self.nx, self.ny, self.nz));

        let n = self.nx * self.ny * self.nz;
        let mut re_data = vec![0.0_f32; n];
        let mut im_data = vec![0.0_f32; n];
        for (idx, pair) in field_hat.chunks_exact(2).enumerate() {
            re_data[idx] = pair[0];
            im_data[idx] = pair[1];
        }

        self.queue
            .write_buffer(&self.full_re_buf, 0, bytemuck::cast_slice(&re_data));
        self.queue
            .write_buffer(&self.full_im_buf, 0, bytemuck::cast_slice(&im_data));

        self.run_device_axis_fft(Axis::X, true);
        self.run_device_axis_fft(Axis::Y, true);
        self.run_device_axis_fft(Axis::Z, true);

        let (re_out, _) = self.read_back_full_buffers();

        for (dst, &value) in out.iter_mut().zip(re_out.iter()) {
            *dst = value as f64;
        }
    }

    pub(crate) fn run_device_axis_fft(&self, axis: Axis, inverse: bool) {
        let axis_len = axis.len(self.nx, self.ny, self.nz) as u32;
        let batch_count = axis.batch_count(self.nx, self.ny, self.nz) as u32;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollofft-wgpu device axis encoder"),
            });

        self.dispatch_pack(&mut encoder, axis, axis_len, batch_count);
        match axis {
            Axis::Z => match self.strategy_z {
                AxisStrategy::Radix2 => {
                    let stages = if inverse {
                        &self.axis_inv_z
                    } else {
                        &self.axis_fwd_z
                    };
                    self.dispatch_radix2(&mut encoder, stages);
                }
                AxisStrategy::ChirpZ { .. } => self.dispatch_chirp(
                    &mut encoder,
                    self.chirp_z.as_ref().expect("chirp_z exists for ChirpZ"),
                    inverse,
                ),
            },
            Axis::Y => match self.strategy_y {
                AxisStrategy::Radix2 => {
                    let stages = if inverse {
                        &self.axis_inv_y
                    } else {
                        &self.axis_fwd_y
                    };
                    self.dispatch_radix2(&mut encoder, stages);
                }
                AxisStrategy::ChirpZ { .. } => self.dispatch_chirp(
                    &mut encoder,
                    self.chirp_y.as_ref().expect("chirp_y exists for ChirpZ"),
                    inverse,
                ),
            },
            Axis::X => match self.strategy_x {
                AxisStrategy::Radix2 => {
                    let stages = if inverse {
                        &self.axis_inv_x
                    } else {
                        &self.axis_fwd_x
                    };
                    self.dispatch_radix2(&mut encoder, stages);
                }
                AxisStrategy::ChirpZ { .. } => self.dispatch_chirp(
                    &mut encoder,
                    self.chirp_x.as_ref().expect("chirp_x exists for ChirpZ"),
                    inverse,
                ),
            },
        }
        self.dispatch_unpack(&mut encoder, axis, axis_len, batch_count);
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub(crate) fn read_back_full_buffers(&self) -> (Vec<f32>, Vec<f32>) {
        let n = self.nx * self.ny * self.nz;
        let buf_size = (n * std::mem::size_of::<f32>()) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollofft-wgpu readback encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.full_re_buf, 0, &self.full_re_staging, 0, buf_size);
        encoder.copy_buffer_to_buffer(&self.full_im_buf, 0, &self.full_im_staging, 0, buf_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let re_slice = self.full_re_staging.slice(..buf_size);
        let im_slice = self.full_im_staging.slice(..buf_size);
        re_slice.map_async(wgpu::MapMode::Read, |_| {});
        im_slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait);

        let re_out = {
            let mapped = re_slice.get_mapped_range();
            bytemuck::cast_slice(&mapped).to_vec()
        };
        let im_out = {
            let mapped = im_slice.get_mapped_range();
            bytemuck::cast_slice(&mapped).to_vec()
        };
        self.full_re_staging.unmap();
        self.full_im_staging.unmap();
        (re_out, im_out)
    }
}
