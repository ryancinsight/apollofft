//! WGPU compute dispatch logic for the FFT execution.

use crate::infrastructure::gpu_fft::pipeline::GpuFft3d;
use crate::infrastructure::gpu_fft::strategy::{Axis, ChirpData, RadixStages};

impl GpuFft3d {
    pub(crate) fn dispatch_pack(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        axis: Axis,
        axis_len: u32,
        batch_count: u32,
    ) {
        let stage = match axis {
            Axis::X => &self.pack_bg_x,
            Axis::Y => &self.pack_bg_y,
            Axis::Z => &self.pack_bg_z,
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apollofft-wgpu pack pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pack_pipeline);
        pass.set_bind_group(0, &self.data_bg, &[]);
        pass.set_bind_group(1, &stage.fft_params_bg, &[]);
        pass.set_bind_group(2, &stage.bg, &[]);
        let total = axis_len * batch_count;
        pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
    }

    pub(crate) fn dispatch_unpack(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        axis: Axis,
        axis_len: u32,
        batch_count: u32,
    ) {
        let stage = match axis {
            Axis::X => &self.pack_bg_x,
            Axis::Y => &self.pack_bg_y,
            Axis::Z => &self.pack_bg_z,
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apollofft-wgpu unpack pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.unpack_pipeline);
        pass.set_bind_group(0, &self.data_bg, &[]);
        pass.set_bind_group(1, &stage.fft_params_bg, &[]);
        pass.set_bind_group(2, &stage.bg, &[]);
        let total = axis_len * batch_count;
        pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
    }

    pub(crate) fn dispatch_radix2(&self, encoder: &mut wgpu::CommandEncoder, stages: &RadixStages) {
        if stages.fft_m == 0 {
            return;
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apollofft-wgpu radix2 pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.bitrev_pipeline);
        pass.set_bind_group(0, &self.data_bg, &[]);
        pass.set_bind_group(1, &stages.bgs[0], &[]);
        pass.dispatch_workgroups(stages.batch_count, stages.fft_m / 2, 1);

        pass.set_pipeline(&self.forward_pipeline);
        for stage_idx in 0..stages.fft_m.trailing_zeros() as usize {
            pass.set_bind_group(1, &stages.bgs[1 + stage_idx], &[]);
            pass.dispatch_workgroups(stages.batch_count, stages.fft_m / 2, 1);
        }

        if stages.bgs.len() > 1 + stages.fft_m.trailing_zeros() as usize {
            pass.set_pipeline(&self.scale_pipeline);
            pass.set_bind_group(1, stages.bgs.last().unwrap(), &[]);
            pass.dispatch_workgroups(stages.batch_count, stages.fft_m, 1);
        }
    }

    pub(crate) fn dispatch_chirp(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        chirp: &ChirpData,
        inverse: bool,
    ) {
        if chirp.n == 0 || chirp.m == 0 {
            return;
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apollofft-wgpu chirp pass (premul)"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&chirp.premul_pipeline);
        pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
        pass.set_bind_group(1, &chirp.params_bg, &[]);
        pass.dispatch_workgroups(chirp.batch_count, chirp.m, 1);
        drop(pass);

        if inverse {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollofft-wgpu chirp pass (negate_im)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.negate_im_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(chirp.batch_count, chirp.m, 1);
        }

        self.dispatch_radix2(encoder, &chirp.radix2_fwd);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apollofft-wgpu chirp pass (pointmul)"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&chirp.pointmul_pipeline);
        pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
        pass.set_bind_group(1, &chirp.params_bg, &[]);
        pass.dispatch_workgroups(chirp.batch_count, chirp.m, 1);
        drop(pass);

        self.dispatch_radix2(encoder, &chirp.radix2_inv);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apollofft-wgpu chirp pass (post)"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&chirp.postmul_pipeline);
        pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
        pass.set_bind_group(1, &chirp.params_bg, &[]);
        pass.dispatch_workgroups(chirp.batch_count, chirp.n, 1);

        if inverse {
            pass.set_pipeline(&chirp.scale_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(chirp.batch_count, chirp.n, 1);
        }
    }
}
