//! GPU compute kernels for the forward and inverse Short-Time Fourier Transform.
//!
//! Forward kernel: each invocation computes one complex output element X[frame_m, bin_k]
//! via a direct DFT sum over the Hann-windowed frame.
//!
//! Inverse kernel: two-pass WOLA reconstruction.
//!   Pass 1 (stft_inverse_frames): per-frame IDFT with Hann window.
//!   Pass 2 (stft_inverse_ola): weighted overlap-add across frames.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use num_complex::Complex32;
use wgpu::util::DeviceExt;

use crate::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ComplexPod {
    re: f32,
    im: f32,
}

/// Uniform parameter block shared by all STFT pipelines.
///
/// Layout: 4 × u32 = 16 bytes, satisfying WGPU uniform alignment.
/// Field order matches the WGSL `StftParams` struct byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct StftParams {
    signal_len: u32,
    frame_len: u32,
    hop_len: u32,
    frame_count: u32,
}

/// GPU compute kernel encapsulating the forward and inverse STFT pipelines.
#[derive(Debug)]
pub struct StftGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    forward_pipeline: wgpu::ComputePipeline,
    inverse_frames_pipeline: wgpu::ComputePipeline,
    inverse_ola_pipeline: wgpu::ComputePipeline,
}

impl StftGpuKernel {
    /// Create a new kernel by compiling both WGSL shaders and building all pipelines.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-stft-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stft.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-stft-wgpu BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(std::mem::size_of::<StftParams>() as u64)
                                .expect("nonzero size"),
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-stft-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-stft-wgpu forward pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("stft_forward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let inverse_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-stft-wgpu inverse shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stft_inverse.wgsl").into()),
        });
        let inverse_frames_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-stft-wgpu inverse frames pipeline"),
                layout: Some(&pipeline_layout),
                module: &inverse_shader,
                entry_point: Some("stft_inverse_frames"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let inverse_ola_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-stft-wgpu inverse ola pipeline"),
                layout: Some(&pipeline_layout),
                module: &inverse_shader,
                entry_point: Some("stft_inverse_ola"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu params"),
            contents: bytemuck::bytes_of(&StftParams {
                signal_len: 0,
                frame_len: 0,
                hop_len: 0,
                frame_count: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            bind_group_layout,
            params_buffer,
            forward_pipeline,
            inverse_frames_pipeline,
            inverse_ola_pipeline,
        }
    }

    /// Execute the forward STFT. Returns interleaved complex values.
    pub fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        signal: &[f32],
        frame_len: usize,
        hop_len: usize,
        frame_count: usize,
    ) -> WgpuResult<Vec<Complex32>> {
        let total = frame_count * frame_len;
        let out_b = (total * std::mem::size_of::<ComplexPod>()) as u64;
        let sig_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu signal"),
            contents: bytemuck::cast_slice(signal),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu output"),
            size: out_b,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu staging"),
            size: out_b,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&StftParams {
                signal_len: signal.len() as u32,
                frame_len: frame_len as u32,
                hop_len: hop_len as u32,
                frame_count: frame_count as u32,
            }),
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sig_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-stft-wgpu encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu forward pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_count(total as u32), 1, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &stg_buf, 0, out_b);
        queue.submit(std::iter::once(enc.finish()));

        let slice = stg_buf.slice(..out_b);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Err(WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })
            }
            Err(e) => {
                return Err(WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })
            }
        }
        let output = {
            let m = slice.get_mapped_range();
            bytemuck::cast_slice::<_, ComplexPod>(&m)
                .iter()
                .map(|p| Complex32::new(p.re, p.im))
                .collect()
        };
        stg_buf.unmap();
        Ok(output)
    }

    /// Execute the inverse STFT via two-pass WOLA reconstruction.
    ///
    /// Pass 1 (stft_inverse_frames): per-frame IDFT with Hann window.
    ///   `frame_data[m·N + j] = (1/N) · Re{ Σ_k X[m,k]·exp(+2πi·k·j/N) } · w[j]`
    ///
    /// Pass 2 (stft_inverse_ola): weighted overlap-add.
    ///   `y[n] = Σ_m frame_data[m·N + (n − start_m)] / Σ_m w[n − start_m]²`
    ///
    /// Formal basis: WOLA identity (Allen–Rabiner 1977, Theorem 1).
    ///
    /// # Invariants
    /// - `spectrum.len() == frame_count * frame_len`
    /// - Returned `Vec<f32>` has length `signal_len`.
    pub fn execute_inverse(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spectrum: &[Complex32],
        frame_len: usize,
        hop_len: usize,
        frame_count: usize,
        signal_len: usize,
    ) -> WgpuResult<Vec<f32>> {
        // Step 1: Interleave spectrum into f32 pairs.
        // spectrum_flat[2i]   = spectrum[i].re
        // spectrum_flat[2i+1] = spectrum[i].im
        let mut spectrum_flat: Vec<f32> = Vec::with_capacity(2 * spectrum.len());
        for c in spectrum {
            spectrum_flat.push(c.re);
            spectrum_flat.push(c.im);
        }

        // Step 2: Allocate buffers.
        // spectrum_buf: initialized from interleaved pairs via create_buffer_init.
        let spectrum_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu inverse spectrum"),
            contents: bytemuck::cast_slice(&spectrum_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // frame_data_buf: Pass 1 writes here (binding 1 = read_write);
        // Pass 2 reads here (binding 0 = read_only). Both share STORAGE usage.
        let frame_data_size = (frame_count * frame_len * std::mem::size_of::<f32>()) as u64;
        let frame_data_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu frame data"),
            size: frame_data_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // signal_buf: Pass 2 writes here (each invocation writes exactly one output[n]).
        let signal_size = (signal_len * std::mem::size_of::<f32>()) as u64;
        let signal_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu inverse signal"),
            size: signal_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // staging: host-read-back buffer.
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu inverse staging"),
            size: signal_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Step 3: Write uniform params into the shared params buffer.
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&StftParams {
                signal_len: signal_len as u32,
                frame_len: frame_len as u32,
                hop_len: hop_len as u32,
                frame_count: frame_count as u32,
            }),
        );

        // Step 5: Build bind groups.
        // frames_bg: Pass 1 reads spectrum at binding 0, writes frame_data at binding 1.
        let frames_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu frames bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spectrum_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frame_data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });
        // ola_bg: Pass 2 reads frame_data at binding 0, writes signal at binding 1.
        let ola_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu ola bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: signal_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        // Step 6: Encode both passes + readback copy in one command buffer.
        // Ending a compute pass and beginning a new one provides an implicit
        // storage-buffer memory barrier: Pass 2 observes all writes from Pass 1.
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-stft-wgpu inverse encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu inverse frames pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inverse_frames_pipeline);
            pass.set_bind_group(0, &frames_bg, &[]);
            pass.dispatch_workgroups(dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu inverse ola pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inverse_ola_pipeline);
            pass.set_bind_group(0, &ola_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(signal_len as u32), 1, 1);
        }
        enc.copy_buffer_to_buffer(&signal_buf, 0, &staging, 0, signal_size);

        // Step 7: Submit, poll for completion, map staging buffer, collect output.
        queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..signal_size);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Err(WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })
            }
            Err(e) => {
                return Err(WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })
            }
        }
        let output = {
            let m = slice.get_mapped_range();
            bytemuck::cast_slice::<_, f32>(&m).to_vec()
        };
        staging.unmap();
        Ok(output)
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
