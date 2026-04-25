//! Exact direct NUFFT Type-1 WGPU kernels.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use num_complex::Complex32;
use wgpu::util::DeviceExt;

use crate::domain::error::{NufftWgpuError, NufftWgpuResult};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ComplexPod {
    re: f32,
    im: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Position3Pod {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct NufftParams {
    n0: u32,
    n1: u32,
    n2: u32,
    sample_count: u32,
    l0: f32,
    l1: f32,
    l2: f32,
    _pad: f32,
}

/// Cached WGPU state for direct Type-1 NUFFT dispatches.
#[derive(Debug)]
pub struct NufftGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    type1_1d_pipeline: wgpu::ComputePipeline,
    type2_1d_pipeline: wgpu::ComputePipeline,
    type1_3d_pipeline: wgpu::ComputePipeline,
    type2_3d_pipeline: wgpu::ComputePipeline,
}

impl NufftGpuKernel {
    /// Compile shader state and allocate the uniform parameter buffer.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-nufft-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/nufft.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-nufft-wgpu bind group layout"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, true),
                storage_layout_entry(2, false),
                uniform_layout_entry(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-nufft-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let type1_1d_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-nufft-wgpu type1 1d pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("nufft_type1_1d"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let type1_3d_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-nufft-wgpu type1 3d pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("nufft_type1_3d"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let type2_1d_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-nufft-wgpu type2 1d pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("nufft_type2_1d"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let type2_3d_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-nufft-wgpu type2 3d pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("nufft_type2_3d"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu params"),
            contents: bytemuck::bytes_of(&NufftParams {
                n0: 0,
                n1: 1,
                n2: 1,
                sample_count: 0,
                l0: 1.0,
                l1: 1.0,
                l2: 1.0,
                _pad: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            bind_group_layout,
            params_buffer,
            type1_1d_pipeline,
            type2_1d_pipeline,
            type1_3d_pipeline,
            type2_3d_pipeline,
        }
    }

    /// Execute exact direct Type-1 1D NUFFT.
    pub fn execute_type1_1d(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n: usize,
        length: f32,
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|x| Position3Pod {
                x: *x,
                y: 0.0,
                z: 0.0,
                _pad: 0.0,
            })
            .collect();
        let params = NufftParams {
            n0: n as u32,
            n1: 1,
            n2: 1,
            sample_count: positions.len() as u32,
            l0: length,
            l1: 1.0,
            l2: 1.0,
            _pad: 0.0,
        };
        self.execute(
            device,
            queue,
            &position_data,
            values,
            n,
            params,
            &self.type1_1d_pipeline,
        )
    }

    /// Execute exact direct Type-2 1D NUFFT.
    pub fn execute_type2_1d(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n: usize,
        length: f32,
        fourier_coeffs: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|x| Position3Pod {
                x: *x,
                y: 0.0,
                z: 0.0,
                _pad: 0.0,
            })
            .collect();
        let params = NufftParams {
            n0: n as u32,
            n1: 1,
            n2: 1,
            sample_count: positions.len() as u32,
            l0: length,
            l1: 1.0,
            l2: 1.0,
            _pad: 0.0,
        };
        self.execute(
            device,
            queue,
            &position_data,
            fourier_coeffs,
            positions.len(),
            params,
            &self.type2_1d_pipeline,
        )
    }

    /// Execute exact direct Type-1 3D NUFFT.
    pub fn execute_type1_3d(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: (usize, usize, usize),
        lengths: (f32, f32, f32),
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|(x, y, z)| Position3Pod {
                x: *x,
                y: *y,
                z: *z,
                _pad: 0.0,
            })
            .collect();
        let output_len = shape.0 * shape.1 * shape.2;
        let params = NufftParams {
            n0: shape.0 as u32,
            n1: shape.1 as u32,
            n2: shape.2 as u32,
            sample_count: positions.len() as u32,
            l0: lengths.0,
            l1: lengths.1,
            l2: lengths.2,
            _pad: 0.0,
        };
        self.execute(
            device,
            queue,
            &position_data,
            values,
            output_len,
            params,
            &self.type1_3d_pipeline,
        )
    }

    /// Execute exact direct Type-2 3D NUFFT.
    pub fn execute_type2_3d(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: (usize, usize, usize),
        lengths: (f32, f32, f32),
        modes: &[Complex32],
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|(x, y, z)| Position3Pod {
                x: *x,
                y: *y,
                z: *z,
                _pad: 0.0,
            })
            .collect();
        let params = NufftParams {
            n0: shape.0 as u32,
            n1: shape.1 as u32,
            n2: shape.2 as u32,
            sample_count: positions.len() as u32,
            l0: lengths.0,
            l1: lengths.1,
            l2: lengths.2,
            _pad: 0.0,
        };
        self.execute(
            device,
            queue,
            &position_data,
            modes,
            positions.len(),
            params,
            &self.type2_3d_pipeline,
        )
    }

    fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &[Position3Pod],
        values: &[Complex32],
        output_len: usize,
        params: NufftParams,
        pipeline: &wgpu::ComputePipeline,
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let value_data: Vec<ComplexPod> = values
            .iter()
            .map(|value| ComplexPod {
                re: value.re,
                im: value.im,
            })
            .collect();
        let position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu positions"),
            contents: bytemuck::cast_slice(positions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let value_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu values"),
            contents: bytemuck::cast_slice(&value_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu output"),
            size: (output_len * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu staging"),
            size: (output_len * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: value_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu direct type1 pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(output_len as u32), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging,
            0,
            (output_len * std::mem::size_of::<ComplexPod>()) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait);
        match receiver.recv() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                return Err(NufftWgpuError::BufferMapFailed {
                    message: error.to_string(),
                });
            }
            Err(error) => {
                return Err(NufftWgpuError::BufferMapFailed {
                    message: error.to_string(),
                });
            }
        }

        let output = {
            let mapped = slice.get_mapped_range();
            let pods: &[ComplexPod] = bytemuck::cast_slice(&mapped);
            pods.iter()
                .map(|value| Complex32::new(value.re, value.im))
                .collect()
        };
        staging.unmap();
        Ok(output)
    }
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(
                std::num::NonZeroU64::new(std::mem::size_of::<NufftParams>() as u64)
                    .expect("nonzero uniform size"),
            ),
        },
        count: None,
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
