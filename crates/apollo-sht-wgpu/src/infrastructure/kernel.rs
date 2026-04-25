//! Direct complex spherical harmonic transform WGPU kernels.

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

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ShtParams {
    output_count: u32,
    reduction_count: u32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GridPod {
    pub(crate) cos_theta: f32,
    pub(crate) phi: f32,
    pub(crate) weight: f32,
    pub(crate) _padding: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BasisParams {
    mode_count: u32,
    sample_count: u32,
    max_degree: u32,
    weighted: u32,
    conjugate: u32,
    _padding: [u32; 3],
}

/// Cached WGPU state for direct SHT dispatches.
#[derive(Debug)]
pub struct ShtGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    basis_bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    basis_params_buffer: wgpu::Buffer,
    basis_pipeline: wgpu::ComputePipeline,
    forward_pipeline: wgpu::ComputePipeline,
    inverse_pipeline: wgpu::ComputePipeline,
}

impl ShtGpuKernel {
    /// Compile shader state and allocate the uniform parameter buffer.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-sht-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sht.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-sht-wgpu bind group layout"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, true),
                storage_layout_entry(2, false),
                uniform_layout_entry(3),
            ],
        });
        let basis_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("apollo-sht-wgpu basis bind group layout"),
                entries: &[
                    storage_layout_entry(4, true),
                    storage_layout_entry(5, false),
                    uniform_basis_layout_entry(6),
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-sht-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let basis_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-sht-wgpu basis pipeline layout"),
                bind_group_layouts: &[&basis_bind_group_layout],
                push_constant_ranges: &[],
            });
        let basis_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-sht-wgpu basis pipeline"),
            layout: Some(&basis_pipeline_layout),
            module: &shader,
            entry_point: Some("sht_basis"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-sht-wgpu forward pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sht_forward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let inverse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-sht-wgpu inverse pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sht_inverse"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-sht-wgpu params"),
            contents: bytemuck::bytes_of(&ShtParams {
                output_count: 0,
                reduction_count: 0,
                _padding: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let basis_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-sht-wgpu basis params"),
            contents: bytemuck::bytes_of(&BasisParams {
                mode_count: 0,
                sample_count: 0,
                max_degree: 0,
                weighted: 0,
                conjugate: 0,
                _padding: [0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            bind_group_layout,
            basis_bind_group_layout,
            params_buffer,
            basis_params_buffer,
            basis_pipeline,
            forward_pipeline,
            inverse_pipeline,
        }
    }

    /// Execute forward matrix sums.
    pub(crate) fn execute_forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mode_count: usize,
        sample_count: usize,
        samples: &[Complex32],
        grid: &[GridPod],
    ) -> WgpuResult<Vec<Complex32>> {
        self.execute(
            device,
            queue,
            mode_count,
            sample_count,
            samples,
            grid,
            mode_count,
            sample_count,
            true,
            true,
            &self.forward_pipeline,
        )
    }

    /// Execute inverse matrix sums.
    pub(crate) fn execute_inverse(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sample_count: usize,
        mode_count: usize,
        coefficients: &[Complex32],
        grid: &[GridPod],
    ) -> WgpuResult<Vec<Complex32>> {
        self.execute(
            device,
            queue,
            sample_count,
            mode_count,
            coefficients,
            grid,
            mode_count,
            sample_count,
            false,
            false,
            &self.inverse_pipeline,
        )
    }

    fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_count: usize,
        reduction_count: usize,
        input: &[Complex32],
        grid: &[GridPod],
        mode_count: usize,
        sample_count: usize,
        weighted: bool,
        conjugate: bool,
        pipeline: &wgpu::ComputePipeline,
    ) -> WgpuResult<Vec<Complex32>> {
        let input_data: Vec<ComplexPod> = input
            .iter()
            .map(|value| ComplexPod {
                re: value.re,
                im: value.im,
            })
            .collect();
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-sht-wgpu input"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let basis_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-sht-wgpu basis"),
            contents: &vec![0_u8; mode_count * sample_count * std::mem::size_of::<ComplexPod>()],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let grid_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-sht-wgpu grid"),
            contents: bytemuck::cast_slice(grid),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-sht-wgpu output"),
            size: (output_count * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-sht-wgpu staging"),
            size: (output_count * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&ShtParams {
                output_count: output_count as u32,
                reduction_count: reduction_count as u32,
                _padding: [0; 2],
            }),
        );
        queue.write_buffer(
            &self.basis_params_buffer,
            0,
            bytemuck::bytes_of(&BasisParams {
                mode_count: mode_count as u32,
                sample_count: sample_count as u32,
                max_degree: (integer_sqrt(mode_count) - 1) as u32,
                weighted: u32::from(weighted),
                conjugate: u32::from(conjugate),
                _padding: [0; 3],
            }),
        );
        let basis_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-sht-wgpu basis bind group"),
            layout: &self.basis_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: basis_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.basis_params_buffer.as_entire_binding(),
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-sht-wgpu bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: basis_buffer.as_entire_binding(),
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
            label: Some("apollo-sht-wgpu encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-sht-wgpu basis generation pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.basis_pipeline);
            pass.set_bind_group(0, &basis_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count((mode_count * sample_count) as u32), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-sht-wgpu matrix sum pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(output_count as u32), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging,
            0,
            (output_count * std::mem::size_of::<ComplexPod>()) as u64,
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
                return Err(WgpuError::BufferMapFailed {
                    message: error.to_string(),
                });
            }
            Err(error) => {
                return Err(WgpuError::BufferMapFailed {
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
                std::num::NonZeroU64::new(std::mem::size_of::<ShtParams>() as u64)
                    .expect("nonzero uniform size"),
            ),
        },
        count: None,
    }
}

fn uniform_basis_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(
                std::num::NonZeroU64::new(std::mem::size_of::<BasisParams>() as u64)
                    .expect("nonzero uniform size"),
            ),
        },
        count: None,
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}

fn integer_sqrt(value: usize) -> usize {
    let mut root = 0;
    while (root + 1) * (root + 1) <= value {
        root += 1;
    }
    root
}
