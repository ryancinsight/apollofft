//! GPU execution for the forward Mellin log-frequency spectrum.
//!
//! This kernel mirrors the owning CPU crate:
//! 1. log-resample the positive-domain signal onto the plan scale grid,
//! 2. compute the direct log-frequency Mellin spectrum over that grid.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use num_complex::Complex32;
use wgpu::util::DeviceExt;

use crate::application::plan::MellinWgpuPlan;
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
struct MellinParams {
    signal_len: u32,
    samples: u32,
    signal_min: f32,
    signal_max: f32,
    log_min: f32,
    log_max: f32,
    _padding: [u32; 2],
}

/// Cached WGPU kernel state for repeated Mellin forward dispatches.
#[derive(Debug)]
pub struct MellinGpuKernel {
    resample_layout: wgpu::BindGroupLayout,
    spectrum_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    resample_pipeline: wgpu::ComputePipeline,
    spectrum_pipeline: wgpu::ComputePipeline,
}

impl MellinGpuKernel {
    /// Compile shader state and allocate the uniform parameter buffer.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-mellin-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mellin.wgsl").into()),
        });
        let resample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-mellin-wgpu resample layout"),
            entries: &[
                buffer_layout_entry(0, true),
                buffer_layout_entry(1, false),
                uniform_layout_entry(2),
            ],
        });
        let spectrum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-mellin-wgpu spectrum layout"),
            entries: &[
                buffer_layout_entry(0, true),
                complex_buffer_layout_entry(1),
                uniform_layout_entry(2),
            ],
        });
        let resample_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-mellin-wgpu resample pipeline layout"),
                bind_group_layouts: &[&resample_layout],
                push_constant_ranges: &[],
            });
        let spectrum_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-mellin-wgpu spectrum pipeline layout"),
                bind_group_layouts: &[&spectrum_layout],
                push_constant_ranges: &[],
            });
        let resample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-mellin-wgpu resample pipeline"),
            layout: Some(&resample_pipeline_layout),
            module: &shader,
            entry_point: Some("mellin_resample"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let spectrum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-mellin-wgpu spectrum pipeline"),
            layout: Some(&spectrum_pipeline_layout),
            module: &shader,
            entry_point: Some("mellin_spectrum"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-mellin-wgpu params"),
            contents: bytemuck::bytes_of(&MellinParams {
                signal_len: 0,
                samples: 0,
                signal_min: 0.0,
                signal_max: 0.0,
                log_min: 0.0,
                log_max: 0.0,
                _padding: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            resample_layout,
            spectrum_layout,
            params_buffer,
            resample_pipeline,
            spectrum_pipeline,
        }
    }

    /// Execute the forward Mellin spectrum path.
    pub fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f64,
        signal_max: f64,
    ) -> WgpuResult<Vec<Complex32>> {
        let signal_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-mellin-wgpu input"),
            contents: bytemuck::cast_slice(signal),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let resample_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-mellin-wgpu resample buffer"),
            size: (plan.samples() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spectrum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-mellin-wgpu spectrum buffer"),
            size: (plan.samples() * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-mellin-wgpu staging"),
            size: (plan.samples() * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&MellinParams {
                signal_len: signal.len() as u32,
                samples: plan.samples() as u32,
                signal_min: signal_min as f32,
                signal_max: signal_max as f32,
                log_min: plan.min_scale().ln() as f32,
                log_max: plan.max_scale().ln() as f32,
                _padding: [0; 2],
            }),
        );

        let resample_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-mellin-wgpu resample bind group"),
            layout: &self.resample_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: signal_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });
        let spectrum_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-mellin-wgpu spectrum bind group"),
            layout: &self.spectrum_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spectrum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-mellin-wgpu encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-mellin-wgpu resample pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resample_pipeline);
            pass.set_bind_group(0, &resample_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(plan.samples() as u32), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-mellin-wgpu spectrum pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.spectrum_pipeline);
            pass.set_bind_group(0, &spectrum_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(plan.samples() as u32), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &spectrum_buffer,
            0,
            &staging,
            0,
            (plan.samples() * std::mem::size_of::<ComplexPod>()) as u64,
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

fn buffer_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn complex_buffer_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    buffer_layout_entry(binding, false)
}

fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(
                std::num::NonZeroU64::new(std::mem::size_of::<MellinParams>() as u64)
                    .expect("nonzero uniform size"),
            ),
        },
        count: None,
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
