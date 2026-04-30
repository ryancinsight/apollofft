//! GPU execution for the forward discrete Radon projection, adjoint backprojection,
//! and ramp-filtered backprojection (FBP).

use std::sync::mpsc;

use apollo_radon::ramp_filter_projection;
use bytemuck::{Pod, Zeroable};
use ndarray::Array2;
use wgpu::util::DeviceExt;

use crate::application::plan::RadonWgpuPlan;
use crate::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RadonParams {
    rows: u32,
    cols: u32,
    angle_count: u32,
    detector_count: u32,
    detector_spacing: f32,
    _padding: [u32; 3],
}

/// Cached WGPU kernel state for repeated Radon forward, backprojection,
/// and filtered backprojection dispatches.
#[derive(Debug)]
pub struct RadonGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    forward_pipeline: wgpu::ComputePipeline,
    backproject_pipeline: wgpu::ComputePipeline,
    fbp_filter_pipeline: wgpu::ComputePipeline,
}

impl RadonGpuKernel {
    /// Compile shader state and allocate the uniform parameter buffer.
    ///
    /// Forward, backprojection, and FBP-filter pipelines share one `BindGroupLayout`
    /// (read-only, read-only, read_write, uniform).
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let forward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-radon-wgpu forward shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/radon.wgsl").into()),
        });
        let backproject_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-radon-wgpu backproject shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/radon_backproject.wgsl").into()),
        });
        let fbp_filter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-radon-wgpu FBP filter shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/radon_fbp_filter.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-radon-wgpu bind group layout"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, true),
                storage_layout_entry(2, false),
                uniform_layout_entry(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-radon-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-radon-wgpu forward pipeline"),
            layout: Some(&pipeline_layout),
            module: &forward_shader,
            entry_point: Some("radon_forward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let backproject_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-radon-wgpu backproject pipeline"),
                layout: Some(&pipeline_layout),
                module: &backproject_shader,
                entry_point: Some("radon_backproject"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fbp_filter_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-radon-wgpu FBP filter pipeline"),
                layout: Some(&pipeline_layout),
                module: &fbp_filter_shader,
                entry_point: Some("radon_fbp_filter"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-radon-wgpu params"),
            contents: bytemuck::bytes_of(&RadonParams {
                rows: 0,
                cols: 0,
                angle_count: 0,
                detector_count: 0,
                detector_spacing: 0.0,
                _padding: [0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            bind_group_layout,
            params_buffer,
            forward_pipeline,
            backproject_pipeline,
            fbp_filter_pipeline,
        }
    }

    /// Execute the GPU adjoint backprojection.
    ///
    /// Adjoint operator of the forward Radon map (Natterer 2001, §II.2).
    /// Input: `sinogram` of shape `(angle_count, detector_count)`.
    /// Output: image of shape `(rows, cols)`.
    pub fn execute_backproject(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plan: &RadonWgpuPlan,
        sinogram: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        let sinogram_flat: Vec<f32> = sinogram.iter().copied().collect();
        let sinogram_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-radon-wgpu backproject sinogram"),
            contents: bytemuck::cast_slice(&sinogram_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let angle_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-radon-wgpu backproject angles"),
            contents: bytemuck::cast_slice(angles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let rows = plan.rows();
        let cols = plan.cols();
        let image_size = (rows * cols * std::mem::size_of::<f32>()) as u64;
        let image_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-radon-wgpu backproject image"),
            size: image_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-radon-wgpu backproject staging"),
            size: image_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&RadonParams {
                rows: rows as u32,
                cols: cols as u32,
                angle_count: plan.angle_count() as u32,
                detector_count: plan.detector_count() as u32,
                detector_spacing: plan.detector_spacing() as f32,
                _padding: [0; 3],
            }),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-radon-wgpu backproject bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sinogram_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: angle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: image_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });
        let total_pixels = (rows * cols) as u32;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-radon-wgpu backproject encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-radon-wgpu backproject pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backproject_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(total_pixels), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&image_buf, 0, &staging, 0, image_size);
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
        let values = {
            let mapped = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&mapped).to_vec()
        };
        staging.unmap();
        Array2::from_shape_vec((rows, cols), values).map_err(|_| WgpuError::BufferMapFailed {
            message: "failed to reshape backprojection readback".to_string(),
        })
    }

    /// Execute the forward projection.
    pub fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plan: &RadonWgpuPlan,
        image: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        let image_data: Vec<f32> = image.iter().copied().collect();
        let image_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-radon-wgpu image"),
            contents: bytemuck::cast_slice(&image_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let angle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-radon-wgpu angles"),
            contents: bytemuck::cast_slice(angles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let sinogram_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-radon-wgpu sinogram"),
            size: (plan.angle_count() * plan.detector_count() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-radon-wgpu staging"),
            size: (plan.angle_count() * plan.detector_count() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&RadonParams {
                rows: plan.rows() as u32,
                cols: plan.cols() as u32,
                angle_count: plan.angle_count() as u32,
                detector_count: plan.detector_count() as u32,
                detector_spacing: plan.detector_spacing() as f32,
                _padding: [0; 3],
            }),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-radon-wgpu bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: image_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: angle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sinogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let total_outputs = (plan.angle_count() * plan.detector_count()) as u32;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-radon-wgpu encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-radon-wgpu forward pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(total_outputs), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &sinogram_buffer,
            0,
            &staging,
            0,
            (plan.angle_count() * plan.detector_count() * std::mem::size_of::<f32>()) as u64,
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

        let values = {
            let mapped = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&mapped).to_vec()
        };
        staging.unmap();
        Array2::from_shape_vec((plan.angle_count(), plan.detector_count()), values).map_err(|_| {
            WgpuError::BufferMapFailed {
                message: "failed to reshape sinogram readback".to_string(),
            }
        })
    }

    /// Execute the GPU ramp-filtered backprojection (FBP).
    ///
    /// ## Algorithm
    ///
    /// Pass 1 (`radon_fbp_filter`): applies the Ram-Lak ramp filter to each
    /// projection row via circular convolution with the filter impulse response `h`.
    /// `h` is computed on the host as `IFFT(R)` where
    /// `R[k] = 2π·|signed_k| / (N·Δ)` (Bracewell & Riddle 1967),
    /// matching the CPU `ramp_filter_projection` kernel exactly.
    ///
    /// Pass 2 (`radon_backproject`): adjoint backprojection of the filtered sinogram.
    ///
    /// Post-processing: host-side multiplication by `π / angle_count` (uniform-angle
    /// normalization for the discrete FBP integral approximation).
    ///
    /// Both passes are encoded in a single `CommandEncoder` and submitted once.
    /// The implicit WebGPU per-pass memory barrier guarantees that pass-1 writes to
    /// `filtered_buf` are visible to pass-2 before it begins.
    pub fn execute_filtered_backproject(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plan: &RadonWgpuPlan,
        sinogram: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        let angle_count = plan.angle_count();
        let detector_count = plan.detector_count();
        let rows = plan.rows();
        let cols = plan.cols();

        // Host-side ramp kernel: h = IFFT(R), computed via the CPU ramp_filter_projection
        // of the unit impulse — gives the filter's impulse response exactly.
        let h = compute_ramp_kernel_f32(detector_count, plan.detector_spacing());

        // Write geometry params to the shared uniform buffer.
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&RadonParams {
                rows: rows as u32,
                cols: cols as u32,
                angle_count: angle_count as u32,
                detector_count: detector_count as u32,
                detector_spacing: plan.detector_spacing() as f32,
                _padding: [0; 3],
            }),
        );

        // Input sinogram (pass 1, binding 0).
        let sinogram_flat: Vec<f32> = sinogram.iter().copied().collect();
        let sinogram_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-radon-wgpu fbp sinogram"),
            contents: bytemuck::cast_slice(&sinogram_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Ramp filter kernel h (pass 1, binding 1).
        let filter_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-radon-wgpu fbp filter kernel"),
            contents: bytemuck::cast_slice(&h),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Filtered sinogram buffer (pass 1 output, pass 2 input).
        let sinogram_size = (angle_count * detector_count * std::mem::size_of::<f32>()) as u64;
        let filtered_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-radon-wgpu fbp filtered sinogram"),
            size: sinogram_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Angles buffer (pass 2, binding 1).
        let angle_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-radon-wgpu fbp angles"),
            contents: bytemuck::cast_slice(angles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Image output buffer (pass 2 output).
        let image_size = (rows * cols * std::mem::size_of::<f32>()) as u64;
        let image_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-radon-wgpu fbp image"),
            size: image_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffer for readback.
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-radon-wgpu fbp staging"),
            size: image_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pass 1 bind group: sinogram (read) | filter_kernel (read) | filtered_sinogram (rw) | params.
        let fbp_filter_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-radon-wgpu fbp filter bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sinogram_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: filter_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: filtered_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        // Pass 2 bind group: filtered_sinogram (read) | angles (read) | image (rw) | params.
        let backproject_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-radon-wgpu fbp backproject bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: filtered_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: angle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: image_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let total_sino = (angle_count * detector_count) as u32;
        let total_pixels = (rows * cols) as u32;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-radon-wgpu fbp encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-radon-wgpu fbp filter pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fbp_filter_pipeline);
            pass.set_bind_group(0, &fbp_filter_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(total_sino), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-radon-wgpu fbp backproject pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backproject_pipeline);
            pass.set_bind_group(0, &backproject_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(total_pixels), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&image_buf, 0, &staging_buf, 0, image_size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging_buf.slice(..);
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
        let raw: Vec<f32> = {
            let mapped = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&mapped).to_vec()
        };
        staging_buf.unmap();

        // Normalize: π / angle_count (uniform-angle FBP integral approximation).
        let norm = std::f32::consts::PI / angle_count as f32;
        Array2::from_shape_vec((rows, cols), raw.into_iter().map(|v| v * norm).collect()).map_err(
            |_| WgpuError::BufferMapFailed {
                message: "failed to reshape FBP readback".to_string(),
            },
        )
    }
}

/// Compute the ramp filter impulse response for `detector_count` bins at spacing `detector_spacing`.
///
/// Uses `apollo_radon::ramp_filter_projection` applied to the unit impulse
/// `[1, 0, 0, ..., 0]`, which by linearity of convolution equals the filter's
/// impulse response h = IFFT(R) where R[k] = 2π·|signed_k| / (N·detector_spacing).
///
/// Result is cast to f32 for GPU upload.
fn compute_ramp_kernel_f32(detector_count: usize, detector_spacing: f64) -> Vec<f32> {
    let impulse: Vec<f64> = std::iter::once(1.0_f64)
        .chain(std::iter::repeat_n(
            0.0_f64,
            detector_count.saturating_sub(1),
        ))
        .collect();
    ramp_filter_projection(&impulse, detector_spacing)
        .iter()
        .map(|&v| v as f32)
        .collect()
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
                std::num::NonZeroU64::new(std::mem::size_of::<RadonParams>() as u64)
                    .expect("nonzero uniform size"),
            ),
        },
        count: None,
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
