//! GPU execution kernel for the unitary discrete fractional Fourier transform.
//!
//! Implements DFrFT_a(x) = V · diag(exp(−i·a·k·π/2)) · V^T · x
//! using the Grünbaum eigenbasis (Candan 2000).
//! Three submissions guarantee cross-workgroup storage ordering.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use num_complex::Complex32;
use wgpu::util::DeviceExt;

use apollo_frft::GrunbaumBasis;

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
struct UnitaryParams {
    len: u32,
    step: u32,
    order: f32,
    _pad: u32,
}

/// Cached WGPU pipeline and bind group layout for repeated unitary FrFT dispatches.
///
/// The unitary DFrFT is DFrFT_a(x) = V · diag(exp(−i·a·k·π/2)) · V^T · x where V is
/// the N×N real orthogonal eigenvector matrix from the Grünbaum commuting matrix.
/// Three separate command encoder submissions enforce cross-workgroup storage ordering.
#[derive(Debug)]
pub struct UnitaryFrftGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl UnitaryFrftGpuKernel {
    /// Compile the unitary FrFT shader and cache the pipeline and bind group layout.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-frft-wgpu unitary shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/frft_unitary.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-frft-wgpu unitary bgl"),
            entries: &[
                // binding 0: input_data — read-only storage (complex signal)
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
                // binding 1: v_mat — read-only storage (f32 column-major eigenvectors)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: intermediate — read-write storage (eigenbasis coefficients)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: output_data — read-write storage (transform result)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: params — uniform (len, step, order)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(std::mem::size_of::<UnitaryParams>() as u64)
                                .expect("nonzero uniform size"),
                        ),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-frft-wgpu unitary pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-frft-wgpu unitary pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("unitary_step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            bind_group_layout,
            pipeline,
        }
    }

    /// Execute the three-pass unitary DFrFT on GPU.
    ///
    /// Pass 0: c[k] = Σ_j V[j,k] · x[j]          (V^T · x)
    /// Pass 1: c[k] *= exp(−i · order · k · π/2)   (phase multiply)
    /// Pass 2: y[j] = Σ_k V[j,k] · c[k]            (V · c)
    ///
    /// Each pass is submitted as a separate command encoder and polled to
    /// completion before the next is submitted, guaranteeing that writes
    /// from one pass are visible to all workgroups in the subsequent pass.
    ///
    /// `input` must be non-empty; validation is the caller's responsibility.
    pub fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[Complex32],
        order: f32,
    ) -> WgpuResult<Vec<Complex32>> {
        let n = input.len();

        // Compute Grünbaum eigenbasis CPU-side (O(N³)).
        let basis = GrunbaumBasis::new(n);
        let v = basis.eigenvectors();

        // nalgebra DMatrix<f64> uses column-major storage.
        // as_slice() layout: [v[0,0], v[1,0], …, v[n-1,0], v[0,1], …]
        // which satisfies v_flat[row + col*n] = v[(row, col)]. ✓
        let v_flat: Vec<f32> = v.as_slice().iter().map(|&x| x as f32).collect();

        let complex_byte_len = (n * std::mem::size_of::<ComplexPod>()) as u64;

        // Build input pod buffer.
        let input_pods: Vec<ComplexPod> = input
            .iter()
            .map(|c| ComplexPod { re: c.re, im: c.im })
            .collect();

        let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-frft-wgpu unitary input"),
            contents: bytemuck::cast_slice(&input_pods),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let v_mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-frft-wgpu unitary v_mat"),
            contents: bytemuck::cast_slice(&v_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // intermediate: written by step 0, updated by step 1, read by step 2.
        let intermediate_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-frft-wgpu unitary intermediate"),
            size: complex_byte_len,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // output: written by step 2, copied to staging for readback.
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-frft-wgpu unitary output"),
            size: complex_byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-frft-wgpu unitary staging"),
            size: complex_byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Three separate submissions with intervening polls guarantee that all
        // workgroups from step k have completed before step k+1 begins.
        for step in 0_u32..3_u32 {
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-frft-wgpu unitary params"),
                contents: bytemuck::bytes_of(&UnitaryParams {
                    len: n as u32,
                    step,
                    order,
                    _pad: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-frft-wgpu unitary bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: v_mat_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: intermediate_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-frft-wgpu unitary encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("apollo-frft-wgpu unitary pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(dispatch_count(n as u32), 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));
            let _ = device.poll(wgpu::PollType::Wait);
        }

        // Copy output_buf → staging_buf, then readback.
        let mut copy_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-frft-wgpu unitary copy encoder"),
        });
        copy_encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, complex_byte_len);
        queue.submit(std::iter::once(copy_encoder.finish()));
        let _ = device.poll(wgpu::PollType::Wait);

        let slice = staging_buf.slice(..);
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
            let mapped = slice.get_mapped_range();
            let pods: &[ComplexPod] = bytemuck::cast_slice(&mapped);
            pods.iter().map(|p| Complex32::new(p.re, p.im)).collect()
        };
        staging_buf.unmap();
        Ok(output)
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
