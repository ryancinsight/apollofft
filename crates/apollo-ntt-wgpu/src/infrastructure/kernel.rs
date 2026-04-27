//! GPU execution for direct modular NTT evaluation.
//!
//! For a prime modulus `q`, transform length `n`, and primitive `n`-th root
//! `omega`, the forward NTT computes `X[k] = sum_j x[j] omega^(j k) (mod q)`.
//! The inverse computes `x[j] = n^{-1} sum_k X[k] omega^(-j k) (mod q)`.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: u32 = 64;

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
/// Execution mode for the direct modular transform.
pub enum NttMode {
    /// Forward NTT.
    Forward = 0,
    /// Inverse NTT.
    Inverse = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct NttParams {
    len: u32,
    modulus: u32,
    root: u32,
    mode: u32,
}

/// Cached WGPU kernel state for repeated NTT dispatches.
#[derive(Debug)]
pub struct NttGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
}

/// Reusable device and host storage for repeated NTT dispatches at one length.
///
/// The buffers preserve the direct NTT contract while moving allocation out of
/// the hot execution path. The input scratch stores canonical `u32` residues,
/// because the current WGPU surface is explicitly bounded to 32-bit moduli.
#[derive(Debug)]
pub struct NttGpuBuffers {
    len: usize,
    input_residues: Vec<u32>,
    output_residues: Vec<u64>,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl NttGpuBuffers {
    /// Return the logical transform length these buffers support.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Return whether these buffers carry zero length.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl NttGpuKernel {
    /// Compile shader state and allocate the uniform parameter buffer.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-ntt-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ntt.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-ntt-wgpu bind group layout"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, false),
                uniform_layout_entry(2),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-ntt-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-ntt-wgpu pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("ntt_transform"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-ntt-wgpu params"),
            contents: bytemuck::bytes_of(&NttParams {
                len: 0,
                modulus: 0,
                root: 0,
                mode: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            bind_group_layout,
            params_buffer,
            pipeline,
        }
    }

    /// Allocate reusable buffers for one direct NTT length.
    pub fn create_buffers(&self, device: &wgpu::Device, len: usize) -> WgpuResult<NttGpuBuffers> {
        if len == 0 {
            return Err(WgpuError::InvalidBufferLength { len });
        }
        let byte_len = buffer_byte_len(len);
        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-ntt-wgpu reusable input"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-ntt-wgpu reusable output"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-ntt-wgpu reusable staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-ntt-wgpu reusable bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });
        Ok(NttGpuBuffers {
            len,
            input_residues: vec![0; len],
            output_residues: vec![0; len],
            input_buffer,
            output_buffer,
            staging_buffer,
            bind_group,
        })
    }

    /// Execute the direct modular transform.
    pub fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[u64],
        len: usize,
        modulus: u64,
        root: u64,
        mode: NttMode,
    ) -> WgpuResult<Vec<u64>> {
        let mut buffers = self.create_buffers(device, len)?;
        self.execute_with_buffers(device, queue, input, modulus, root, mode, &mut buffers)?;
        Ok(buffers.output_residues.clone())
    }

    /// Execute the direct modular transform with caller-owned reusable buffers.
    pub fn execute_with_buffers(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[u64],
        modulus: u64,
        root: u64,
        mode: NttMode,
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        let len = buffers.len;
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        for (slot, &value) in buffers.input_residues.iter_mut().zip(input.iter()) {
            *slot = (value % modulus) as u32;
        }
        queue.write_buffer(
            &buffers.input_buffer,
            0,
            bytemuck::cast_slice(&buffers.input_residues),
        );
        self.dispatch_with_bound_buffers(device, queue, len, modulus, root, mode, buffers)?;
        Ok(())
    }

    /// Execute with caller-owned reusable buffers from exact `u32` residue storage.
    pub fn execute_quantized_with_buffers(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[u32],
        modulus: u64,
        root: u64,
        mode: NttMode,
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        let len = buffers.len;
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        for (slot, &value) in buffers.input_residues.iter_mut().zip(input.iter()) {
            *slot = (u64::from(value) % modulus) as u32;
        }
        queue.write_buffer(
            &buffers.input_buffer,
            0,
            bytemuck::cast_slice(&buffers.input_residues),
        );
        self.dispatch_with_bound_buffers(device, queue, len, modulus, root, mode, buffers)?;
        Ok(())
    }

    fn dispatch_with_bound_buffers(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        len: usize,
        modulus: u64,
        root: u64,
        mode: NttMode,
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&NttParams {
                len: len as u32,
                modulus: modulus as u32,
                root: root as u32,
                mode: mode as u32,
            }),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-ntt-wgpu encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-ntt-wgpu transform pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &buffers.bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(len as u32), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &buffers.output_buffer,
            0,
            &buffers.staging_buffer,
            0,
            buffer_byte_len(len),
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = buffers.staging_buffer.slice(..);
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

        {
            let mapped = slice.get_mapped_range();
            let values: &[u32] = bytemuck::cast_slice(&mapped);
            for (slot, &value) in buffers.output_residues.iter_mut().zip(values.iter()) {
                *slot = u64::from(value);
            }
        }
        buffers.staging_buffer.unmap();
        Ok(())
    }

    /// Return the last readback values written by `execute_with_buffers`.
    #[must_use]
    pub fn buffer_output<'a>(&self, buffers: &'a NttGpuBuffers) -> &'a [u64] {
        &buffers.output_residues
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
                std::num::NonZeroU64::new(std::mem::size_of::<NttParams>() as u64)
                    .expect("nonzero uniform size"),
            ),
        },
        count: None,
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}

fn buffer_byte_len(len: usize) -> u64 {
    (len * std::mem::size_of::<u32>()) as u64
}
