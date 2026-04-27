//! Direct and fast-gridded NUFFT WGPU kernels.

use std::sync::mpsc;
use std::sync::Arc;

use apollo_fft_wgpu::GpuFft3d;
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FastNufftParams {
    n: u32,
    m: u32,
    sample_count: u32,
    kernel_width: u32,
    length: f32,
    beta: f32,
    i0_beta: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FastNufftParams3D {
    nx: u32,
    ny: u32,
    nz: u32,
    mx: u32,
    my: u32,
    mz: u32,
    sample_count: u32,
    kernel_width: u32,
    lx: f32,
    ly: f32,
    lz: f32,
    beta: f32,
    i0_beta: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

/// Pre-allocated GPU buffers for repeated 1D NUFFT fast-path execution.
///
/// Buffers are sized for a specific transform configuration (`n`, `m`, `max_samples`).
/// Reusing these buffers across calls eliminates per-dispatch GPU buffer creation overhead.
#[allow(dead_code)]
pub struct NufftGpuBuffers1D {
    position_buffer: wgpu::Buffer,
    value_buffer: wgpu::Buffer,
    deconv_buffer: wgpu::Buffer,
    re_buffer: wgpu::Buffer,
    im_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    /// Output grid length (number of Fourier modes).
    n: usize,
    /// Oversampled grid length.
    m: usize,
    /// Maximum non-uniform sample count per dispatch.
    max_samples: usize,
}

impl NufftGpuBuffers1D {
    /// Pre-allocate all GPU buffers for 1D fast-path transforms of the given configuration.
    ///
    /// `n` is the output grid length, `m` is the oversampled grid length, and
    /// `max_samples` is the maximum number of non-uniform samples per dispatch.
    /// Each call to `execute_fast_type1_1d_with_buffers` or
    /// `execute_fast_type2_1d_with_buffers` may use fewer samples; the excess
    /// capacity is unused but not reallocated.
    pub fn new(device: &wgpu::Device, n: usize, m: usize, max_samples: usize) -> Self {
        let upload_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let grid_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let position_size = (max_samples.max(1) * std::mem::size_of::<ComplexPod>()) as u64;
        let position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers1d positions"),
            size: position_size,
            usage: upload_usage,
            mapped_at_creation: false,
        });

        let value_size = (max_samples.max(1) * std::mem::size_of::<ComplexPod>()) as u64;
        let value_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers1d values"),
            size: value_size,
            usage: upload_usage,
            mapped_at_creation: false,
        });

        let deconv_size = (m.max(1) * std::mem::size_of::<ComplexPod>()) as u64;
        let deconv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers1d deconv"),
            size: deconv_size,
            usage: upload_usage,
            mapped_at_creation: false,
        });

        let grid_elem_size = (m.max(1) * std::mem::size_of::<f32>()) as u64;
        let re_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers1d grid re"),
            size: grid_elem_size,
            usage: grid_usage,
            mapped_at_creation: false,
        });
        let im_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers1d grid im"),
            size: grid_elem_size,
            usage: grid_usage,
            mapped_at_creation: false,
        });

        let output_size = (n.max(1) * std::mem::size_of::<ComplexPod>()) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers1d output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_size = output_size;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers1d staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            position_buffer,
            value_buffer,
            deconv_buffer,
            re_buffer,
            im_buffer,
            output_buffer,
            staging_buffer,
            n,
            m,
            max_samples,
        }
    }
}

/// Pre-allocated GPU buffers for repeated 3D NUFFT fast-path execution.
///
/// Buffers are sized for a specific transform configuration (`shape`, `oversampled`,
/// `max_samples`). Reusing these buffers across calls eliminates per-dispatch GPU
/// buffer creation overhead.
#[allow(dead_code)]
pub struct NufftGpuBuffers3D {
    position_buffer: wgpu::Buffer,
    value_buffer: wgpu::Buffer,
    deconv_buffer: wgpu::Buffer,
    re_buffer: wgpu::Buffer,
    im_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    /// Output shape `(nx, ny, nz)`.
    shape: (usize, usize, usize),
    /// Oversampled grid dimensions `(mx, my, mz)`.
    oversampled: (usize, usize, usize),
    /// Maximum non-uniform sample count per dispatch.
    max_samples: usize,
}

/// Diagnostic snapshot of split real/imaginary NUFFT grid state.
///
/// This type is compiled only for tests or the explicit `diagnostics` feature.
/// It records computed GPU grid values after named fast-path checkpoints without
/// changing production dispatch behavior.
#[cfg(any(test, feature = "diagnostics"))]
#[derive(Clone, Debug)]
pub struct NufftGridSnapshot {
    /// Real grid component in row-major storage order.
    pub re: Vec<f32>,
    /// Imaginary grid component in row-major storage order.
    pub im: Vec<f32>,
}

/// Diagnostic checkpoints for fast type-2 NUFFT execution.
#[cfg(any(test, feature = "diagnostics"))]
#[derive(Clone, Debug)]
pub struct NufftType2GridDiagnostics {
    /// Grid state immediately after coefficient load/deconvolution.
    pub after_load: NufftGridSnapshot,
    /// Grid state immediately after inverse FFT and before interpolation.
    pub after_ifft: NufftGridSnapshot,
}

impl NufftGpuBuffers3D {
    /// Pre-allocate all GPU buffers for 3D fast-path transforms of the given configuration.
    ///
    /// `shape` is the output grid dimensions `(nx, ny, nz)`, `oversampled` is the
    /// oversampled grid dimensions `(mx, my, mz)`, and `max_samples` is the maximum
    /// number of non-uniform samples per dispatch.
    pub fn new(
        device: &wgpu::Device,
        shape: (usize, usize, usize),
        oversampled: (usize, usize, usize),
        max_samples: usize,
    ) -> Self {
        let upload_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let grid_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let (nx, ny, nz) = shape;
        let (mx, my, mz) = oversampled;
        let grid_len = mx * my * mz;
        let output_len = nx * ny * nz;

        let position_size = (max_samples.max(1) * std::mem::size_of::<Position3Pod>()) as u64;
        let position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers3d positions"),
            size: position_size,
            usage: upload_usage,
            mapped_at_creation: false,
        });

        let value_size = (max_samples.max(1) * std::mem::size_of::<ComplexPod>()) as u64;
        let value_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers3d values"),
            size: value_size,
            usage: upload_usage,
            mapped_at_creation: false,
        });

        let deconv_size = (grid_len.max(1) * std::mem::size_of::<f32>()) as u64;
        let deconv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers3d deconv"),
            size: deconv_size,
            usage: upload_usage,
            mapped_at_creation: false,
        });

        let grid_elem_size = (grid_len.max(1) * std::mem::size_of::<f32>()) as u64;
        let re_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers3d grid re"),
            size: grid_elem_size,
            usage: grid_usage,
            mapped_at_creation: false,
        });
        let im_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers3d grid im"),
            size: grid_elem_size,
            usage: grid_usage,
            mapped_at_creation: false,
        });

        let output_size = (output_len.max(1) * std::mem::size_of::<ComplexPod>()) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers3d output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_size = output_size;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers3d staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            position_buffer,
            value_buffer,
            deconv_buffer,
            re_buffer,
            im_buffer,
            output_buffer,
            staging_buffer,
            shape,
            oversampled,
            max_samples,
        }
    }
}

/// Cached WGPU state for direct and fast-gridded NUFFT dispatches.
#[derive(Debug)]
pub struct NufftGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    fast_spread_layout: wgpu::BindGroupLayout,
    fast_extract_layout: wgpu::BindGroupLayout,
    fast_3d_spread_layout: wgpu::BindGroupLayout,
    fast_3d_extract_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    fast_params_buffer: wgpu::Buffer,
    type1_1d_pipeline: wgpu::ComputePipeline,
    type2_1d_pipeline: wgpu::ComputePipeline,
    type1_3d_pipeline: wgpu::ComputePipeline,
    type2_3d_pipeline: wgpu::ComputePipeline,
    fast_type1_spread_1d_pipeline: wgpu::ComputePipeline,
    fast_type1_extract_1d_pipeline: wgpu::ComputePipeline,
    fast_type2_load_1d_pipeline: wgpu::ComputePipeline,
    fast_type2_interpolate_1d_pipeline: wgpu::ComputePipeline,
    fast_3d_params_buffer: wgpu::Buffer,
    fast_type1_spread_3d_pipeline: wgpu::ComputePipeline,
    fast_type1_extract_3d_pipeline: wgpu::ComputePipeline,
    fast_type2_load_3d_pipeline: wgpu::ComputePipeline,
    fast_type2_interpolate_3d_pipeline: wgpu::ComputePipeline,
}

impl NufftGpuKernel {
    /// Compile shader state and allocate the uniform parameter buffer.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-nufft-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/nufft.wgsl").into()),
        });
        let fast_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-nufft-wgpu fast gridding shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/nufft_fast_1d.wgsl").into()),
        });
        let fast_3d_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-nufft-wgpu fast 3d gridding shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/nufft_fast_3d.wgsl").into()),
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
        let fast_spread_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast 1d layout"),
                entries: &[
                    storage_layout_entry(0, true),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, false),
                    storage_layout_entry(3, false),
                    storage_layout_entry(4, true),
                    storage_layout_entry(5, false),
                    storage_layout_entry(6, true),
                    uniform_layout_entry(7),
                ],
            });
        let fast_extract_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast 1d layout duplicate"),
                entries: &[
                    storage_layout_entry(0, true),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, false),
                    storage_layout_entry(3, false),
                    storage_layout_entry(4, true),
                    storage_layout_entry(5, false),
                    storage_layout_entry(6, true),
                    uniform_layout_entry(7),
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-nufft-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let fast_spread_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast spread pipeline layout"),
                bind_group_layouts: &[&fast_spread_layout],
                push_constant_ranges: &[],
            });
        let fast_extract_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast extract pipeline layout"),
                bind_group_layouts: &[&fast_extract_layout],
                push_constant_ranges: &[],
            });
        let fast_3d_spread_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast 3d spread layout"),
                entries: &[
                    storage_layout_entry(0, true),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, false),
                    storage_layout_entry(3, false),
                    storage_layout_entry(4, true),
                    storage_layout_entry(5, false),
                    storage_layout_entry(6, true),
                    uniform_layout_entry_3d(7),
                ],
            });
        let fast_3d_extract_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast 3d extract layout"),
                entries: &[
                    storage_layout_entry(0, true),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, false),
                    storage_layout_entry(3, false),
                    storage_layout_entry(4, true),
                    storage_layout_entry(5, false),
                    storage_layout_entry(6, true),
                    uniform_layout_entry_3d(7),
                ],
            });
        let fast_3d_spread_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast 3d spread pipeline layout"),
                bind_group_layouts: &[&fast_3d_spread_layout],
                push_constant_ranges: &[],
            });
        let fast_3d_extract_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast 3d extract pipeline layout"),
                bind_group_layouts: &[&fast_3d_extract_layout],
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
        let fast_type1_spread_1d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 1d spread pipeline"),
                layout: Some(&fast_spread_pipeline_layout),
                module: &fast_shader,
                entry_point: Some("fast_type1_spread_1d"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fast_type1_extract_1d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 1d extract pipeline"),
                layout: Some(&fast_extract_pipeline_layout),
                module: &fast_shader,
                entry_point: Some("fast_type1_extract_1d"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fast_type2_load_1d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 1d load pipeline"),
                layout: Some(&fast_extract_pipeline_layout),
                module: &fast_shader,
                entry_point: Some("fast_type2_load_1d"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fast_type2_interpolate_1d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 1d interpolate pipeline"),
                layout: Some(&fast_spread_pipeline_layout),
                module: &fast_shader,
                entry_point: Some("fast_type2_interpolate_1d"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fast_3d_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu fast 3d params"),
            contents: bytemuck::bytes_of(&FastNufftParams3D {
                nx: 1,
                ny: 1,
                nz: 1,
                mx: 2,
                my: 2,
                mz: 2,
                sample_count: 0,
                kernel_width: 6,
                lx: 1.0,
                ly: 1.0,
                lz: 1.0,
                beta: 1.0,
                i0_beta: 1.0,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let fast_type1_spread_3d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 3d spread pipeline"),
                layout: Some(&fast_3d_spread_pipeline_layout),
                module: &fast_3d_shader,
                entry_point: Some("fast_type1_spread_3d"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fast_type1_extract_3d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 3d extract pipeline"),
                layout: Some(&fast_3d_extract_pipeline_layout),
                module: &fast_3d_shader,
                entry_point: Some("fast_type1_extract_3d"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fast_type2_load_3d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 3d load pipeline"),
                layout: Some(&fast_3d_extract_pipeline_layout),
                module: &fast_3d_shader,
                entry_point: Some("fast_type2_load_3d"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fast_type2_interpolate_3d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 3d interpolate pipeline"),
                layout: Some(&fast_3d_spread_pipeline_layout),
                module: &fast_3d_shader,
                entry_point: Some("fast_type2_interpolate_3d"),
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
        let fast_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu fast params"),
            contents: bytemuck::bytes_of(&FastNufftParams {
                n: 0,
                m: 0,
                sample_count: 0,
                kernel_width: 0,
                length: 1.0,
                beta: 1.0,
                i0_beta: 1.0,
                _pad: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            bind_group_layout,
            fast_spread_layout,
            fast_extract_layout,
            fast_3d_spread_layout,
            fast_3d_extract_layout,
            params_buffer,
            fast_params_buffer,
            type1_1d_pipeline,
            type2_1d_pipeline,
            type1_3d_pipeline,
            type2_3d_pipeline,
            fast_type1_spread_1d_pipeline,
            fast_type1_extract_1d_pipeline,
            fast_type2_load_1d_pipeline,
            fast_type2_interpolate_1d_pipeline,
            fast_3d_params_buffer,
            fast_type1_spread_3d_pipeline,
            fast_type1_extract_3d_pipeline,
            fast_type2_load_3d_pipeline,
            fast_type2_interpolate_3d_pipeline,
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

    /// Execute fast gridded Type-1 1D NUFFT with GPU spreading, FFT, and deconvolution.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type1_1d(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        n: usize,
        oversampled_len: usize,
        kernel_width: usize,
        length: f32,
        beta: f32,
        i0_beta: f32,
        deconv: &[f32],
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let position_data = positions_to_complex_pods_1d(positions);
        let value_data = complex_to_pods(values);
        let deconv_data = real_to_complex_pods(deconv);
        let position_buffer =
            storage_buffer(device, "apollo-nufft-wgpu fast positions", &position_data);
        let value_buffer = storage_buffer(device, "apollo-nufft-wgpu fast values", &value_data);
        let deconv_buffer = storage_buffer(device, "apollo-nufft-wgpu fast deconv", &deconv_data);
        let coefficient_buffer =
            placeholder_storage_buffer(device, "apollo-nufft-wgpu fast empty coefficients", n);
        let (re_buffer, im_buffer) = split_grid_buffers(device, oversampled_len);
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu fast type1 output"),
            size: (n * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = FastNufftParams {
            n: n as u32,
            m: oversampled_len as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            length,
            beta,
            i0_beta,
            _pad: 0.0,
        };
        queue.write_buffer(&self.fast_params_buffer, 0, bytemuck::bytes_of(&params));

        let spread_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type1 spread bg"),
            layout: &self.fast_spread_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &value_buffer),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coefficient_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast type1 spread encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 spread pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_spread_1d_pipeline);
            pass.set_bind_group(0, &spread_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(oversampled_len as u32), 1, 1);
        }

        let fft = GpuFft3d::new(Arc::clone(device), Arc::clone(queue), oversampled_len, 1, 1)
            .map_err(|_| NufftWgpuError::InvalidPlan {
                message: "oversampled FFT plan is invalid for WGPU execution",
            })?;
        fft.encode_forward_split(&mut encoder, &re_buffer, &im_buffer);

        let extract_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type1 extract bg"),
            layout: &self.fast_extract_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &value_buffer),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coefficient_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 extract pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_extract_1d_pipeline);
            pass.set_bind_group(0, &extract_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(n as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer(device, queue, &output_buffer, n)
    }

    /// Execute fast gridded Type-2 1D NUFFT with GPU deconvolution, FFT, and interpolation.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type2_1d(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        n: usize,
        oversampled_len: usize,
        kernel_width: usize,
        length: f32,
        beta: f32,
        i0_beta: f32,
        deconv: &[f32],
        coefficients: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let coefficient_data = complex_to_pods(coefficients);
        let position_data = positions_to_complex_pods_1d(positions);
        // encode_inverse_split applies a normalized IFFT that divides by oversampled_len.
        // The CPU type2_into path multiplies by oversampled_len after its normalized IFFT
        // to recover the unnormalized IDFT required by the KB interpolation kernel.
        // Embedding that factor here (deconv × m) achieves the same effect on the GPU:
        //   grid_load = f[k] × (m × deconv[k])
        //   after GPU IFFT (÷m): g[j] = Σ_k f[k]×deconv[k]×exp(2πi jk'/m)  [unnormalized IDFT]
        let deconv_data = real_to_complex_pods_scaled(deconv, oversampled_len as f32);
        let coefficients_buffer = storage_buffer(
            device,
            "apollo-nufft-wgpu fast coefficients",
            &coefficient_data,
        );
        let position_buffer =
            storage_buffer(device, "apollo-nufft-wgpu fast positions", &position_data);
        let deconv_buffer = storage_buffer(device, "apollo-nufft-wgpu fast deconv", &deconv_data);
        let (re_buffer, im_buffer) = split_grid_buffers(device, oversampled_len);
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 output"),
            size: (positions.len() * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = FastNufftParams {
            n: n as u32,
            m: oversampled_len as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            length,
            beta,
            i0_beta,
            _pad: 0.0,
        };
        queue.write_buffer(&self.fast_params_buffer, 0, bytemuck::bytes_of(&params));

        let load_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 load bg"),
            layout: &self.fast_extract_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &coefficients_buffer),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coefficients_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 load encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 load pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_load_1d_pipeline);
            pass.set_bind_group(0, &load_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(oversampled_len as u32), 1, 1);
        }

        let fft = GpuFft3d::new(Arc::clone(device), Arc::clone(queue), oversampled_len, 1, 1)
            .map_err(|_| NufftWgpuError::InvalidPlan {
                message: "oversampled FFT plan is invalid for WGPU execution",
            })?;
        fft.encode_inverse_split(&mut encoder, &re_buffer, &im_buffer);

        let interp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 interpolate bg"),
            layout: &self.fast_spread_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &coefficients_buffer),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coefficients_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 interpolate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_interpolate_1d_pipeline);
            pass.set_bind_group(0, &interp_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(positions.len() as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer(device, queue, &output_buffer, positions.len())
    }

    /// Execute fast gridded Type-1 3D NUFFT with GPU separable spreading, 3D FFT, and deconvolution.
    ///
    /// Algorithm:
    /// 1. Spread: one thread per oversampled grid cell, accumulates KB-weighted sample contributions.
    /// 2. 3D FFT: uses `GpuFft3d` on the oversampled split grid (mx, my, mz).
    /// 3. Extract: one thread per output mode, reads FFT'd grid at mapped index, applies deconvolution.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type1_3d(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        shape: (usize, usize, usize),
        oversampled: (usize, usize, usize),
        kernel_width: usize,
        lengths: (f32, f32, f32),
        beta: f32,
        i0_beta: f32,
        deconv_xyz: &[f32],
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let (nx, ny, nz) = shape;
        let (mx, my, mz) = oversampled;
        let (lx, ly, lz) = lengths;
        let grid_len = mx * my * mz;
        let output_len = nx * ny * nz;

        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|(x, y, z)| Position3Pod {
                x: *x,
                y: *y,
                z: *z,
                _pad: 0.0,
            })
            .collect();
        let value_data = complex_to_pods(values);

        let position_buffer =
            storage_buffer(device, "apollo-nufft-wgpu fast3d positions", &position_data);
        let value_buffer = storage_buffer(device, "apollo-nufft-wgpu fast3d values", &value_data);
        let deconv_buffer = storage_buffer(device, "apollo-nufft-wgpu fast3d deconv", deconv_xyz);
        let coeff_buffer = placeholder_storage_buffer(
            device,
            "apollo-nufft-wgpu fast3d coeffs placeholder",
            output_len,
        );
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type1 output"),
            size: (output_len * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (re_buffer, im_buffer) = split_grid_buffers(device, grid_len);

        let params = FastNufftParams3D {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            mx: mx as u32,
            my: my as u32,
            mz: mz as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            lx,
            ly,
            lz,
            beta,
            i0_beta,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        queue.write_buffer(&self.fast_3d_params_buffer, 0, bytemuck::bytes_of(&params));

        let spread_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type1 spread bg"),
            layout: &self.fast_3d_spread_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &value_buffer),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &self.fast_3d_params_buffer),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type1 encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type1 spread pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_spread_3d_pipeline);
            pass.set_bind_group(0, &spread_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(grid_len as u32), 1, 1);
        }

        let fft =
            GpuFft3d::new(Arc::clone(device), Arc::clone(queue), mx, my, mz).map_err(|_| {
                NufftWgpuError::InvalidPlan {
                    message: "oversampled 3D FFT plan is invalid for WGPU execution",
                }
            })?;
        fft.encode_forward_split(&mut encoder, &re_buffer, &im_buffer);

        let extract_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type1 extract bg"),
            layout: &self.fast_3d_extract_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &value_buffer),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &self.fast_3d_params_buffer),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type1 extract pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_extract_3d_pipeline);
            pass.set_bind_group(0, &extract_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(output_len as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer(device, queue, &output_buffer, output_len)
    }

    /// Execute fast gridded Type-2 3D NUFFT with GPU separable IFFT, deconvolution, and interpolation.
    ///
    /// Algorithm:
    /// 1. Load: one thread per oversampled grid cell, places deconvolved mode at grid position.
    /// 2. Inverse 3D FFT: uses `GpuFft3d` on the oversampled split grid (mx, my, mz).
    /// 3. Interpolate: one thread per non-uniform sample, reads IFFT'd grid with KB kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type2_3d(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        shape: (usize, usize, usize),
        oversampled: (usize, usize, usize),
        kernel_width: usize,
        lengths: (f32, f32, f32),
        beta: f32,
        i0_beta: f32,
        deconv_xyz: &[f32],
        modes: &[Complex32],
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let (nx, ny, nz) = shape;
        let (mx, my, mz) = oversampled;
        let (lx, ly, lz) = lengths;
        let grid_len = mx * my * mz;

        let coeff_data = complex_to_pods(modes);
        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|(x, y, z)| Position3Pod {
                x: *x,
                y: *y,
                z: *z,
                _pad: 0.0,
            })
            .collect();

        let coeff_buffer =
            storage_buffer(device, "apollo-nufft-wgpu fast3d type2 coeffs", &coeff_data);
        let position_buffer = storage_buffer(
            device,
            "apollo-nufft-wgpu fast3d type2 positions",
            &position_data,
        );
        let deconv_buffer =
            storage_buffer(device, "apollo-nufft-wgpu fast3d type2 deconv", deconv_xyz);
        let values_placeholder = placeholder_storage_buffer(
            device,
            "apollo-nufft-wgpu fast3d type2 values placeholder",
            positions.len(),
        );
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 output"),
            size: (positions.len() * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (re_buffer, im_buffer) = split_grid_buffers(device, grid_len);

        let params = FastNufftParams3D {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            mx: mx as u32,
            my: my as u32,
            mz: mz as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            lx,
            ly,
            lz,
            beta,
            i0_beta,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        queue.write_buffer(&self.fast_3d_params_buffer, 0, bytemuck::bytes_of(&params));

        let load_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 load bg"),
            layout: &self.fast_3d_extract_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &values_placeholder),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &self.fast_3d_params_buffer),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type2 load pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_load_3d_pipeline);
            pass.set_bind_group(0, &load_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(grid_len as u32), 1, 1);
        }

        let fft =
            GpuFft3d::new(Arc::clone(device), Arc::clone(queue), mx, my, mz).map_err(|_| {
                NufftWgpuError::InvalidPlan {
                    message: "oversampled 3D IFFT plan is invalid for WGPU execution",
                }
            })?;
        fft.encode_inverse_split(&mut encoder, &re_buffer, &im_buffer);

        let interp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 interpolate bg"),
            layout: &self.fast_3d_spread_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &values_placeholder),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &self.fast_3d_params_buffer),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type2 interpolate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_interpolate_3d_pipeline);
            pass.set_bind_group(0, &interp_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(positions.len() as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer(device, queue, &output_buffer, positions.len())
    }

    /// Execute fast gridded Type-1 1D NUFFT with pre-allocated buffers.
    ///
    /// Semantically identical to [`execute_fast_type1_1d`](Self::execute_fast_type1_1d)
    /// but reuses GPU buffers from `buffers` to eliminate per-dispatch allocation.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type1_1d_with_buffers(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        buffers: &NufftGpuBuffers1D,
        kernel_width: usize,
        length: f32,
        beta: f32,
        i0_beta: f32,
        deconv: &[f32],
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let n = buffers.n;
        let oversampled_len = buffers.m;

        let position_data = positions_to_complex_pods_1d(positions);
        let value_data = complex_to_pods(values);
        let deconv_data = real_to_complex_pods(deconv);

        queue.write_buffer(
            &buffers.position_buffer,
            0,
            bytemuck::cast_slice(&position_data),
        );
        queue.write_buffer(&buffers.value_buffer, 0, bytemuck::cast_slice(&value_data));
        queue.write_buffer(
            &buffers.deconv_buffer,
            0,
            bytemuck::cast_slice(&deconv_data),
        );

        let coeff_buffer =
            placeholder_storage_buffer(device, "apollo-nufft-wgpu fast empty coefficients", n);

        let params = FastNufftParams {
            n: n as u32,
            m: oversampled_len as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            length,
            beta,
            i0_beta,
            _pad: 0.0,
        };
        queue.write_buffer(&self.fast_params_buffer, 0, bytemuck::bytes_of(&params));

        let spread_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type1 spread bg"),
            layout: &self.fast_spread_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &buffers.value_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &buffers.output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast type1 spread encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 spread pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_spread_1d_pipeline);
            pass.set_bind_group(0, &spread_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(oversampled_len as u32), 1, 1);
        }

        let fft = GpuFft3d::new(Arc::clone(device), Arc::clone(queue), oversampled_len, 1, 1)
            .map_err(|_| NufftWgpuError::InvalidPlan {
                message: "oversampled FFT plan is invalid for WGPU execution",
            })?;
        fft.encode_forward_split(&mut encoder, &buffers.re_buffer, &buffers.im_buffer);

        let extract_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type1 extract bg"),
            layout: &self.fast_extract_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &buffers.value_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &buffers.output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 extract pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_extract_1d_pipeline);
            pass.set_bind_group(0, &extract_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(n as u32), 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer_with_staging(
            device,
            queue,
            &buffers.output_buffer,
            &buffers.staging_buffer,
            n,
        )
    }

    /// Execute fast gridded Type-2 1D NUFFT with pre-allocated buffers.
    ///
    /// Semantically identical to [`execute_fast_type2_1d`](Self::execute_fast_type2_1d)
    /// but reuses GPU buffers from `buffers` to eliminate per-dispatch allocation.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type2_1d_with_buffers(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        buffers: &NufftGpuBuffers1D,
        kernel_width: usize,
        length: f32,
        beta: f32,
        i0_beta: f32,
        deconv: &[f32],
        coefficients: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let oversampled_len = buffers.m;

        let coefficient_data = complex_to_pods(coefficients);
        let position_data = positions_to_complex_pods_1d(positions);
        let deconv_data = real_to_complex_pods_scaled(deconv, oversampled_len as f32);

        let coefficients_buffer = storage_buffer(
            device,
            "apollo-nufft-wgpu fast coefficients",
            &coefficient_data,
        );

        queue.write_buffer(
            &buffers.position_buffer,
            0,
            bytemuck::cast_slice(&position_data),
        );
        queue.write_buffer(
            &buffers.deconv_buffer,
            0,
            bytemuck::cast_slice(&deconv_data),
        );

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 output"),
            size: (positions.len() * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = FastNufftParams {
            n: buffers.n as u32,
            m: oversampled_len as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            length,
            beta,
            i0_beta,
            _pad: 0.0,
        };
        queue.write_buffer(&self.fast_params_buffer, 0, bytemuck::bytes_of(&params));

        let load_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 load bg"),
            layout: &self.fast_extract_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &coefficients_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coefficients_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 load encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 load pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_load_1d_pipeline);
            pass.set_bind_group(0, &load_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(oversampled_len as u32), 1, 1);
        }

        let fft = GpuFft3d::new(Arc::clone(device), Arc::clone(queue), oversampled_len, 1, 1)
            .map_err(|_| NufftWgpuError::InvalidPlan {
                message: "oversampled FFT plan is invalid for WGPU execution",
            })?;
        fft.encode_inverse_split(&mut encoder, &buffers.re_buffer, &buffers.im_buffer);

        let interp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 interpolate bg"),
            layout: &self.fast_spread_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &coefficients_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coefficients_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 interpolate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_interpolate_1d_pipeline);
            pass.set_bind_group(0, &interp_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(positions.len() as u32), 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer(device, queue, &output_buffer, positions.len())
    }

    /// Execute fast gridded Type-2 1D NUFFT and return diagnostic grid snapshots.
    ///
    /// This test/feature-gated path records the split oversampled grid
    /// immediately after coefficient load/deconvolution and immediately after
    /// the inverse FFT. Production execution remains on
    /// [`execute_fast_type2_1d_with_buffers`](Self::execute_fast_type2_1d_with_buffers).
    #[cfg(any(test, feature = "diagnostics"))]
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type2_1d_with_diagnostics(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        buffers: &NufftGpuBuffers1D,
        kernel_width: usize,
        length: f32,
        beta: f32,
        i0_beta: f32,
        deconv: &[f32],
        coefficients: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<(Vec<Complex32>, NufftType2GridDiagnostics)> {
        let oversampled_len = buffers.m;
        let coefficient_data = complex_to_pods(coefficients);
        let position_data = positions_to_complex_pods_1d(positions);
        let deconv_data = real_to_complex_pods_scaled(deconv, oversampled_len as f32);
        let coefficients_buffer = storage_buffer(
            device,
            "apollo-nufft-wgpu diagnostic fast coefficients",
            &coefficient_data,
        );
        queue.write_buffer(
            &buffers.position_buffer,
            0,
            bytemuck::cast_slice(&position_data),
        );
        queue.write_buffer(
            &buffers.deconv_buffer,
            0,
            bytemuck::cast_slice(&deconv_data),
        );

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast type2 output"),
            size: (positions.len() * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = FastNufftParams {
            n: buffers.n as u32,
            m: oversampled_len as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            length,
            beta,
            i0_beta,
            _pad: 0.0,
        };
        queue.write_buffer(&self.fast_params_buffer, 0, bytemuck::bytes_of(&params));

        let load_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast type2 load bg"),
            layout: &self.fast_extract_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &coefficients_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coefficients_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast type2 load encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu diagnostic fast type2 load pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_load_1d_pipeline);
            pass.set_bind_group(0, &load_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(oversampled_len as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        let after_load = read_split_grid_snapshot(
            device,
            queue,
            &buffers.re_buffer,
            &buffers.im_buffer,
            oversampled_len,
        )?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast type2 ifft/interp encoder"),
        });
        let fft = GpuFft3d::new(Arc::clone(device), Arc::clone(queue), oversampled_len, 1, 1)
            .map_err(|_| NufftWgpuError::InvalidPlan {
                message: "oversampled FFT plan is invalid for WGPU execution",
            })?;
        fft.encode_inverse_split(&mut encoder, &buffers.re_buffer, &buffers.im_buffer);
        queue.submit(std::iter::once(encoder.finish()));
        let after_ifft = read_split_grid_snapshot(
            device,
            queue,
            &buffers.re_buffer,
            &buffers.im_buffer,
            oversampled_len,
        )?;

        let interp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast type2 interpolate bg"),
            layout: &self.fast_spread_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &coefficients_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coefficients_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast type2 interpolate encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu diagnostic fast type2 interpolate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_interpolate_1d_pipeline);
            pass.set_bind_group(0, &interp_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(positions.len() as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        let output = read_complex_buffer(device, queue, &output_buffer, positions.len())?;
        Ok((
            output,
            NufftType2GridDiagnostics {
                after_load,
                after_ifft,
            },
        ))
    }

    /// Execute fast gridded Type-1 3D NUFFT with pre-allocated buffers.
    ///
    /// Semantically identical to [`execute_fast_type1_3d`](Self::execute_fast_type1_3d)
    /// but reuses GPU buffers from `buffers` to eliminate per-dispatch allocation.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type1_3d_with_buffers(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        buffers: &NufftGpuBuffers3D,
        kernel_width: usize,
        lengths: (f32, f32, f32),
        beta: f32,
        i0_beta: f32,
        deconv_xyz: &[f32],
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let (nx, ny, nz) = buffers.shape;
        let (mx, my, mz) = buffers.oversampled;
        let (lx, ly, lz) = lengths;
        let grid_len = mx * my * mz;
        let output_len = nx * ny * nz;

        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|(x, y, z)| Position3Pod {
                x: *x,
                y: *y,
                z: *z,
                _pad: 0.0,
            })
            .collect();
        let value_data = complex_to_pods(values);

        queue.write_buffer(
            &buffers.position_buffer,
            0,
            bytemuck::cast_slice(&position_data),
        );
        queue.write_buffer(&buffers.value_buffer, 0, bytemuck::cast_slice(&value_data));
        queue.write_buffer(&buffers.deconv_buffer, 0, bytemuck::cast_slice(deconv_xyz));

        let coeff_buffer = placeholder_storage_buffer(
            device,
            "apollo-nufft-wgpu fast3d coeffs placeholder",
            output_len,
        );

        let params = FastNufftParams3D {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            mx: mx as u32,
            my: my as u32,
            mz: mz as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            lx,
            ly,
            lz,
            beta,
            i0_beta,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        queue.write_buffer(&self.fast_3d_params_buffer, 0, bytemuck::bytes_of(&params));

        let spread_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type1 spread bg"),
            layout: &self.fast_3d_spread_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &buffers.value_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &buffers.output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &self.fast_3d_params_buffer),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type1 encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type1 spread pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_spread_3d_pipeline);
            pass.set_bind_group(0, &spread_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(grid_len as u32), 1, 1);
        }

        let fft =
            GpuFft3d::new(Arc::clone(device), Arc::clone(queue), mx, my, mz).map_err(|_| {
                NufftWgpuError::InvalidPlan {
                    message: "oversampled 3D FFT plan is invalid for WGPU execution",
                }
            })?;
        fft.encode_forward_split(&mut encoder, &buffers.re_buffer, &buffers.im_buffer);

        let extract_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type1 extract bg"),
            layout: &self.fast_3d_extract_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &buffers.value_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &buffers.output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &self.fast_3d_params_buffer),
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type1 extract pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_extract_3d_pipeline);
            pass.set_bind_group(0, &extract_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(output_len as u32), 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer_with_staging(
            device,
            queue,
            &buffers.output_buffer,
            &buffers.staging_buffer,
            output_len,
        )
    }

    /// Execute fast gridded Type-2 3D NUFFT with pre-allocated buffers.
    ///
    /// Semantically identical to [`execute_fast_type2_3d`](Self::execute_fast_type2_3d)
    /// but reuses GPU buffers from `buffers` to eliminate per-dispatch allocation.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type2_3d_with_buffers(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        buffers: &NufftGpuBuffers3D,
        kernel_width: usize,
        lengths: (f32, f32, f32),
        beta: f32,
        i0_beta: f32,
        deconv_xyz: &[f32],
        modes: &[Complex32],
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let (nx, ny, nz) = buffers.shape;
        let (mx, my, mz) = buffers.oversampled;
        let (lx, ly, lz) = lengths;
        let grid_len = mx * my * mz;

        let coeff_data = complex_to_pods(modes);
        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|(x, y, z)| Position3Pod {
                x: *x,
                y: *y,
                z: *z,
                _pad: 0.0,
            })
            .collect();

        let coeff_buffer =
            storage_buffer(device, "apollo-nufft-wgpu fast3d type2 coeffs", &coeff_data);

        queue.write_buffer(
            &buffers.position_buffer,
            0,
            bytemuck::cast_slice(&position_data),
        );
        queue.write_buffer(&buffers.deconv_buffer, 0, bytemuck::cast_slice(deconv_xyz));

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 output"),
            size: (positions.len() * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let values_placeholder = placeholder_storage_buffer(
            device,
            "apollo-nufft-wgpu fast3d type2 values placeholder",
            positions.len(),
        );

        let params = FastNufftParams3D {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            mx: mx as u32,
            my: my as u32,
            mz: mz as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            lx,
            ly,
            lz,
            beta,
            i0_beta,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        queue.write_buffer(&self.fast_3d_params_buffer, 0, bytemuck::bytes_of(&params));

        let load_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 load bg"),
            layout: &self.fast_3d_extract_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &values_placeholder),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &self.fast_3d_params_buffer),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type2 load pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_load_3d_pipeline);
            pass.set_bind_group(0, &load_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(grid_len as u32), 1, 1);
        }

        let fft =
            GpuFft3d::new(Arc::clone(device), Arc::clone(queue), mx, my, mz).map_err(|_| {
                NufftWgpuError::InvalidPlan {
                    message: "oversampled 3D IFFT plan is invalid for WGPU execution",
                }
            })?;
        fft.encode_inverse_split(&mut encoder, &buffers.re_buffer, &buffers.im_buffer);

        let interp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 interpolate bg"),
            layout: &self.fast_3d_spread_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &values_placeholder),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &self.fast_3d_params_buffer),
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type2 interpolate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_interpolate_3d_pipeline);
            pass.set_bind_group(0, &interp_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(positions.len() as u32), 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer(device, queue, &output_buffer, positions.len())
    }

    /// Execute fast gridded Type-2 3D NUFFT and return diagnostic grid snapshots.
    ///
    /// This test/feature-gated path records the split oversampled grid
    /// immediately after coefficient load/deconvolution and immediately after
    /// the inverse 3D FFT. Production execution remains on
    /// [`execute_fast_type2_3d_with_buffers`](Self::execute_fast_type2_3d_with_buffers).
    #[cfg(any(test, feature = "diagnostics"))]
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type2_3d_with_diagnostics(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        buffers: &NufftGpuBuffers3D,
        kernel_width: usize,
        lengths: (f32, f32, f32),
        beta: f32,
        i0_beta: f32,
        deconv_xyz: &[f32],
        modes: &[Complex32],
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<(Vec<Complex32>, NufftType2GridDiagnostics)> {
        let (nx, ny, nz) = buffers.shape;
        let (mx, my, mz) = buffers.oversampled;
        let (lx, ly, lz) = lengths;
        let grid_len = mx * my * mz;

        let coeff_data = complex_to_pods(modes);
        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|(x, y, z)| Position3Pod {
                x: *x,
                y: *y,
                z: *z,
                _pad: 0.0,
            })
            .collect();
        let coeff_buffer = storage_buffer(
            device,
            "apollo-nufft-wgpu diagnostic fast3d type2 coeffs",
            &coeff_data,
        );
        queue.write_buffer(
            &buffers.position_buffer,
            0,
            bytemuck::cast_slice(&position_data),
        );
        queue.write_buffer(&buffers.deconv_buffer, 0, bytemuck::cast_slice(deconv_xyz));

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast3d type2 output"),
            size: (positions.len() * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let values_placeholder = placeholder_storage_buffer(
            device,
            "apollo-nufft-wgpu diagnostic fast3d values placeholder",
            positions.len(),
        );
        let params = FastNufftParams3D {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            mx: mx as u32,
            my: my as u32,
            mz: mz as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            lx,
            ly,
            lz,
            beta,
            i0_beta,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        queue.write_buffer(&self.fast_3d_params_buffer, 0, bytemuck::bytes_of(&params));

        let load_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast3d type2 load bg"),
            layout: &self.fast_3d_extract_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &values_placeholder),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &self.fast_3d_params_buffer),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast3d type2 load encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu diagnostic fast3d type2 load pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_load_3d_pipeline);
            pass.set_bind_group(0, &load_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(grid_len as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        let after_load = read_split_grid_snapshot(
            device,
            queue,
            &buffers.re_buffer,
            &buffers.im_buffer,
            grid_len,
        )?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast3d type2 ifft encoder"),
        });
        let fft =
            GpuFft3d::new(Arc::clone(device), Arc::clone(queue), mx, my, mz).map_err(|_| {
                NufftWgpuError::InvalidPlan {
                    message: "oversampled 3D IFFT plan is invalid for WGPU execution",
                }
            })?;
        fft.encode_inverse_split(&mut encoder, &buffers.re_buffer, &buffers.im_buffer);
        queue.submit(std::iter::once(encoder.finish()));
        let after_ifft = read_split_grid_snapshot(
            device,
            queue,
            &buffers.re_buffer,
            &buffers.im_buffer,
            grid_len,
        )?;

        let interp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast3d type2 interpolate bg"),
            layout: &self.fast_3d_spread_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &values_placeholder),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &self.fast_3d_params_buffer),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast3d type2 interpolate encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu diagnostic fast3d type2 interpolate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_interpolate_3d_pipeline);
            pass.set_bind_group(0, &interp_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(positions.len() as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        let output = read_complex_buffer(device, queue, &output_buffer, positions.len())?;
        Ok((
            output,
            NufftType2GridDiagnostics {
                after_load,
                after_ifft,
            },
        ))
    }

    /// Read back the split real/imaginary grid buffers as `Vec<Complex32>`.
    ///
    /// This method is only available with the `debug-readbacks` feature enabled.
    /// It creates a staging buffer, copies the grid data, maps it, and reads it back.
    /// This is slow (GPU→CPU synchronization) and should only be used for triage.
    #[cfg(feature = "debug-readbacks")]
    pub fn read_grid_1d(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        re_buffer: &wgpu::Buffer,
        im_buffer: &wgpu::Buffer,
        len: usize,
    ) -> NufftWgpuResult<Vec<Complex32>> {
        read_split_grid(device, queue, re_buffer, im_buffer, len)
    }

    /// Read back the 3D split grid buffers as `Vec<Complex32>`.
    #[cfg(feature = "debug-readbacks")]
    pub fn read_grid_3d(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        re_buffer: &wgpu::Buffer,
        im_buffer: &wgpu::Buffer,
        len: usize,
    ) -> NufftWgpuResult<Vec<Complex32>> {
        read_split_grid(device, queue, re_buffer, im_buffer, len)
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

fn uniform_layout_entry_3d(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(
                std::num::NonZeroU64::new(std::mem::size_of::<FastNufftParams3D>() as u64)
                    .expect("nonzero 3d uniform size"),
            ),
        },
        count: None,
    }
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn complex_to_pods(values: &[Complex32]) -> Vec<ComplexPod> {
    values
        .iter()
        .map(|value| ComplexPod {
            re: value.re,
            im: value.im,
        })
        .collect()
}

fn positions_to_complex_pods_1d(positions: &[f32]) -> Vec<ComplexPod> {
    positions
        .iter()
        .map(|x| ComplexPod { re: *x, im: 0.0 })
        .collect()
}

fn real_to_complex_pods(values: &[f32]) -> Vec<ComplexPod> {
    values
        .iter()
        .map(|value| ComplexPod {
            re: *value,
            im: 0.0,
        })
        .collect()
}

fn real_to_complex_pods_scaled(values: &[f32], scale: f32) -> Vec<ComplexPod> {
    values
        .iter()
        .map(|value| ComplexPod {
            re: *value * scale,
            im: 0.0,
        })
        .collect()
}

fn storage_buffer<T: Pod>(device: &wgpu::Device, label: &'static str, data: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

fn placeholder_storage_buffer(
    device: &wgpu::Device,
    label: &'static str,
    len: usize,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len.max(1) * std::mem::size_of::<ComplexPod>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn split_grid_buffers(device: &wgpu::Device, len: usize) -> (wgpu::Buffer, wgpu::Buffer) {
    let size = (len * std::mem::size_of::<f32>()) as u64;
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let re = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("apollo-nufft-wgpu fast grid re"),
        size,
        usage,
        mapped_at_creation: false,
    });
    let im = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("apollo-nufft-wgpu fast grid im"),
        size,
        usage,
        mapped_at_creation: false,
    });
    (re, im)
}

fn read_complex_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
) -> NufftWgpuResult<Vec<Complex32>> {
    let size = (len * std::mem::size_of::<ComplexPod>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("apollo-nufft-wgpu fast output staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("apollo-nufft-wgpu fast readback encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
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

/// Read back `len` complex values from `source` into a pre-allocated `staging` buffer.
///
/// The staging buffer must have `MAP_READ | COPY_DST` usage and size >= `len * size_of::<ComplexPod>()`.
/// This variant avoids allocating a new staging buffer on each call.
fn read_complex_buffer_with_staging(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    len: usize,
) -> NufftWgpuResult<Vec<Complex32>> {
    let size = (len * std::mem::size_of::<ComplexPod>()) as u64;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("apollo-nufft-wgpu fast readback encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..size);
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

#[cfg(any(test, feature = "diagnostics"))]
fn read_real_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
) -> NufftWgpuResult<Vec<f32>> {
    let size = (len * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("apollo-nufft-wgpu diagnostic grid staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("apollo-nufft-wgpu diagnostic grid readback encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..size);
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
        bytemuck::cast_slice(&mapped).to_vec()
    };
    staging.unmap();
    Ok(output)
}

#[cfg(any(test, feature = "diagnostics"))]
fn read_split_grid_snapshot(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    re_buffer: &wgpu::Buffer,
    im_buffer: &wgpu::Buffer,
    len: usize,
) -> NufftWgpuResult<NufftGridSnapshot> {
    Ok(NufftGridSnapshot {
        re: read_real_buffer(device, queue, re_buffer, len)?,
        im: read_real_buffer(device, queue, im_buffer, len)?,
    })
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}

/// Read back split real/imaginary GPU buffers into interleaved `Complex32` values.
///
/// Creates staging buffers, copies both components, maps them, and interleaves
/// the real and imaginary parts into a single `Vec<Complex32>`.
#[cfg(feature = "debug-readbacks")]
fn read_split_grid(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    re_buffer: &wgpu::Buffer,
    im_buffer: &wgpu::Buffer,
    len: usize,
) -> NufftWgpuResult<Vec<Complex32>> {
    let size = (len * std::mem::size_of::<f32>()) as u64;
    let staging_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
    let re_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("apollo-nufft-wgpu debug re staging"),
        size,
        usage: staging_usage,
        mapped_at_creation: false,
    });
    let im_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("apollo-nufft-wgpu debug im staging"),
        size,
        usage: staging_usage,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("apollo-nufft-wgpu debug readback encoder"),
    });
    encoder.copy_buffer_to_buffer(re_buffer, 0, &re_staging, 0, size);
    encoder.copy_buffer_to_buffer(im_buffer, 0, &im_staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    let re_slice = re_staging.slice(..);
    let im_slice = im_staging.slice(..);
    let (re_tx, re_rx) = mpsc::channel();
    let (im_tx, im_rx) = mpsc::channel();
    re_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = re_tx.send(result);
    });
    im_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = im_tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    match (re_rx.recv(), im_rx.recv()) {
        (Ok(Ok(())), Ok(Ok(()))) => {}
        (Ok(Err(e)), _) | (_, Ok(Err(e))) => {
            return Err(NufftWgpuError::BufferMapFailed {
                message: e.to_string(),
            });
        }
        (Err(e), _) | (_, Err(e)) => {
            return Err(NufftWgpuError::BufferMapFailed {
                message: e.to_string(),
            });
        }
    }
    let output = {
        let re_mapped = re_slice.get_mapped_range();
        let im_mapped = im_slice.get_mapped_range();
        let re_data: &[f32] = bytemuck::cast_slice(&re_mapped);
        let im_data: &[f32] = bytemuck::cast_slice(&im_mapped);
        re_data
            .iter()
            .zip(im_data.iter())
            .map(|(&re, &im)| Complex32::new(re, im))
            .collect()
    };
    re_staging.unmap();
    im_staging.unmap();
    Ok(output)
}
