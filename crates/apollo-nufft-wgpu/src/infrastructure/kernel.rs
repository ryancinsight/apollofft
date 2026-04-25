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

/// Cached WGPU state for direct and fast-gridded NUFFT dispatches.
#[derive(Debug)]
pub struct NufftGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    fast_spread_layout: wgpu::BindGroupLayout,
    fast_extract_layout: wgpu::BindGroupLayout,
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
        let empty_coefficients = vec![ComplexPod::zeroed(); n.max(1)];
        let position_buffer =
            storage_buffer(device, "apollo-nufft-wgpu fast positions", &position_data);
        let value_buffer = storage_buffer(device, "apollo-nufft-wgpu fast values", &value_data);
        let deconv_buffer = storage_buffer(device, "apollo-nufft-wgpu fast deconv", &deconv_data);
        let coefficient_buffer = storage_buffer(
            device,
            "apollo-nufft-wgpu fast empty coefficients",
            &empty_coefficients,
        );
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
        let deconv_data = real_to_complex_pods(deconv);
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

fn storage_buffer<T: Pod>(device: &wgpu::Device, label: &'static str, data: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
