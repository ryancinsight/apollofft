//! Shader-backed 3D FFT implementation for the Apollo WGPU backend.

use ndarray::Array3;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AxisStrategy {
    Radix2,
    ChirpZ { n: usize, m: usize },
}

#[derive(Debug, Clone, Copy)]
enum Axis {
    X,
    Y,
    Z,
}

struct RadixStages {
    _param_bufs: Vec<wgpu::Buffer>,
    bgs: Vec<wgpu::BindGroup>,
    fft_m: u32,
}

impl RadixStages {
    fn empty() -> Self {
        Self {
            _param_bufs: Vec::new(),
            bgs: Vec::new(),
            fft_m: 0,
        }
    }

    fn precompute(
        device: &wgpu::Device,
        params_layout: &wgpu::BindGroupLayout,
        fft_m: u32,
        inverse: bool,
    ) -> Self {
        let inv_flag = u32::from(inverse);
        let log2_m = fft_m.trailing_zeros() as usize;
        let stage_count = 1 + log2_m + usize::from(inverse);
        let mut param_bufs = Vec::with_capacity(stage_count);
        let mut bgs = Vec::with_capacity(stage_count);

        let mut push_stage = |data: [u32; 4]| {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollofft-wgpu radix2 params"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollofft-wgpu radix2 params bg"),
                layout: params_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            param_bufs.push(buf);
            bg
        };

        bgs.push(push_stage([fft_m, 0, inv_flag, 0]));
        for stage in 0..log2_m as u32 {
            bgs.push(push_stage([fft_m, stage, inv_flag, 0]));
        }
        if inverse {
            bgs.push(push_stage([fft_m, 0, 1, 0]));
        }

        Self {
            _param_bufs: param_bufs,
            bgs,
            fft_m,
        }
    }
}

#[allow(dead_code)]
struct ChirpData {
    h_fft_re: wgpu::Buffer,
    h_fft_im: wgpu::Buffer,
    data_chirp_layout: wgpu::BindGroupLayout,
    params_layout: wgpu::BindGroupLayout,
    premul_pipeline: wgpu::ComputePipeline,
    pointmul_pipeline: wgpu::ComputePipeline,
    scale_pipeline: wgpu::ComputePipeline,
    postmul_pipeline: wgpu::ComputePipeline,
    negate_im_pipeline: wgpu::ComputePipeline,
    n: u32,
    m: u32,
    data_chirp_bg: wgpu::BindGroup,
    _params_buf: wgpu::Buffer,
    params_bg: wgpu::BindGroup,
    radix2_fwd: RadixStages,
    radix2_inv: RadixStages,
}

/// Returns true when this crate is linked and the WGPU backend is available.
#[must_use]
pub fn gpu_fft_available() -> bool {
    true
}

/// GPU-backed 3D FFT plan.
pub struct GpuFft3d {
    nx: usize,
    ny: usize,
    nz: usize,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    strategy_x: AxisStrategy,
    strategy_y: AxisStrategy,
    strategy_z: AxisStrategy,
    bitrev_pipeline: wgpu::ComputePipeline,
    forward_pipeline: wgpu::ComputePipeline,
    scale_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    params_layout: wgpu::BindGroupLayout,
    re_buf: wgpu::Buffer,
    im_buf: wgpu::Buffer,
    re_staging: wgpu::Buffer,
    im_staging: wgpu::Buffer,
    chirp_x: Option<ChirpData>,
    chirp_y: Option<ChirpData>,
    chirp_z: Option<ChirpData>,
    data_bg: wgpu::BindGroup,
    axis_fwd_x: RadixStages,
    axis_inv_x: RadixStages,
    axis_fwd_y: RadixStages,
    axis_inv_y: RadixStages,
    axis_fwd_z: RadixStages,
    axis_inv_z: RadixStages,
}

fn next_pow2(n: usize) -> usize {
    let mut m = 1usize;
    while m < n {
        m <<= 1;
    }
    m
}

#[inline]
fn is_pow2(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

fn axis_strategy_for(n: usize) -> AxisStrategy {
    if is_pow2(n) {
        AxisStrategy::Radix2
    } else {
        AxisStrategy::ChirpZ {
            n,
            m: next_pow2(2 * n - 1),
        }
    }
}

fn validate_dimensions(
    max_buffer_size: u64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<(), String> {
    for (name, n) in [("nx", nx), ("ny", ny), ("nz", nz)] {
        if n == 0 {
            return Err(format!("GpuFft3d: {name} must be > 0"));
        }
        let m = match axis_strategy_for(n) {
            AxisStrategy::Radix2 => n,
            AxisStrategy::ChirpZ { m, .. } => m,
        };
        let required = m as u64 * std::mem::size_of::<f32>() as u64;
        if required > max_buffer_size {
            return Err(format!(
                "GpuFft3d: {name}={n} requires M={m} × 4 bytes = {required} bytes, exceeds device max_buffer_size={max_buffer_size}"
            ));
        }
    }
    Ok(())
}

impl GpuFft3d {
    /// Create a new WGPU-backed 3D FFT plan.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<Self, String> {
        validate_dimensions(device.limits().max_buffer_size, nx, ny, nz)?;

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollofft-wgpu fft shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fft.wgsl").into()),
        });

        let data_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollofft-wgpu fft data layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
            ],
        });

        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollofft-wgpu fft params layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollofft-wgpu fft pipeline layout"),
            bind_group_layouts: &[&data_layout, &params_layout],
            push_constant_ranges: &[],
        });

        let build_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let bitrev_pipeline = build_pipeline("fft_bitrev");
        let forward_pipeline = build_pipeline("fft_forward");
        let scale_pipeline = build_pipeline("fft_scale");

        let strategy_x = axis_strategy_for(nx);
        let strategy_y = axis_strategy_for(ny);
        let strategy_z = axis_strategy_for(nz);

        let n_total = nx * ny * nz;
        let buf_size = (n_total * std::mem::size_of::<f32>()) as u64;
        let working_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let staging_usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;

        let re_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollofft-wgpu re"),
            size: buf_size,
            usage: working_usage,
            mapped_at_creation: false,
        });
        let im_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollofft-wgpu im"),
            size: buf_size,
            usage: working_usage,
            mapped_at_creation: false,
        });
        let re_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollofft-wgpu re staging"),
            size: buf_size,
            usage: staging_usage,
            mapped_at_creation: false,
        });
        let im_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollofft-wgpu im staging"),
            size: buf_size,
            usage: staging_usage,
            mapped_at_creation: false,
        });

        let data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollofft-wgpu data bg"),
            layout: &data_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: re_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: im_buf.as_entire_binding(),
                },
            ],
        });

        let chirp_x = match strategy_x {
            AxisStrategy::Radix2 => None,
            AxisStrategy::ChirpZ { n, m } => Some(Self::build_chirp_data(
                &device,
                &params_layout,
                &re_buf,
                &im_buf,
                n,
                m,
            )),
        };
        let chirp_y = match strategy_y {
            AxisStrategy::Radix2 => None,
            AxisStrategy::ChirpZ { n, m } => Some(Self::build_chirp_data(
                &device,
                &params_layout,
                &re_buf,
                &im_buf,
                n,
                m,
            )),
        };
        let chirp_z = match strategy_z {
            AxisStrategy::Radix2 => None,
            AxisStrategy::ChirpZ { n, m } => Some(Self::build_chirp_data(
                &device,
                &params_layout,
                &re_buf,
                &im_buf,
                n,
                m,
            )),
        };

        let axis_fwd_x = match strategy_x {
            AxisStrategy::Radix2 => {
                RadixStages::precompute(&device, &params_layout, nx as u32, false)
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };
        let axis_inv_x = match strategy_x {
            AxisStrategy::Radix2 => {
                RadixStages::precompute(&device, &params_layout, nx as u32, true)
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };
        let axis_fwd_y = match strategy_y {
            AxisStrategy::Radix2 => {
                RadixStages::precompute(&device, &params_layout, ny as u32, false)
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };
        let axis_inv_y = match strategy_y {
            AxisStrategy::Radix2 => {
                RadixStages::precompute(&device, &params_layout, ny as u32, true)
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };
        let axis_fwd_z = match strategy_z {
            AxisStrategy::Radix2 => {
                RadixStages::precompute(&device, &params_layout, nz as u32, false)
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };
        let axis_inv_z = match strategy_z {
            AxisStrategy::Radix2 => {
                RadixStages::precompute(&device, &params_layout, nz as u32, true)
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };

        Ok(Self {
            nx,
            ny,
            nz,
            device,
            queue,
            strategy_x,
            strategy_y,
            strategy_z,
            bitrev_pipeline,
            forward_pipeline,
            scale_pipeline,
            bind_group_layout: data_layout,
            params_layout,
            re_buf,
            im_buf,
            re_staging,
            im_staging,
            chirp_x,
            chirp_y,
            chirp_z,
            data_bg,
            axis_fwd_x,
            axis_inv_x,
            axis_fwd_y,
            axis_inv_y,
            axis_fwd_z,
            axis_inv_z,
        })
    }

    fn build_chirp_data(
        device: &wgpu::Device,
        params_layout: &wgpu::BindGroupLayout,
        re_buf: &wgpu::Buffer,
        im_buf: &wgpu::Buffer,
        n: usize,
        m: usize,
    ) -> ChirpData {
        let mut h_re = vec![0.0_f32; m];
        let mut h_im = vec![0.0_f32; m];
        for idx in 0..n {
            let arg = -std::f32::consts::PI * (idx * idx) as f32 / n as f32;
            let re = arg.cos();
            let im = arg.sin();
            h_re[idx] = re;
            h_im[idx] = im;
            if idx > 0 {
                h_re[m - idx] = re;
                h_im[m - idx] = im;
            }
        }

        let h_fft_re = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollofft-wgpu chirp re"),
            contents: bytemuck::cast_slice(&h_re),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let h_fft_im = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollofft-wgpu chirp im"),
            contents: bytemuck::cast_slice(&h_im),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let data_chirp_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollofft-wgpu chirp data layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let chirp_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollofft-wgpu chirp shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/chirp.wgsl").into()),
        });
        let chirp_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollofft-wgpu chirp pipeline layout"),
            bind_group_layouts: &[&data_chirp_layout, params_layout],
            push_constant_ranges: &[],
        });
        let build_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&chirp_layout),
                module: &chirp_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let data_chirp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollofft-wgpu chirp data bg"),
            layout: &data_chirp_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: re_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: im_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: h_fft_re.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: h_fft_im.as_entire_binding(),
                },
            ],
        });

        let params_data = [n as u32, m as u32, 0, 0];
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollofft-wgpu chirp params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollofft-wgpu chirp params bg"),
            layout: params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }],
        });

        ChirpData {
            h_fft_re,
            h_fft_im,
            data_chirp_layout,
            params_layout: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("apollofft-wgpu chirp params layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            }),
            premul_pipeline: build_pipeline("chirp_premul"),
            pointmul_pipeline: build_pipeline("chirp_pointmul"),
            scale_pipeline: build_pipeline("chirp_scale"),
            postmul_pipeline: build_pipeline("chirp_postmul"),
            negate_im_pipeline: build_pipeline("chirp_negate_im"),
            n: n as u32,
            m: m as u32,
            data_chirp_bg,
            _params_buf: params_buf,
            params_bg,
            radix2_fwd: RadixStages::precompute(device, params_layout, m as u32, false),
            radix2_inv: RadixStages::precompute(device, params_layout, m as u32, true),
        }
    }

    /// Forward transform of a real field into an interleaved complex buffer.
    #[must_use]
    pub fn forward(&self, field: &Array3<f64>) -> Vec<f32> {
        assert_eq!(field.dim(), (self.nx, self.ny, self.nz));
        let n = self.nx * self.ny * self.nz;
        let mut re_data: Vec<f32> = field.iter().map(|&value| value as f32).collect();
        let mut im_data = vec![0.0_f32; n];

        self.run_batched_axis_fft(&mut re_data, &mut im_data, Axis::Z, false);
        self.run_batched_axis_fft(&mut re_data, &mut im_data, Axis::Y, false);
        self.run_batched_axis_fft(&mut re_data, &mut im_data, Axis::X, false);

        let mut out = Vec::with_capacity(2 * n);
        for (re, im) in re_data.iter().zip(im_data.iter()) {
            out.push(*re);
            out.push(*im);
        }
        out
    }

    /// Inverse transform of an interleaved complex buffer into a real field.
    pub fn inverse(&self, field_hat: &[f32], out: &mut Array3<f64>) {
        assert_eq!(field_hat.len(), 2 * self.nx * self.ny * self.nz);
        assert_eq!(out.dim(), (self.nx, self.ny, self.nz));

        let n = self.nx * self.ny * self.nz;
        let mut re_data: Vec<f32> = (0..n).map(|idx| field_hat[2 * idx]).collect();
        let mut im_data: Vec<f32> = (0..n).map(|idx| field_hat[2 * idx + 1]).collect();

        self.run_batched_axis_fft(&mut re_data, &mut im_data, Axis::X, true);
        self.run_batched_axis_fft(&mut re_data, &mut im_data, Axis::Y, true);
        self.run_batched_axis_fft(&mut re_data, &mut im_data, Axis::Z, true);

        for (dst, &value) in out.iter_mut().zip(re_data.iter()) {
            *dst = value as f64;
        }
    }

    fn run_batched_axis_fft(&self, re: &mut [f32], im: &mut [f32], axis: Axis, inverse: bool) {
        match axis {
            Axis::Z => {
                for ix in 0..self.nx {
                    for iy in 0..self.ny {
                        let base = ((ix * self.ny) + iy) * self.nz;
                        let (row_re, row_im) = self.run_axis_fft(
                            axis,
                            &re[base..base + self.nz],
                            &im[base..base + self.nz],
                            inverse,
                        );
                        re[base..base + self.nz].copy_from_slice(&row_re);
                        im[base..base + self.nz].copy_from_slice(&row_im);
                    }
                }
            }
            Axis::Y => {
                for ix in 0..self.nx {
                    for iz in 0..self.nz {
                        let mut row_re = Vec::with_capacity(self.ny);
                        let mut row_im = Vec::with_capacity(self.ny);
                        for iy in 0..self.ny {
                            let idx = ((ix * self.ny) + iy) * self.nz + iz;
                            row_re.push(re[idx]);
                            row_im.push(im[idx]);
                        }
                        let (row_re, row_im) = self.run_axis_fft(axis, &row_re, &row_im, inverse);
                        for iy in 0..self.ny {
                            let idx = ((ix * self.ny) + iy) * self.nz + iz;
                            re[idx] = row_re[iy];
                            im[idx] = row_im[iy];
                        }
                    }
                }
            }
            Axis::X => {
                for iy in 0..self.ny {
                    for iz in 0..self.nz {
                        let mut row_re = Vec::with_capacity(self.nx);
                        let mut row_im = Vec::with_capacity(self.nx);
                        for ix in 0..self.nx {
                            let idx = ((ix * self.ny) + iy) * self.nz + iz;
                            row_re.push(re[idx]);
                            row_im.push(im[idx]);
                        }
                        let (row_re, row_im) = self.run_axis_fft(axis, &row_re, &row_im, inverse);
                        for ix in 0..self.nx {
                            let idx = ((ix * self.ny) + iy) * self.nz + iz;
                            re[idx] = row_re[ix];
                            im[idx] = row_im[ix];
                        }
                    }
                }
            }
        }
    }

    fn run_axis_fft(
        &self,
        axis: Axis,
        re_data: &[f32],
        im_data: &[f32],
        inverse: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let len = match axis {
            Axis::X => self.nx,
            Axis::Y => self.ny,
            Axis::Z => self.nz,
        };
        let buf_size = (len * std::mem::size_of::<f32>()) as u64;
        self.queue
            .write_buffer(&self.re_buf, 0, bytemuck::cast_slice(re_data));
        self.queue
            .write_buffer(&self.im_buf, 0, bytemuck::cast_slice(im_data));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollofft-wgpu fft encoder"),
            });

        match axis {
            Axis::Z => match self.strategy_z {
                AxisStrategy::Radix2 => {
                    let stages = if inverse {
                        &self.axis_inv_z
                    } else {
                        &self.axis_fwd_z
                    };
                    self.dispatch_radix2(&mut encoder, self.nz as u32, inverse, stages);
                }
                AxisStrategy::ChirpZ { .. } => self.dispatch_chirp(
                    &mut encoder,
                    self.chirp_z.as_ref().expect("chirp_z exists for ChirpZ"),
                    inverse,
                ),
            },
            Axis::Y => match self.strategy_y {
                AxisStrategy::Radix2 => {
                    let stages = if inverse {
                        &self.axis_inv_y
                    } else {
                        &self.axis_fwd_y
                    };
                    self.dispatch_radix2(&mut encoder, self.ny as u32, inverse, stages);
                }
                AxisStrategy::ChirpZ { .. } => self.dispatch_chirp(
                    &mut encoder,
                    self.chirp_y.as_ref().expect("chirp_y exists for ChirpZ"),
                    inverse,
                ),
            },
            Axis::X => match self.strategy_x {
                AxisStrategy::Radix2 => {
                    let stages = if inverse {
                        &self.axis_inv_x
                    } else {
                        &self.axis_fwd_x
                    };
                    self.dispatch_radix2(&mut encoder, self.nx as u32, inverse, stages);
                }
                AxisStrategy::ChirpZ { .. } => self.dispatch_chirp(
                    &mut encoder,
                    self.chirp_x.as_ref().expect("chirp_x exists for ChirpZ"),
                    inverse,
                ),
            },
        }

        encoder.copy_buffer_to_buffer(&self.re_buf, 0, &self.re_staging, 0, buf_size);
        encoder.copy_buffer_to_buffer(&self.im_buf, 0, &self.im_staging, 0, buf_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let re_slice = self.re_staging.slice(..buf_size);
        let im_slice = self.im_staging.slice(..buf_size);
        re_slice.map_async(wgpu::MapMode::Read, |_| {});
        im_slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait);

        let re_out = {
            let mapped = re_slice.get_mapped_range();
            bytemuck::cast_slice(&mapped).to_vec()
        };
        let im_out = {
            let mapped = im_slice.get_mapped_range();
            bytemuck::cast_slice(&mapped).to_vec()
        };

        self.re_staging.unmap();
        self.im_staging.unmap();

        (re_out, im_out)
    }

    fn dispatch_radix2(
        &self,
        enc: &mut wgpu::CommandEncoder,
        fft_m: u32,
        inverse: bool,
        stages: &RadixStages,
    ) {
        debug_assert_eq!(stages.fft_m, fft_m);
        let log2_m = fft_m.trailing_zeros() as usize;

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollofft-wgpu bitrev"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bitrev_pipeline);
            pass.set_bind_group(0, &self.data_bg, &[]);
            pass.set_bind_group(1, &stages.bgs[0], &[]);
            pass.dispatch_workgroups((fft_m / 2).div_ceil(256), 1, 1);
        }

        for stage in 0..log2_m {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollofft-wgpu butterfly"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_pipeline);
            pass.set_bind_group(0, &self.data_bg, &[]);
            pass.set_bind_group(1, &stages.bgs[1 + stage], &[]);
            pass.dispatch_workgroups((fft_m / 2).div_ceil(256), 1, 1);
        }

        if inverse {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollofft-wgpu scale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scale_pipeline);
            pass.set_bind_group(0, &self.data_bg, &[]);
            pass.set_bind_group(
                1,
                stages.bgs.last().expect("inverse stages include scale"),
                &[],
            );
            pass.dispatch_workgroups(fft_m.div_ceil(256), 1, 1);
        }
    }

    fn dispatch_chirp(&self, enc: &mut wgpu::CommandEncoder, chirp: &ChirpData, inverse: bool) {
        let dispatch_n = chirp.n.div_ceil(256);
        let dispatch_m = chirp.m.div_ceil(256);

        if inverse {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollofft-wgpu chirp negate pre"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.negate_im_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(dispatch_n, 1, 1);
        }

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollofft-wgpu chirp premul"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.premul_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(dispatch_m, 1, 1);
        }

        self.dispatch_radix2(enc, chirp.m, false, &chirp.radix2_fwd);

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollofft-wgpu chirp pointmul"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.pointmul_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(dispatch_m, 1, 1);
        }

        self.dispatch_radix2(enc, chirp.m, true, &chirp.radix2_inv);

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollofft-wgpu chirp postmul"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.postmul_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(dispatch_n, 1, 1);
        }

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollofft-wgpu chirp barrier"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.scale_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(dispatch_n, 1, 1);
        }

        if inverse {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollofft-wgpu chirp negate post"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.negate_im_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(dispatch_n, 1, 1);
        }
    }
}

impl std::fmt::Debug for GpuFft3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuFft3d")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WgpuBackend;
    use apollofft::{FftBackend, FftPlan3D, Shape3D};
    use ndarray::Array3;

    #[test]
    fn gpu_fft_available_flag_is_true() {
        assert!(gpu_fft_available());
    }

    #[test]
    fn rejects_zero_dimensions() {
        let err = validate_dimensions(1024, 0, 4, 4).expect_err("zero nx must be rejected");
        assert!(err.contains("nx must be > 0"));
    }

    #[test]
    fn rejects_dimensions_that_exceed_device_limit() {
        let err =
            validate_dimensions(16, 4, 5, 4).expect_err("chirp embedding should exceed limit");
        assert!(err.contains("ny=5"));
        assert!(err.contains("max_buffer_size=16"));
    }

    #[test]
    fn selects_axis_strategy_per_dimension() {
        assert_eq!(axis_strategy_for(8), AxisStrategy::Radix2);
        assert_eq!(axis_strategy_for(5), AxisStrategy::ChirpZ { n: 5, m: 16 });
        assert_eq!(axis_strategy_for(9), AxisStrategy::ChirpZ { n: 9, m: 32 });
    }

    #[test]
    fn gpu_roundtrip_tracks_cpu_reference_when_adapter_is_available() {
        let Ok(backend) = WgpuBackend::try_default() else {
            return;
        };
        let shape = Shape3D::new(8, 8, 8).expect("valid shape");
        let gpu_plan = backend.plan_3d(shape).expect("gpu plan");
        let cpu_plan = FftPlan3D::new(8, 8, 8);
        let field = Array3::from_shape_fn((8, 8, 8), |(i, j, k)| {
            (i as f64 * 0.3 + j as f64 * 0.5 + k as f64 * 0.7).sin()
        });

        let cpu_spectrum = cpu_plan.forward(&field);
        let gpu_spectrum = gpu_plan.forward(&field);
        let gpu_complex = Array3::from_shape_fn((8, 8, 8), |(i, j, k)| {
            let flat = ((i * 8 + j) * 8 + k) * 2;
            num_complex::Complex64::new(gpu_spectrum[flat] as f64, gpu_spectrum[flat + 1] as f64)
        });
        let forward_max_abs_error = cpu_spectrum
            .iter()
            .zip(gpu_complex.iter())
            .map(|(lhs, rhs)| (lhs - rhs).norm())
            .fold(0.0_f64, f64::max);
        assert!(forward_max_abs_error < 2e-5);

        let cpu_inverse = cpu_plan.inverse(&cpu_spectrum);
        let mut gpu_inverse = Array3::<f64>::zeros((8, 8, 8));
        gpu_plan.inverse(&gpu_spectrum, &mut gpu_inverse);
        let inverse_max_abs_error = cpu_inverse
            .iter()
            .zip(gpu_inverse.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);
        assert!(inverse_max_abs_error < 1e-5);
    }
}
