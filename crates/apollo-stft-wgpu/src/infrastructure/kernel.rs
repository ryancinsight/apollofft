//! GPU compute kernels for the forward and inverse Short-Time Fourier Transform.
//!
//! Forward kernel: each invocation computes one complex output element X[frame_m, bin_k]
//! via a direct DFT sum over the Hann-windowed frame.
//!
//! Inverse kernel: FFT-accelerated two-stage WOLA reconstruction.
//!
//! ## Inverse dispatch sequence
//! The following passes are all encoded in one `CommandEncoder`, exploiting
//! the implicit per-pass memory barrier in WebGPU (§3.4 sequential pass ordering):
//!
//! 1. `stft_deinterleave`: interleaved complex spectrum f32 → split re/im scratch buffers.
//! 2. `stft_bitrev`:       Cooley-Tukey bit-reversal permutation (in-place, batched).
//! 3. `stft_butterfly`:    one Radix-2 DIT butterfly stage; dispatched `log₂(N)` times,
//!    each time with a distinct params bind group (group 1) carrying the stage index.
//! 4. `stft_scale_and_window`: scale by 1/N and apply Hann synthesis window → frame_data.
//! 5. `stft_inverse_ola`:  weighted overlap-add reconstruction → output signal.
//!
//! Complexity reduction: O(N²) per frame → O(N log N) per frame.
//! Formal basis: Cooley & Tukey (1965); Allen & Rabiner (1977) Theorem 1.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use num_complex::Complex32;
use wgpu::util::DeviceExt;

use crate::domain::error::{WgpuError, WgpuResult};

/// Workgroup size for the forward pass and OLA pass (matches `@workgroup_size(64)` in
/// `stft.wgsl` and `stft_inverse.wgsl`).
const WORKGROUP_SIZE: u32 = 64;

/// Workgroup size for the four FFT inverse passes (matches `@workgroup_size(256)` in
/// `stft_inverse_fft.wgsl`).
const FFT_WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ComplexPod {
    re: f32,
    im: f32,
}

/// Uniform parameter block for forward pass and OLA pass.
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

/// Uniform parameter block for the FFT inverse passes.
///
/// Layout: 4 × u32 = 16 bytes, satisfying WGPU uniform alignment.
/// Field order matches the WGSL `FftStageParams` struct byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FftStageParams {
    frame_count: u32,
    frame_len: u32,
    stage: u32,
    _pad: u32,
}

/// GPU compute kernel encapsulating the forward and inverse STFT pipelines.
///
/// ## Bind group layouts
/// - `bind_group_layout`: 3-binding layout (read-only, read_write, uniform).
///   Used by the forward pass and the OLA inverse pass.
/// - `fft_data_bgl`: 4-binding layout for FFT inverse data buffers (group 0).
///   Bindings: interleaved spectrum (read-only), re scratch (read_write),
///   im scratch (read_write), frame_data (read_write).
/// - `fft_params_bgl`: 1-binding layout for per-stage FFT parameters (group 1).
#[derive(Debug)]
pub struct StftGpuKernel {
    /// 3-binding layout shared by forward and OLA passes.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Uniform params buffer for forward and OLA passes (StftParams).
    params_buffer: wgpu::Buffer,
    /// Pipeline for the direct-DFT forward pass.
    forward_pipeline: wgpu::ComputePipeline,
    /// Pipeline for the weighted overlap-add OLA reconstruction pass.
    inverse_ola_pipeline: wgpu::ComputePipeline,
    /// Data bind group layout for the FFT inverse passes (group 0, 4 bindings).
    fft_data_bgl: wgpu::BindGroupLayout,
    /// Per-stage params bind group layout for the FFT inverse passes (group 1, 1 uniform).
    fft_params_bgl: wgpu::BindGroupLayout,
    /// Deinterleave pipeline: interleaved complex f32 → split re/im buffers.
    deinterleave_pipeline: wgpu::ComputePipeline,
    /// Bit-reversal permutation pipeline.
    bitrev_pipeline: wgpu::ComputePipeline,
    /// Radix-2 DIT butterfly stage pipeline (one dispatch per stage).
    butterfly_pipeline: wgpu::ComputePipeline,
    /// Scale-by-1/N and Hann-window pipeline writing frame_data.
    scale_window_pipeline: wgpu::ComputePipeline,
}

impl StftGpuKernel {
    /// Create a new kernel by compiling all WGSL shaders and building all pipelines.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        // ── 3-binding layout (forward + OLA) ─────────────────────────────────
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

        // ── Forward shader & pipeline ─────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-stft-wgpu forward shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stft.wgsl").into()),
        });
        let forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-stft-wgpu forward pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("stft_forward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ── OLA inverse shader & pipeline (3-binding layout) ──────────────────
        let inverse_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-stft-wgpu inverse shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stft_inverse.wgsl").into()),
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

        // ── Shared uniform params buffer (StftParams) ─────────────────────────
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

        // ── FFT inverse: data bind group layout (group 0, 4 bindings) ─────────
        // binding 0: read-only storage (interleaved spectrum)
        // binding 1: read_write storage (re scratch)
        // binding 2: read_write storage (im scratch)
        // binding 3: read_write storage (frame_data)
        let fft_data_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-stft-wgpu FFT data BGL"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            ],
        });

        // ── FFT inverse: per-stage params bind group layout (group 1, 1 uniform) ─
        let fft_params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-stft-wgpu FFT params BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(std::mem::size_of::<FftStageParams>() as u64)
                            .expect("nonzero size"),
                    ),
                },
                count: None,
            }],
        });
        let fft_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-stft-wgpu FFT pipeline layout"),
            bind_group_layouts: &[&fft_data_bgl, &fft_params_bgl],
            push_constant_ranges: &[],
        });

        // ── FFT inverse shader & four pipelines ───────────────────────────────
        let fft_inv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-stft-wgpu FFT inverse shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stft_inverse_fft.wgsl").into()),
        });
        let deinterleave_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-stft-wgpu deinterleave pipeline"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_inv_shader,
                entry_point: Some("stft_deinterleave"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let bitrev_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-stft-wgpu bitrev pipeline"),
            layout: Some(&fft_pipeline_layout),
            module: &fft_inv_shader,
            entry_point: Some("stft_bitrev"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let butterfly_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-stft-wgpu butterfly pipeline"),
            layout: Some(&fft_pipeline_layout),
            module: &fft_inv_shader,
            entry_point: Some("stft_butterfly"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let scale_window_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-stft-wgpu scale-window pipeline"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_inv_shader,
                entry_point: Some("stft_scale_and_window"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            bind_group_layout,
            params_buffer,
            forward_pipeline,
            inverse_ola_pipeline,
            fft_data_bgl,
            fft_params_bgl,
            deinterleave_pipeline,
            bitrev_pipeline,
            butterfly_pipeline,
            scale_window_pipeline,
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

    /// Execute the inverse STFT via FFT-accelerated WOLA reconstruction.
    ///
    /// ## Algorithm
    /// 1. `stft_deinterleave`: interleaved complex f32 spectrum → split re/im scratch.
    /// 2. `stft_bitrev`: Cooley-Tukey bit-reversal permutation (in-place, batched).
    /// 3. `stft_butterfly` × log₂(frame_len): Radix-2 DIT butterfly stages (IDFT twiddle).
    /// 4. `stft_scale_and_window`: scale by 1/N, apply Hann window → frame_data.
    /// 5. `stft_inverse_ola`: weighted overlap-add → output signal.
    ///
    /// Complexity: O(N log N) per frame (reduced from O(N²)).
    ///
    /// ## Invariants
    /// - `frame_len` must be a power of two (Radix-2 requirement).
    /// - `spectrum.len() == frame_count * frame_len`
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
        // Radix-2 IFFT requires frame_len to be a power of two.
        if !frame_len.is_power_of_two() {
            return Err(WgpuError::FrameLenNotPowerOfTwo { frame_len });
        }
        let log2_n = frame_len.trailing_zeros();

        // ── Step 1: Build flat interleaved spectrum for GPU upload ────────────
        let spectrum_flat: Vec<f32> = spectrum.iter().flat_map(|c| [c.re, c.im]).collect();

        // ── Step 2: Allocate GPU buffers ──────────────────────────────────────
        let spectrum_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu inverse spectrum"),
            contents: bytemuck::cast_slice(&spectrum_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let scratch_size = (frame_count * frame_len * std::mem::size_of::<f32>()) as u64;
        // re_scratch: initially zeroed device-only scratch buffer.
        let re_scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu re scratch"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let im_scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu im scratch"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // frame_data: written by scale_window pass; read by OLA pass.
        let frame_data_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu frame data"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let signal_size = (signal_len * std::mem::size_of::<f32>()) as u64;
        let signal_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu inverse signal"),
            size: signal_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu inverse staging"),
            size: signal_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Step 3: Write OLA uniform params (StftParams) ─────────────────────
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

        // ── Step 4: Build the shared FFT data bind group (group 0) ────────────
        let fft_data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu FFT data bind group"),
            layout: &self.fft_data_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spectrum_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: re_scratch_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: im_scratch_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: frame_data_buf.as_entire_binding(),
                },
            ],
        });

        // ── Step 5: Pre-allocate per-stage params buffers and bind groups ─────
        // base_params_bg: used for deinterleave, bitrev, and scale_window passes
        // (stage field is unused by those entry points).
        let base_params = FftStageParams {
            frame_count: frame_count as u32,
            frame_len: frame_len as u32,
            stage: 0,
            _pad: 0,
        };
        let base_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu base FFT params"),
            contents: bytemuck::bytes_of(&base_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let base_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu base FFT params BG"),
            layout: &self.fft_params_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: base_params_buf.as_entire_binding(),
            }],
        });

        // One distinct params buffer + bind group per butterfly stage.
        // Retained in butterfly_bufs to keep GPU buffers alive until submit.
        let mut butterfly_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(log2_n as usize);
        let mut butterfly_bgs: Vec<wgpu::BindGroup> = Vec::with_capacity(log2_n as usize);
        for s in 0..log2_n {
            let stage_params = FftStageParams {
                frame_count: frame_count as u32,
                frame_len: frame_len as u32,
                stage: s,
                _pad: 0,
            };
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-stft-wgpu butterfly stage params"),
                contents: bytemuck::bytes_of(&stage_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu butterfly stage params BG"),
                layout: &self.fft_params_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            butterfly_bufs.push(buf);
            butterfly_bgs.push(bg);
        }

        // ── Step 6: Build the OLA bind group (3-binding layout) ──────────────
        // frame_data_buf (group 0 binding 3, read_write) is also bound here as
        // group 0 binding 0 (read-only) in the OLA bind group — different bind
        // groups, sequential passes, implicit barrier guarantees visibility.
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

        // ── Step 7: Encode all passes in one CommandEncoder ───────────────────
        // Each compute-pass boundary provides an implicit storage-buffer memory
        // barrier (WebGPU spec §3.4): every subsequent pass observes all writes
        // from all preceding passes.
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-stft-wgpu inverse encoder"),
        });

        // Pass 1: deinterleave — interleaved spectrum → split re/im scratch.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu deinterleave pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.deinterleave_pipeline);
            pass.set_bind_group(0, &fft_data_bg, &[]);
            pass.set_bind_group(1, &base_params_bg, &[]);
            pass.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        // Pass 2: bitrev — bit-reversal permutation.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu bitrev pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bitrev_pipeline);
            pass.set_bind_group(0, &fft_data_bg, &[]);
            pass.set_bind_group(1, &base_params_bg, &[]);
            pass.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        // Pass 3: butterfly stages — log₂(frame_len) Radix-2 DIT passes.
        for s in 0..log2_n as usize {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu butterfly pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.butterfly_pipeline);
            pass.set_bind_group(0, &fft_data_bg, &[]);
            pass.set_bind_group(1, &butterfly_bgs[s], &[]);
            pass.dispatch_workgroups(
                fft_dispatch_count((frame_count * frame_len / 2) as u32),
                1,
                1,
            );
        }

        // Pass 4: scale + window — apply 1/N scale and Hann synthesis window.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu scale-window pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scale_window_pipeline);
            pass.set_bind_group(0, &fft_data_bg, &[]);
            pass.set_bind_group(1, &base_params_bg, &[]);
            pass.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        // Pass 5: OLA — weighted overlap-add → output signal.
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

        // ── Step 8: Submit, poll, map, and collect output ─────────────────────
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

/// Returns the number of workgroups for the forward and OLA passes
/// (workgroup_size = 64, matching `stft.wgsl` and `stft_inverse.wgsl`).
fn dispatch_count(total: u32) -> u32 {
    total.div_ceil(WORKGROUP_SIZE)
}

/// Returns the number of workgroups for the FFT inverse passes
/// (workgroup_size = 256, matching `stft_inverse_fft.wgsl`).
fn fft_dispatch_count(total: u32) -> u32 {
    total.div_ceil(FFT_WORKGROUP_SIZE)
}
