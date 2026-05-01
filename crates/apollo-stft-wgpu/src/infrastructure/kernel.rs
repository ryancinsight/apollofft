//! GPU compute kernels for the forward and inverse Short-Time Fourier Transform.
//!
//! Both paths are FFT-accelerated (O(N log N) per frame). Power-of-two `frame_len` uses the
//! Cooley-Tukey Radix-2 path; non-power-of-two `frame_len` uses the Bluestein/Chirp-Z reduction
//! (see `infrastructure::chirp::StftChirpData`).
//!
//! ## Forward dispatch sequence (stft_forward_fft.wgsl)
//! 1. `stft_fwd_pack_window`: Hann analysis window + pack to split re/im scratch.
//! 2. `stft_fwd_bitrev`:      Cooley-Tukey bit-reversal permutation (batched).
//! 3. `stft_fwd_butterfly`:   one Radix-2 DIT stage; dispatched `log₂(N)` times (DFT twiddle).
//! 4. `stft_fwd_interleave`:  split re/im → interleaved ComplexValue output.
//!
//! ## Inverse dispatch sequence (stft_inverse_fft.wgsl + stft_inverse.wgsl)
//! All passes are encoded in one `CommandEncoder` (implicit per-pass barriers):
//! 1. `stft_deinterleave`:    interleaved spectrum f32 → split re/im scratch.
//! 2. `stft_bitrev`:          bit-reversal permutation (batched).
//! 3. `stft_butterfly`:       one Radix-2 DIT stage; dispatched `log₂(N)` times (IDFT twiddle).
//! 4. `stft_scale_and_window`: scale by 1/N, Hann synthesis window → frame_data.
//! 5. `stft_inverse_ola`:     weighted overlap-add reconstruction → output signal.
//!
//! Formal basis: Cooley & Tukey (1965); Allen & Rabiner (1977) Theorem 1.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use num_complex::Complex32;
use wgpu::util::DeviceExt;

use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::chirp::{StftChirpData, chirp_padded_len};

/// Workgroup size for the OLA reconstruction pass (matches `@workgroup_size(64)` in
/// `stft_inverse.wgsl`).
const WORKGROUP_SIZE: u32 = 64;

/// Workgroup size for the four FFT inverse passes (matches `@workgroup_size(256)` in
/// `stft_inverse_fft.wgsl`).
const FFT_WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ComplexPod {
    pub(crate) re: f32,
    pub(crate) im: f32,
}

/// Uniform parameter block for forward pass and OLA pass.
///
/// Layout: 4 × u32 = 16 bytes, satisfying WGPU uniform alignment.
/// Field order matches the WGSL `StftParams` struct byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct StftParams {
    pub(crate) signal_len: u32,
    pub(crate) frame_len: u32,
    pub(crate) hop_len: u32,
    pub(crate) frame_count: u32,
}

/// Uniform parameter block for the FFT inverse passes.
///
/// Layout: 4 × u32 = 16 bytes, satisfying WGPU uniform alignment.
/// Field order matches the WGSL `FftStageParams` struct byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct FftStageParams {
    pub(crate) frame_count: u32,
    pub(crate) frame_len: u32,
    pub(crate) stage: u32,
    pub(crate) _pad: u32,
}

/// Uniform parameter block for the FFT forward passes.
///
/// Layout: 4 × u32 = 16 bytes, satisfying WGPU uniform alignment.
/// `hop_len` occupies the position that `_pad` occupies in `FftStageParams`;
/// the same `fft_params_bgl` (min_binding_size = 16) is therefore reusable for both paths.
/// Field order matches the WGSL `FwdFftParams` struct byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct FwdFftStageParams {
    pub(crate) frame_count: u32,
    pub(crate) frame_len: u32,
    pub(crate) hop_len: u32,
    pub(crate) stage: u32,
}

/// GPU compute kernel encapsulating the forward and inverse FFT-accelerated STFT pipelines.
///
/// ## Bind group layouts
/// - `bind_group_layout`: 3-binding layout (read-only, read_write, uniform).
///   Used exclusively by the OLA inverse reconstruction pass.
/// - `fft_data_bgl`: 4-binding layout shared by all FFT forward and inverse data passes (group 0).
///   Bindings: spectrum/signal (read-only), re scratch (read_write),
///   im scratch (read_write), frame_data/output (read_write).
/// - `fft_params_bgl`: 1-binding layout for per-stage FFT parameters (group 1).
#[derive(Debug)]
pub struct StftGpuKernel {
    /// 3-binding layout used by the OLA reconstruction pass.
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    /// Uniform params buffer for the OLA pass (StftParams).
    params_buffer: wgpu::Buffer,
    /// Pipeline for the weighted overlap-add OLA reconstruction pass.
    inverse_ola_pipeline: wgpu::ComputePipeline,
    /// Data bind group layout for the FFT inverse passes (group 0, 4 bindings).
    pub(crate) fft_data_bgl: wgpu::BindGroupLayout,
    /// Per-stage params bind group layout for the FFT inverse passes (group 1, 1 uniform).
    pub(crate) fft_params_bgl: wgpu::BindGroupLayout,
    /// Deinterleave pipeline: interleaved complex f32 → split re/im buffers.
    deinterleave_pipeline: wgpu::ComputePipeline,
    /// Bit-reversal permutation pipeline.
    bitrev_pipeline: wgpu::ComputePipeline,
    /// Radix-2 DIT butterfly stage pipeline (one dispatch per stage).
    butterfly_pipeline: wgpu::ComputePipeline,
    /// Scale-by-1/N and Hann-window pipeline writing frame_data.
    scale_window_pipeline: wgpu::ComputePipeline,
    /// Pack + Hann-analysis-window pipeline (pass 1 of forward FFT).
    fwd_pack_window_pipeline: wgpu::ComputePipeline,
    /// Bit-reversal permutation pipeline for the forward FFT path (pass 2).
    fwd_bitrev_pipeline: wgpu::ComputePipeline,
    /// Radix-2 DIT butterfly stage pipeline with DFT twiddle exp(−2πi) (pass 3).
    fwd_butterfly_pipeline: wgpu::ComputePipeline,
    /// Interleave split re/im → output ComplexValue pipeline (pass 4).
    fwd_interleave_pipeline: wgpu::ComputePipeline,
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

        // ── Forward FFT shader & four pipelines ───────────────────────────────────────
        let fft_fwd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-stft-wgpu FFT forward shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stft_forward_fft.wgsl").into()),
        });
        let fwd_pack_window_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-stft-wgpu fwd pack-window pipeline"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_fwd_shader,
                entry_point: Some("stft_fwd_pack_window"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fwd_bitrev_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-stft-wgpu fwd bitrev pipeline"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_fwd_shader,
                entry_point: Some("stft_fwd_bitrev"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fwd_butterfly_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-stft-wgpu fwd butterfly pipeline"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_fwd_shader,
                entry_point: Some("stft_fwd_butterfly"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fwd_interleave_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-stft-wgpu fwd interleave pipeline"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_fwd_shader,
                entry_point: Some("stft_fwd_interleave"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            bind_group_layout,
            params_buffer,
            inverse_ola_pipeline,
            fft_data_bgl,
            fft_params_bgl,
            deinterleave_pipeline,
            bitrev_pipeline,
            butterfly_pipeline,
            scale_window_pipeline,
            fwd_pack_window_pipeline,
            fwd_bitrev_pipeline,
            fwd_butterfly_pipeline,
            fwd_interleave_pipeline,
        }
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
        // Non-power-of-two frame_len: delegate to Bluestein/Chirp-Z path.
        if !frame_len.is_power_of_two() {
            return self.execute_inverse_chirp(
                device, queue, spectrum, frame_len, hop_len, frame_count, signal_len,
            );
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

    /// Execute the forward STFT via FFT-accelerated batch DFT (O(N log N) per frame).
    ///
    /// ## Algorithm
    /// 1. `stft_fwd_pack_window`: apply Hann analysis window to centred frames,
    ///    pack real windowed samples into split re/im scratch.
    /// 2. `stft_fwd_bitrev`: Cooley-Tukey bit-reversal permutation (in-place, batched).
    /// 3. `stft_fwd_butterfly` × log₂(frame_len): Radix-2 DIT butterfly stages (DFT twiddle).
    /// 4. `stft_fwd_interleave`: pack split re/im scratch → interleaved ComplexPod output.
    ///
    /// Complexity: O(N log N) per frame (reduced from O(N²)).
    ///
    /// ## Invariants
    /// - `frame_len` must be a power of two (Radix-2 requirement).
    pub fn execute_forward_fft(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        signal: &[f32],
        frame_len: usize,
        hop_len: usize,
        frame_count: usize,
    ) -> WgpuResult<Vec<Complex32>> {
        // Non-power-of-two frame_len: delegate to Bluestein/Chirp-Z path.
        if !frame_len.is_power_of_two() {
            return self.execute_forward_fft_chirp(
                device, queue, signal, frame_len, hop_len, frame_count,
            );
        }
        let log2_n = frame_len.trailing_zeros();

        let signal_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu fwd signal"),
            contents: bytemuck::cast_slice(signal),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let scratch_size = (frame_count * frame_len * std::mem::size_of::<f32>()) as u64;
        let re_scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu fwd re scratch"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let im_scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu fwd im scratch"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let out_size = (frame_count * frame_len * std::mem::size_of::<ComplexPod>()) as u64;
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu fwd output"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu fwd staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group 0: reuse fft_data_bgl (binding types are identical: ro, rw, rw, rw).
        let fft_fwd_data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu fwd FFT data BG"),
            layout: &self.fft_data_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: signal_buf.as_entire_binding(),
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
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        // Base params bind group: stage=0, hop_len filled.
        // Used for pack_window, bitrev, and interleave passes (stage field unused for these).
        let base_params = FwdFftStageParams {
            frame_count: frame_count as u32,
            frame_len: frame_len as u32,
            hop_len: hop_len as u32,
            stage: 0,
        };
        let base_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu fwd base params"),
            contents: bytemuck::bytes_of(&base_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let base_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu fwd base params BG"),
            layout: &self.fft_params_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: base_params_buf.as_entire_binding(),
            }],
        });

        // Per-butterfly-stage params bind groups (one per stage, stage index varies).
        let mut butterfly_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(log2_n as usize);
        let mut butterfly_bgs: Vec<wgpu::BindGroup> = Vec::with_capacity(log2_n as usize);
        for s in 0..log2_n {
            let stage_params = FwdFftStageParams {
                frame_count: frame_count as u32,
                frame_len: frame_len as u32,
                hop_len: hop_len as u32,
                stage: s,
            };
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-stft-wgpu fwd butterfly stage params"),
                contents: bytemuck::bytes_of(&stage_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu fwd butterfly stage params BG"),
                layout: &self.fft_params_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            butterfly_bufs.push(buf);
            butterfly_bgs.push(bg);
        }

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-stft-wgpu fwd encoder"),
        });

        // Pass 1: pack + Hann analysis window → split re/im scratch.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_pack_window"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_pack_window_pipeline);
            p.set_bind_group(0, &fft_fwd_data_bg, &[]);
            p.set_bind_group(1, &base_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }
        // Pass 2: bit-reversal permutation.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_bitrev"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_bitrev_pipeline);
            p.set_bind_group(0, &fft_fwd_data_bg, &[]);
            p.set_bind_group(1, &base_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }
        // Pass 3: Radix-2 DIT butterfly × log₂(frame_len).
        for s in 0..log2_n as usize {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_butterfly"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_butterfly_pipeline);
            p.set_bind_group(0, &fft_fwd_data_bg, &[]);
            p.set_bind_group(1, &butterfly_bgs[s], &[]);
            p.dispatch_workgroups(
                fft_dispatch_count((frame_count * frame_len / 2) as u32),
                1,
                1,
            );
        }
        // Pass 4: interleave split re/im → output ComplexValue array.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_interleave"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_interleave_pipeline);
            p.set_bind_group(0, &fft_fwd_data_bg, &[]);
            p.set_bind_group(1, &base_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        enc.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, out_size);
        queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..out_size);
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
        staging.unmap();
        Ok(output)
    }

    /// Execute the forward STFT using pre-allocated GPU buffers, avoiding per-dispatch
    /// buffer and bind-group creation overhead.
    ///
    /// Uploads `signal` to `buffers.signal_buf` via `queue.write_buffer`, then dispatches
    /// the 4-pass forward FFT pipeline using pre-built bind groups from `buffers`.
    /// Result is written into `buffers.fwd_output_host` and is accessible via
    /// `buffers.fwd_output()`.
    ///
    /// ## Errors
    /// Returns `WgpuError::LengthMismatch` if `signal.len() != buffers.signal_len()`.
    pub fn execute_forward_fft_with_buffers(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        signal: &[f32],
        buffers: &mut super::buffers::StftGpuBuffers,
    ) -> WgpuResult<()> {
        if signal.len() != buffers.signal_len() {
            return Err(WgpuError::LengthMismatch {
                expected: buffers.signal_len(),
                actual: signal.len(),
            });
        }

        let frame_count = buffers.frame_count();
        let frame_len = buffers.frame_len();
        let log2_n = buffers.log2_n;
        let out_size = (frame_count * frame_len * std::mem::size_of::<ComplexPod>()) as u64;

        queue.write_buffer(&buffers.signal_buf, 0, bytemuck::cast_slice(signal));

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-stft-wgpu fwd reuse encoder"),
        });

        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_pack_window (reuse)"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_pack_window_pipeline);
            p.set_bind_group(0, &buffers.fwd_data_bg, &[]);
            p.set_bind_group(1, &buffers.fwd_base_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_bitrev (reuse)"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_bitrev_pipeline);
            p.set_bind_group(0, &buffers.fwd_data_bg, &[]);
            p.set_bind_group(1, &buffers.fwd_base_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        for s in 0..log2_n as usize {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_butterfly (reuse)"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_butterfly_pipeline);
            p.set_bind_group(0, &buffers.fwd_data_bg, &[]);
            p.set_bind_group(1, &buffers.fwd_butterfly_bgs[s], &[]);
            p.dispatch_workgroups(
                fft_dispatch_count((frame_count * frame_len / 2) as u32),
                1,
                1,
            );
        }

        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_interleave (reuse)"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_interleave_pipeline);
            p.set_bind_group(0, &buffers.fwd_data_bg, &[]);
            p.set_bind_group(1, &buffers.fwd_base_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        enc.copy_buffer_to_buffer(
            &buffers.fwd_output_buf,
            0,
            &buffers.fwd_staging_buf,
            0,
            out_size,
        );

        queue.submit(std::iter::once(enc.finish()));

        let slice = buffers.fwd_staging_buf.slice(..out_size);
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

        {
            let m = slice.get_mapped_range();
            let pods = bytemuck::cast_slice::<u8, ComplexPod>(&m);
            for (slot, p) in buffers.fwd_output_host.iter_mut().zip(pods.iter()) {
                *slot = Complex32::new(p.re, p.im);
            }
        }

        buffers.fwd_staging_buf.unmap();
        Ok(())
    }

    /// Execute the inverse STFT using pre-allocated GPU buffers, avoiding per-dispatch
    /// buffer and bind-group creation overhead.
    ///
    /// Uploads `spectrum` to `buffers.spectrum_buf` via `queue.write_buffer`, writes
    /// OLA params to `buffers.inv_ola_params_buf`, then dispatches the 5-pass inverse FFT
    /// pipeline using pre-built bind groups from `buffers`.
    /// Result is written into `buffers.inv_output_host` and is accessible via
    /// `buffers.inv_output()`.
    ///
    /// ## Errors
    /// Returns `WgpuError::LengthMismatch` if `spectrum.len() != buffers.frame_count() * buffers.frame_len()`
    /// or `signal_len != buffers.signal_len()`.
    pub fn execute_inverse_with_buffers(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spectrum: &[Complex32],
        signal_len: usize,
        buffers: &mut super::buffers::StftGpuBuffers,
    ) -> WgpuResult<()> {
        let expected_spectrum = buffers.frame_count() * buffers.frame_len();
        if spectrum.len() != expected_spectrum {
            return Err(WgpuError::LengthMismatch {
                expected: expected_spectrum,
                actual: spectrum.len(),
            });
        }
        if signal_len != buffers.signal_len() {
            return Err(WgpuError::LengthMismatch {
                expected: buffers.signal_len(),
                actual: signal_len,
            });
        }

        let frame_count = buffers.frame_count();
        let frame_len = buffers.frame_len();
        let hop_len = buffers.hop_len();
        let log2_n = buffers.log2_n;
        let inv_signal_size = (signal_len * std::mem::size_of::<f32>()) as u64;

        let spectrum_flat: Vec<f32> = spectrum.iter().flat_map(|c| [c.re, c.im]).collect();
        queue.write_buffer(
            &buffers.spectrum_buf,
            0,
            bytemuck::cast_slice(&spectrum_flat),
        );

        queue.write_buffer(
            &buffers.inv_ola_params_buf,
            0,
            bytemuck::bytes_of(&StftParams {
                signal_len: signal_len as u32,
                frame_len: frame_len as u32,
                hop_len: hop_len as u32,
                frame_count: frame_count as u32,
            }),
        );

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-stft-wgpu inv reuse encoder"),
        });

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu deinterleave pass (reuse)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.deinterleave_pipeline);
            pass.set_bind_group(0, &buffers.inv_data_bg, &[]);
            pass.set_bind_group(1, &buffers.inv_base_params_bg, &[]);
            pass.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu bitrev pass (reuse)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bitrev_pipeline);
            pass.set_bind_group(0, &buffers.inv_data_bg, &[]);
            pass.set_bind_group(1, &buffers.inv_base_params_bg, &[]);
            pass.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        for s in 0..log2_n as usize {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu butterfly pass (reuse)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.butterfly_pipeline);
            pass.set_bind_group(0, &buffers.inv_data_bg, &[]);
            pass.set_bind_group(1, &buffers.inv_butterfly_bgs[s], &[]);
            pass.dispatch_workgroups(
                fft_dispatch_count((frame_count * frame_len / 2) as u32),
                1,
                1,
            );
        }

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu scale-window pass (reuse)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scale_window_pipeline);
            pass.set_bind_group(0, &buffers.inv_data_bg, &[]);
            pass.set_bind_group(1, &buffers.inv_base_params_bg, &[]);
            pass.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu inverse ola pass (reuse)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inverse_ola_pipeline);
            pass.set_bind_group(0, &buffers.ola_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(signal_len as u32), 1, 1);
        }

        enc.copy_buffer_to_buffer(
            &buffers.inv_signal_buf,
            0,
            &buffers.inv_staging_buf,
            0,
            inv_signal_size,
        );

        queue.submit(std::iter::once(enc.finish()));

        let slice = buffers.inv_staging_buf.slice(..inv_signal_size);
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

        {
            let m = slice.get_mapped_range();
            buffers
                .inv_output_host
                .copy_from_slice(bytemuck::cast_slice::<u8, f32>(&m));
        }

        buffers.inv_staging_buf.unmap();
        Ok(())
    }

    // ── Bluestein/Chirp-Z forward dispatch ────────────────────────────────────

    /// Execute the forward STFT for non-power-of-two `frame_len` via Bluestein's identity.
    ///
    /// Constructs `StftChirpData` once per call (allocates GPU resources for the
    /// padded chirp sub-FFTs), dispatches all passes in one encoder, then reads back
    /// the interleaved ComplexValue output.
    ///
    /// Complexity: O(N log N) per frame via Bluestein reduction to a padded Radix-2 FFT
    /// of length M = 2^⌈log₂(2N−1)⌉.
    ///
    /// Formal basis: Rabiner, Schafer & Rader (1969); Bluestein (1970).
    fn execute_forward_fft_chirp(
        &self,
        device:      &wgpu::Device,
        queue:       &wgpu::Queue,
        signal:      &[f32],
        frame_len:   usize,
        hop_len:     usize,
        frame_count: usize,
    ) -> WgpuResult<Vec<Complex32>> {
        use wgpu::util::DeviceExt;

        let signal_len = signal.len();
        let m = chirp_padded_len(frame_len);
        let log2_m = m.trailing_zeros();

        // Build StftChirpData (precomputes chirp kernel and all pipeline objects).
        let chirp = StftChirpData::new(device, queue, frame_len, frame_count, hop_len, signal_len);

        // Upload signal to GPU.
        let signal_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu chirp fwd signal"),
            contents: bytemuck::cast_slice(signal),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Output buffer: frame_count × frame_len × ComplexPod.
        let out_size = (frame_count * frame_len * std::mem::size_of::<ComplexPod>()) as u64;
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu chirp fwd output"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu chirp fwd staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // IO bind group for premul_fwd (binding 0 = signal, binding 1 = output_data).
        let io_bg_fwd = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu chirp fwd IO BG"),
            layout: &chirp.io_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: signal_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-stft-wgpu chirp fwd encoder"),
        });

        // Pass A: premul_fwd — Hann window + Bluestein exp(+πi·n²/N) premultiply.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_chirp_premul_fwd"),
                timestamp_writes: None,
            });
            p.set_pipeline(&chirp.premul_fwd_pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &chirp.chirp_params_bg, &[]);
            p.set_bind_group(2, &io_bg_fwd, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * m) as u32), 1, 1);
        }

        // Pass B: Radix-2 forward sub-FFT over M on chirp working buffers.
        self.dispatch_chirp_radix2(&mut enc, &chirp, frame_count, m, log2_m, false);

        // Pass C: pointmul — pointwise multiply by precomputed H in DFT domain.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("stft_chirp_pointmul_fwd"),
                timestamp_writes: None,
            });
                p.set_pipeline(&chirp.pointmul_fwd_pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &chirp.chirp_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * m) as u32), 1, 1);
        }

        // Pass D: Radix-2 inverse sub-FFT over M (+ 1/M scale).
        self.dispatch_chirp_radix2(&mut enc, &chirp, frame_count, m, log2_m, true);

        // Pass E: postmul_fwd — Bluestein exp(+πi·k²/N) postmultiply + write output.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_chirp_postmul_fwd"),
                timestamp_writes: None,
            });
            p.set_pipeline(&chirp.postmul_fwd_pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &chirp.chirp_params_bg, &[]);
            p.set_bind_group(2, &io_bg_fwd, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        enc.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, out_size);
        queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..out_size);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::PollType::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(WgpuError::BufferMapFailed { message: e.to_string() }),
            Err(e)     => return Err(WgpuError::BufferMapFailed { message: e.to_string() }),
        }
        let output = {
            let mapped = slice.get_mapped_range();
            bytemuck::cast_slice::<_, ComplexPod>(&mapped)
                .iter()
                .map(|p| Complex32::new(p.re, p.im))
                .collect()
        };
        staging.unmap();
        Ok(output)
    }

    /// Execute the inverse STFT for non-power-of-two `frame_len` via Bluestein's identity.
    ///
    /// Dispatches: premul_inv → Radix-2 forward → pointmul → Radix-2 inverse → postmul_inv → OLA.
    fn execute_inverse_chirp(
        &self,
        device:      &wgpu::Device,
        queue:       &wgpu::Queue,
        spectrum:    &[Complex32],
        frame_len:   usize,
        hop_len:     usize,
        frame_count: usize,
        signal_len:  usize,
    ) -> WgpuResult<Vec<f32>> {
        use wgpu::util::DeviceExt;

        let m = chirp_padded_len(frame_len);
        let log2_m = m.trailing_zeros();

        let chirp = StftChirpData::new(device, queue, frame_len, frame_count, hop_len, signal_len);

        // Flat interleaved spectrum for GPU upload.
        let spectrum_flat: Vec<f32> = spectrum.iter().flat_map(|c| [c.re, c.im]).collect();
        let spectrum_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu chirp inv spectrum"),
            contents: bytemuck::cast_slice(&spectrum_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // frame_data buffer: written by postmul_inv, read by OLA pass.
        let frame_data_size = (frame_count * frame_len * std::mem::size_of::<f32>()) as u64;
        let frame_data_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu chirp inv frame_data"),
            size: frame_data_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // OLA signal output + staging.
        let signal_size = (signal_len * std::mem::size_of::<f32>()) as u64;
        let signal_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu chirp inv signal out"),
            size: signal_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu chirp inv staging"),
            size: signal_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // IO bind group for premul_inv (binding 0 = interleaved spectrum, binding 1 = frame_data).
        let io_bg_inv = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu chirp inv IO BG"),
            layout: &chirp.io_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: spectrum_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: frame_data_buf.as_entire_binding() },
            ],
        });

        // OLA bind group (3-binding layout: frame_data ro, signal rw, params uniform).
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&StftParams {
                signal_len:  signal_len as u32,
                frame_len:   frame_len as u32,
                hop_len:     hop_len as u32,
                frame_count: frame_count as u32,
            }),
        );
        let ola_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu chirp inv OLA BG"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: frame_data_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: signal_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.params_buffer.as_entire_binding() },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-stft-wgpu chirp inv encoder"),
        });

        // Pass A: premul_inv.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_chirp_premul_inv"),
                timestamp_writes: None,
            });
            p.set_pipeline(&chirp.premul_inv_pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &chirp.chirp_params_bg, &[]);
            p.set_bind_group(2, &io_bg_inv, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * m) as u32), 1, 1);
        }

        // Pass B: Radix-2 forward sub-FFT over M.
        self.dispatch_chirp_radix2(&mut enc, &chirp, frame_count, m, log2_m, false);

        // Pass C: pointmul.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_chirp_pointmul"),
                timestamp_writes: None,
            });
            p.set_pipeline(&chirp.pointmul_pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &chirp.chirp_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * m) as u32), 1, 1);
        }

        // Pass D: Radix-2 inverse sub-FFT over M (+ 1/M scale).
        self.dispatch_chirp_radix2(&mut enc, &chirp, frame_count, m, log2_m, true);

        // Pass E: postmul_inv — conjugate postmul + 1/N scale + Hann window → frame_data.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_chirp_postmul_inv"),
                timestamp_writes: None,
            });
            p.set_pipeline(&chirp.postmul_inv_pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &chirp.chirp_params_bg, &[]);
            p.set_bind_group(2, &io_bg_inv, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        // Pass F: OLA reconstruction.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_inverse_ola (chirp)"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.inverse_ola_pipeline);
            p.set_bind_group(0, &ola_bg, &[]);
            p.dispatch_workgroups(dispatch_count(signal_len as u32), 1, 1);
        }

        enc.copy_buffer_to_buffer(&signal_buf, 0, &staging, 0, signal_size);
        queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..signal_size);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::PollType::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(WgpuError::BufferMapFailed { message: e.to_string() }),
            Err(e)     => return Err(WgpuError::BufferMapFailed { message: e.to_string() }),
        }
        let output = {
            let mapped = slice.get_mapped_range();
            bytemuck::cast_slice::<_, f32>(&mapped).to_vec()
        };
        staging.unmap();
        Ok(output)
    }

    /// Dispatch the Radix-2 sub-FFT passes of the Chirp-Z path over the chirp working buffers.
    ///
    /// `forward = false` → inverse sub-FFT (IDFT twiddles + 1/M scale).
    fn dispatch_chirp_radix2(
        &self,
        enc:         &mut wgpu::CommandEncoder,
        chirp:       &StftChirpData,
        frame_count: usize,
        m:           usize,
        log2_m:      u32,
        inverse:     bool,
    ) {
        let bgs = if inverse { &chirp.radix2_inv_bgs } else { &chirp.radix2_fwd_bgs };
        let bitrev_total     = (frame_count * m) as u32;
        let butterfly_total  = (frame_count * m / 2) as u32;

        // Bitrev pass (bgs[0]).
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chirp_fft_bitrev"),
                timestamp_writes: None,
            });
            let pipeline = if inverse {
                &chirp.chirp_bitrev_pipeline
            } else {
                &chirp.chirp_bitrev_pipeline
            };
            p.set_pipeline(pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &bgs[0], &[]);
            p.dispatch_workgroups(fft_dispatch_count(bitrev_total), 1, 1);
        }

        // Butterfly passes (bgs[1..=log2_m]).
        for s in 0..log2_m as usize {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chirp_fft_butterfly"),
                timestamp_writes: None,
            });
            let pipeline = if inverse {
                &chirp.chirp_inv_butterfly_pipeline
            } else {
                &chirp.chirp_fwd_butterfly_pipeline
            };
            p.set_pipeline(pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &bgs[1 + s], &[]);
            p.dispatch_workgroups(fft_dispatch_count(butterfly_total), 1, 1);
        }

        // Scale pass for inverse (bgs[log2_m + 1]).
        if inverse {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chirp_fft_scale"),
                timestamp_writes: None,
            });
            p.set_pipeline(&chirp.chirp_scale_pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, bgs.last().unwrap(), &[]);
            p.dispatch_workgroups(fft_dispatch_count(bitrev_total), 1, 1);
        }
    }
}

/// Returns the number of workgroups for the OLA reconstruction pass
/// (workgroup_size = 64, matching `@workgroup_size(64)` in `stft_inverse.wgsl`).
fn dispatch_count(total: u32) -> u32 {
    total.div_ceil(WORKGROUP_SIZE)
}

/// Returns the number of workgroups for the FFT forward and inverse passes
/// (workgroup_size = 256, matching `@workgroup_size(256)` in `stft_forward_fft.wgsl`
/// and `stft_inverse_fft.wgsl`).
fn fft_dispatch_count(total: u32) -> u32 {
    total.div_ceil(FFT_WORKGROUP_SIZE)
}
