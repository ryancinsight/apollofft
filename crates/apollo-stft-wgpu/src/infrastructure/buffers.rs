//! Pre-allocated GPU buffer set for repeated STFT forward and inverse dispatches.
//!
//! Eliminates per-dispatch GPU buffer and bind-group allocation overhead by fixing the
//! plan shape `(frame_count, frame_len, signal_len, hop_len)` at construction time.
//!
//! ## Pattern
//! Mirrors `GpuFft3dBuffers` in `apollo-fft-wgpu`: construct once per plan shape;
//! call `StftWgpuBackend::execute_forward_with_buffers` / `execute_inverse_with_buffers`
//! to reuse allocations across repeated STFT dispatches.
//!
//! ## Formal contract
//! - `frame_len` must be a power of two (Radix-2 Cooley–Tukey constraint).
//! - `frame_count == 1 + signal_len.div_ceil(hop_len)` (caller invariant).
//! - Passing mismatched dimensions to the `_with_buffers` methods returns
//!   `WgpuError::LengthMismatch`.
//!
//! ## Eliminated allocations per dispatch
//! | Non-buffered call                  | Buffered call              |
//! |------------------------------------|----------------------------|
//! | 5–8 `device.create_buffer` calls  | 0 buffer allocations       |
//! | 4+ `device.create_bind_group`     | 0 bind-group allocations   |
//! | log₂(N) uniform buffer allocs     | 0 uniform allocs           |
//! | 1 `Vec<Complex32>` output alloc   | 0 host Vec allocations     |

use num_complex::Complex32;
use wgpu::util::DeviceExt;

use super::chirp::chirp_padded_len;
use super::kernel::{ComplexPod, FftStageParams, FwdFftStageParams, StftGpuKernel, StftParams};

/// Pre-allocated GPU buffer set for a fixed STFT plan shape.
///
/// Holds all GPU data buffers, staging buffers, bind groups, and per-butterfly-stage
/// uniform buffers for one `(frame_count, frame_len, signal_len, hop_len)` configuration.
///
/// ## Forward path buffers
/// - `signal_buf` (STORAGE | COPY_DST): input signal, written per call.
/// - `re_scratch_buf`, `im_scratch_buf` (STORAGE): shared FFT scratch (forward + inverse).
/// - `fwd_output_buf` (STORAGE | COPY_SRC): ComplexPod spectrum output.
/// - `fwd_staging_buf` (MAP_READ | COPY_DST): CPU readback of forward output.
///
/// ## Inverse path buffers
/// - `spectrum_buf` (STORAGE | COPY_DST): input spectrum, written per call.
/// - `frame_data_buf` (STORAGE): windowed frame data (scale_window → OLA).
/// - `inv_signal_buf` (STORAGE | COPY_SRC): reconstructed signal output.
/// - `inv_staging_buf` (MAP_READ | COPY_DST): CPU readback of inverse output.
/// - `inv_ola_params_buf` (UNIFORM | COPY_DST): OLA StftParams, written per call.
#[derive(Debug)]
pub struct StftGpuBuffers {
    pub(crate) frame_count: usize,
    pub(crate) frame_len: usize,
    pub(crate) signal_len: usize,
    pub(crate) hop_len: usize,
    pub(crate) log2_n: u32,
    // ── Data buffers ──────────────────────────────────────────────────────────
    pub(crate) signal_buf: wgpu::Buffer,
    pub(crate) spectrum_buf: wgpu::Buffer,
    /// GPU-only re scratch: written by pack_window/deinterleave; consumed by subsequent GPU
    /// butterfly passes. Never CPU-read; owned here to prevent premature buffer deallocation.
    #[allow(dead_code)]
    pub(crate) re_scratch_buf: wgpu::Buffer,
    /// GPU-only im scratch: same ownership contract as `re_scratch_buf`.
    #[allow(dead_code)]
    pub(crate) im_scratch_buf: wgpu::Buffer,
    pub(crate) fwd_output_buf: wgpu::Buffer,
    pub(crate) fwd_staging_buf: wgpu::Buffer,
    /// GPU-only frame data: written by scale_window pass; consumed by OLA pass. Never CPU-read.
    #[allow(dead_code)]
    pub(crate) frame_data_buf: wgpu::Buffer,
    pub(crate) inv_signal_buf: wgpu::Buffer,
    pub(crate) inv_staging_buf: wgpu::Buffer,
    pub(crate) inv_ola_params_buf: wgpu::Buffer,
    // ── Pre-built bind groups ─────────────────────────────────────────────────
    pub(crate) fwd_data_bg: wgpu::BindGroup,
    pub(crate) inv_data_bg: wgpu::BindGroup,
    pub(crate) ola_bg: wgpu::BindGroup,
    // ── Forward FFT stage params ──────────────────────────────────────────────
    // Buffers kept as fields to prevent drop before BG is valid (wgpu holds a Rc internally,
    // but explicit ownership avoids any refcount/drop-order ambiguity in the abstraction).
    #[allow(dead_code)]
    fwd_base_params_buf: wgpu::Buffer,
    pub(crate) fwd_base_params_bg: wgpu::BindGroup,
    #[allow(dead_code)]
    fwd_butterfly_bufs: Vec<wgpu::Buffer>,
    pub(crate) fwd_butterfly_bgs: Vec<wgpu::BindGroup>,
    // ── Inverse FFT stage params ──────────────────────────────────────────────
    #[allow(dead_code)]
    inv_base_params_buf: wgpu::Buffer,
    pub(crate) inv_base_params_bg: wgpu::BindGroup,
    #[allow(dead_code)]
    inv_butterfly_bufs: Vec<wgpu::Buffer>,
    pub(crate) inv_butterfly_bgs: Vec<wgpu::BindGroup>,
    // ── Host output vectors (reused across calls) ─────────────────────────────
    pub(crate) fwd_output_host: Vec<Complex32>,
    pub(crate) inv_output_host: Vec<f32>,
}

impl StftGpuBuffers {
    /// Allocate all GPU buffers and bind groups for the given plan shape.
    ///
    /// All bind groups are pre-built against `kernel`'s bind-group layouts and remain
    /// valid for the lifetime of this struct.
    ///
    /// Scratch buffers are automatically sized for both Radix-2 (power-of-two) and
    /// Bluestein/Chirp-Z (arbitrary) `frame_len` via `chirp_padded_len`.
    ///
    /// ## Panics
    /// Panics if `frame_len == 0`.
    ///
    /// ## Parameters
    /// - `device`: acquired WGPU device.
    /// - `kernel`: kernel whose BGL references are used for bind-group construction.
    /// - `frame_count`: `1 + signal_len.div_ceil(hop_len)`.
    /// - `frame_len`: FFT length per frame (arbitrary, not constrained to power of two).
    /// - `signal_len`: signal sample count.
    /// - `hop_len`: analysis hop size in samples.
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
        kernel: &StftGpuKernel,
        frame_count: usize,
        frame_len: usize,
        signal_len: usize,
        hop_len: usize,
    ) -> Self {
        assert!(
            frame_len != 0,
            "frame_len must be non-zero; got {frame_len}"
        );

        // Compute scratch element count: use padded length for non-PoT, original for PoT.
        let scratch_elem_count = frame_count
            * if frame_len.is_power_of_two() {
                frame_len
            } else {
                chirp_padded_len(frame_len)
            };
        let log2_n = frame_len.trailing_zeros();

        let signal_size = (signal_len * std::mem::size_of::<f32>()) as u64;
        let spectrum_size = (frame_count * frame_len * std::mem::size_of::<ComplexPod>()) as u64;
        let scratch_size = (scratch_elem_count * std::mem::size_of::<f32>()) as u64;
        let fwd_output_size = spectrum_size; // ComplexPod == 2×f32, same layout

        // ── Data buffers ──────────────────────────────────────────────────────
        let signal_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu reusable signal"),
            size: signal_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spectrum_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu reusable spectrum"),
            size: spectrum_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let re_scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu reusable re scratch"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let im_scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu reusable im scratch"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let fwd_output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu reusable fwd output"),
            size: fwd_output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let fwd_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu reusable fwd staging"),
            size: fwd_output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_data_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu reusable frame data"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let inv_signal_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu reusable inv signal"),
            size: signal_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let inv_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu reusable inv staging"),
            size: signal_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let inv_ola_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu reusable ola params"),
            contents: bytemuck::bytes_of(&StftParams {
                signal_len: 0,
                frame_len: 0,
                hop_len: 0,
                frame_count: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ── Bind groups ───────────────────────────────────────────────────────
        let fwd_data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu reusable fwd data BG"),
            layout: &kernel.fft_data_bgl,
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
                    resource: fwd_output_buf.as_entire_binding(),
                },
            ],
        });
        let inv_data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu reusable inv data BG"),
            layout: &kernel.fft_data_bgl,
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
        let ola_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu reusable OLA BG"),
            layout: &kernel.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: inv_signal_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: inv_ola_params_buf.as_entire_binding(),
                },
            ],
        });

        // ── Forward FFT stage params ──────────────────────────────────────────
        let fwd_base_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu reusable fwd base params"),
            contents: bytemuck::bytes_of(&FwdFftStageParams {
                frame_count: frame_count as u32,
                frame_len: frame_len as u32,
                hop_len: hop_len as u32,
                stage: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let fwd_base_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu reusable fwd base params BG"),
            layout: &kernel.fft_params_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: fwd_base_params_buf.as_entire_binding(),
            }],
        });
        let mut fwd_butterfly_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(log2_n as usize);
        let mut fwd_butterfly_bgs: Vec<wgpu::BindGroup> = Vec::with_capacity(log2_n as usize);
        for s in 0..log2_n {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-stft-wgpu reusable fwd butterfly params"),
                contents: bytemuck::bytes_of(&FwdFftStageParams {
                    frame_count: frame_count as u32,
                    frame_len: frame_len as u32,
                    hop_len: hop_len as u32,
                    stage: s,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu reusable fwd butterfly params BG"),
                layout: &kernel.fft_params_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            fwd_butterfly_bufs.push(buf);
            fwd_butterfly_bgs.push(bg);
        }

        // ── Inverse FFT stage params ──────────────────────────────────────────
        let inv_base_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu reusable inv base params"),
            contents: bytemuck::bytes_of(&FftStageParams {
                frame_count: frame_count as u32,
                frame_len: frame_len as u32,
                stage: 0,
                _pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let inv_base_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu reusable inv base params BG"),
            layout: &kernel.fft_params_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: inv_base_params_buf.as_entire_binding(),
            }],
        });
        let mut inv_butterfly_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(log2_n as usize);
        let mut inv_butterfly_bgs: Vec<wgpu::BindGroup> = Vec::with_capacity(log2_n as usize);
        for s in 0..log2_n {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-stft-wgpu reusable inv butterfly params"),
                contents: bytemuck::bytes_of(&FftStageParams {
                    frame_count: frame_count as u32,
                    frame_len: frame_len as u32,
                    stage: s,
                    _pad: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu reusable inv butterfly params BG"),
                layout: &kernel.fft_params_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            inv_butterfly_bufs.push(buf);
            inv_butterfly_bgs.push(bg);
        }

        Self {
            frame_count,
            frame_len,
            signal_len,
            hop_len,
            log2_n,
            signal_buf,
            spectrum_buf,
            re_scratch_buf,
            im_scratch_buf,
            fwd_output_buf,
            fwd_staging_buf,
            frame_data_buf,
            inv_signal_buf,
            inv_staging_buf,
            inv_ola_params_buf,
            fwd_data_bg,
            inv_data_bg,
            ola_bg,
            fwd_base_params_buf,
            fwd_base_params_bg,
            fwd_butterfly_bufs,
            fwd_butterfly_bgs,
            inv_base_params_buf,
            inv_base_params_bg,
            inv_butterfly_bufs,
            inv_butterfly_bgs,
            fwd_output_host: vec![Complex32::new(0.0, 0.0); frame_count * frame_len],
            inv_output_host: vec![0.0f32; signal_len],
        }
    }

    /// Return the frame count this buffer set was allocated for.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Return the FFT frame length this buffer set was allocated for.
    #[must_use]
    pub fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// Return the signal length this buffer set was allocated for.
    #[must_use]
    pub fn signal_len(&self) -> usize {
        self.signal_len
    }

    /// Return the hop length this buffer set was allocated for.
    #[must_use]
    pub fn hop_len(&self) -> usize {
        self.hop_len
    }

    /// Return the forward FFT output computed by the most recent
    /// `execute_forward_with_buffers` call.
    ///
    /// Length = `frame_count * frame_len`.
    #[must_use]
    pub fn fwd_output(&self) -> &[Complex32] {
        &self.fwd_output_host
    }

    /// Return the reconstructed signal computed by the most recent
    /// `execute_inverse_with_buffers` call.
    ///
    /// Length = `signal_len`.
    #[must_use]
    pub fn inv_output(&self) -> &[f32] {
        &self.inv_output_host
    }
}
