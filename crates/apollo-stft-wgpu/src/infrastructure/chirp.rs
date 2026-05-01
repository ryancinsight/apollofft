//! Pre-allocated Bluestein/Chirp-Z resources for non-power-of-two `frame_len`.
//!
//! ## Algorithm
//! Bluestein's identity maps an N-point DFT to a linear convolution over a padded length
//! M = 2^⌈log₂(2N−1)⌉ (Rabiner, Schafer & Rader, 1969; Bluestein, 1970):
//!
//! ```text
//! X[k] = W^{k²/2} · Σ_{n=0}^{N-1} (x[n]·W^{n²/2}) · W^{-(k-n)²/2}
//! ```
//!
//! where W = exp(−2πi/N) for the forward DFT and W = exp(+2πi/N) for the IDFT.
//!
//! The convolution kernel h[n] = W^{-n²/2} is precomputed host-side, DFT'd to length M
//! using the Radix-2 FFT (via `apollo_fft::Fft1D`), and uploaded to GPU once at construction.
//!
//! ## Dispatch sequence (per `StftChirpData::dispatch_forward` / `dispatch_inverse`)
//! 1. `stft_chirp_premul_fwd` / `stft_chirp_premul_inv` — premultiply + pack to chirp buffers.
//! 2. Radix-2 forward FFT over M on the chirp working buffers.
//! 3. `stft_chirp_pointmul` — pointwise multiply by precomputed H.
//! 4. Radix-2 inverse FFT over M on the chirp working buffers.
//! 5. `stft_chirp_postmul_fwd` / `stft_chirp_postmul_inv` — postmultiply + write output.

use std::f64::consts::PI;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Uniform parameter block for all Chirp-Z passes.
///
/// Layout: 8 × u32 = 32 bytes (satisfies WGPU alignment).
/// Field order matches the WGSL `StftChirpParams` struct byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct StftChirpParamsPod {
    pub(crate) frame_count: u32,
    pub(crate) frame_len:   u32,
    pub(crate) chirp_len:   u32,
    pub(crate) hop_len:     u32,
    pub(crate) signal_len:  u32,
    pub(crate) _pad0:       u32,
    pub(crate) _pad1:       u32,
    pub(crate) _pad2:       u32,
}

/// Pre-allocated GPU resources for the Bluestein/Chirp-Z STFT path.
///
/// One `StftChirpData` is constructed per unique `(frame_count, frame_len)` pair.
/// Construction uploads the precomputed chirp kernel H to the GPU; no per-dispatch
/// allocation is required.
#[allow(dead_code)]
pub(crate) struct StftChirpData {
    // ── Precomputed chirp kernel (length M, read-only after construction) ────
    pub(crate) _h_fft_re: wgpu::Buffer,
    pub(crate) _h_fft_im: wgpu::Buffer,

    // ── Working buffers (length frame_count × M, rewritten each dispatch) ───
    pub(crate) chirp_re_buf: wgpu::Buffer,
    pub(crate) chirp_im_buf: wgpu::Buffer,

    // ── Chirp BGL (group 0): [rw chirp_re, rw chirp_im, ro h_fft_re, ro h_fft_im] ──
    pub(crate) chirp_data_bgl: wgpu::BindGroupLayout,
    pub(crate) chirp_data_bg:  wgpu::BindGroup,

    // ── Chirp params BGL (group 1): 1 uniform (StftChirpParamsPod) ──────────
    pub(crate) chirp_params_bgl:  wgpu::BindGroupLayout,
    pub(crate) _chirp_params_buf: wgpu::Buffer,
    pub(crate) chirp_params_bg:   wgpu::BindGroup,

    // ── IO BGL (group 2): [ro input, rw output] ──────────────────────────────
    /// Used for both the forward premul pass (signal input, complex output slot unused)
    /// and the forward postmul pass (unused input slot, complex output).
    pub(crate) io_bgl: wgpu::BindGroupLayout,

    // ── Radix-2 sub-FFT pipeline layout (groups 0 + 1 = chirp_data + chirp_params) ──
    pub(crate) radix2_pipeline_layout: wgpu::PipelineLayout,

    // ── Precomputed Radix-2 stage bind groups for the chirp sub-FFTs ─────────
    /// Forward sub-FFT stage bind groups (log₂M + 1 bind groups).
    /// bgs[0] = bitrev params; bgs[1..] = butterfly stage params.
    pub(crate) radix2_fwd_bgs: Vec<wgpu::BindGroup>,
    /// Inverse sub-FFT stage bind groups (log₂M + 2 bind groups).
    pub(crate) radix2_inv_bgs: Vec<wgpu::BindGroup>,
    /// Retained param buffers (keep alive for bind groups).
    pub(crate) _radix2_param_bufs: Vec<wgpu::Buffer>,

    // ── Pipelines ─────────────────────────────────────────────────────────────
    pub(crate) premul_fwd_pipeline:  wgpu::ComputePipeline,
    pub(crate) premul_inv_pipeline:  wgpu::ComputePipeline,
    pub(crate) pointmul_pipeline:    wgpu::ComputePipeline,
    pub(crate) postmul_fwd_pipeline: wgpu::ComputePipeline,
    pub(crate) postmul_inv_pipeline: wgpu::ComputePipeline,
    /// Radix-2 bitrev pipeline operating on chirp_re/im (group 0 = chirp_data_bg).
    pub(crate) chirp_bitrev_pipeline:    wgpu::ComputePipeline,
    /// Radix-2 butterfly pipeline for forward sub-FFT on chirp buffers.
    pub(crate) chirp_fwd_butterfly_pipeline: wgpu::ComputePipeline,
    /// Radix-2 butterfly pipeline for inverse sub-FFT on chirp buffers.
    pub(crate) chirp_inv_butterfly_pipeline: wgpu::ComputePipeline,
    /// 1/M scale pipeline for the inverse sub-FFT.
    pub(crate) chirp_scale_pipeline: wgpu::ComputePipeline,
    /// Forward pointmul: multiply by conj(h_stored) = h_fwd = exp(+pi*i*j^2/N).
    pub(crate) pointmul_fwd_pipeline: wgpu::ComputePipeline,

    // ── Dimensions ────────────────────────────────────────────────────────────
    pub(crate) n:           u32,   // original frame_len
    pub(crate) m:           u32,   // padded Radix-2 length
    pub(crate) frame_count: u32,
    pub(crate) log2_m:      u32,
}

/// Compute M = 2^⌈log₂(2N−1)⌉ — the smallest power-of-two ≥ 2N−1.
pub(crate) fn chirp_padded_len(n: usize) -> usize {
    let min = 2 * n - 1;
    min.next_power_of_two()
}

impl StftChirpData {
    /// Construct all GPU resources for the Chirp-Z STFT path.
    ///
    /// `signal_len` and `hop_len` are stored in the params buffer so the premul pass
    /// can locate each frame in the signal.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device:      &wgpu::Device,
        queue:       &wgpu::Queue,
        n:           usize,  // frame_len
        frame_count: usize,
        hop_len:     usize,
        signal_len:  usize,
    ) -> Self {
        let m = chirp_padded_len(n);
        let log2_m = m.trailing_zeros();

        // ── Precompute chirp kernel h[j] = exp(−πi·j²/N), 0 ≤ j < M ─────────
        // h is the Bluestein convolution kernel: h[j] = exp(−πi·j²/N).
        // Circular-convolution symmetry: h[M−j] = h[j] for 1 ≤ j < n.
        // H = DFT_M(h) computed host-side via apollo_fft::fft_1d_complex (Complex64 API).
        let mut h_arr = ndarray::Array1::<num_complex::Complex64>::zeros(m);
        for j in 0..n {
            let arg = PI * (j * j) as f64 / n as f64;
            let val = num_complex::Complex64::new(arg.cos(), -arg.sin());
            h_arr[j] = val;
            if j > 0 {
                h_arr[m - j] = val;
            }
        }
        let h_fft_arr = apollo_fft::fft_1d_complex(&h_arr);
        let h_fft_re_data: Vec<f32> = h_fft_arr.iter().map(|c| c.re as f32).collect();
        let h_fft_im_data: Vec<f32> = h_fft_arr.iter().map(|c| c.im as f32).collect();

        // ── Upload chirp kernel to GPU ────────────────────────────────────────
        let h_fft_re_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu chirp h_fft_re"),
            contents: bytemuck::cast_slice(&h_fft_re_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let h_fft_im_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu chirp h_fft_im"),
            contents: bytemuck::cast_slice(&h_fft_im_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // ── Working buffers (frame_count × M × f32) ──────────────────────────
        let working_size = (frame_count * m * std::mem::size_of::<f32>()) as u64;
        let working_usage = wgpu::BufferUsages::STORAGE;
        let chirp_re_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu chirp_re"),
            size: working_size,
            usage: working_usage,
            mapped_at_creation: false,
        });
        let chirp_im_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu chirp_im"),
            size: working_size,
            usage: working_usage,
            mapped_at_creation: false,
        });

        // ── Chirp data BGL: [rw, rw, ro, ro] ─────────────────────────────────
        let chirp_data_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-stft-wgpu chirp data BGL"),
            entries: &[
                bgl_storage_entry(0, false),
                bgl_storage_entry(1, false),
                bgl_storage_entry(2, true),
                bgl_storage_entry(3, true),
            ],
        });
        let chirp_data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu chirp data BG"),
            layout: &chirp_data_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: chirp_re_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: chirp_im_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: h_fft_re_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: h_fft_im_buf.as_entire_binding() },
            ],
        });

        // ── Chirp params BGL: [uniform StftChirpParamsPod] ───────────────────
        let chirp_params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-stft-wgpu chirp params BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(
                            std::mem::size_of::<StftChirpParamsPod>() as u64,
                        )
                        .expect("nonzero"),
                    ),
                },
                count: None,
            }],
        });
        let chirp_params_pod = StftChirpParamsPod {
            frame_count: frame_count as u32,
            frame_len:   n as u32,
            chirp_len:   m as u32,
            hop_len:     hop_len as u32,
            signal_len:  signal_len as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let chirp_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu chirp params"),
            contents: bytemuck::bytes_of(&chirp_params_pod),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chirp_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-stft-wgpu chirp params BG"),
            layout: &chirp_params_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: chirp_params_buf.as_entire_binding(),
            }],
        });

        // ── IO BGL (group 2): [ro input, rw output] ──────────────────────────
        let io_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-stft-wgpu chirp IO BGL"),
            entries: &[
                bgl_storage_entry(0, true),
                bgl_storage_entry(1, false),
            ],
        });

        // ── Chirp Radix-2 sub-FFT params BGL: 1 uniform ──────────────────────
        // Reuses same layout as fft_params_bgl (16-byte uniform).
        // The chirp sub-FFT operates on the chirp working buffers (group 0 = chirp_data_bg).
        // We need a separate params BGL for sub-FFT stage params.
        let chirp_radix2_params_bgl = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("apollo-stft-wgpu chirp radix2 params BGL"),
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
            },
        );

        // ── Pipeline layouts ──────────────────────────────────────────────────
        // Chirp premul/postmul: groups 0 (chirp_data) + 1 (chirp_params) + 2 (io).
        let chirp_io_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-stft-wgpu chirp IO pipeline layout"),
                bind_group_layouts: &[&chirp_data_bgl, &chirp_params_bgl, &io_bgl],
                push_constant_ranges: &[],
            });
        // Chirp radix2 sub-FFT: groups 0 (chirp_data) + 1 (chirp_radix2_params).
        let radix2_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-stft-wgpu chirp radix2 pipeline layout"),
                bind_group_layouts: &[&chirp_data_bgl, &chirp_radix2_params_bgl],
                push_constant_ranges: &[],
            });
        // Chirp pointmul uses only groups 0 + 1.
        let chirp_pm_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-stft-wgpu chirp pointmul pipeline layout"),
                bind_group_layouts: &[&chirp_data_bgl, &chirp_params_bgl],
                push_constant_ranges: &[],
            });

        // ── Compile chirp shader ──────────────────────────────────────────────
        let chirp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-stft-wgpu chirp shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/stft_chirp.wgsl").into(),
            ),
        });

        // ── Compile chirp sub-FFT shader ──────────────────────────────────────
        // The chirp sub-FFTs operate on the chirp working buffers, which have a
        // different bind group layout (group 0 = chirp_data_bgl) than the STFT FFT passes.
        // We compile a dedicated sub-FFT shader targeting the chirp BGL.
        let chirp_fft_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-stft-wgpu chirp sub-FFT shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/stft_chirp_fft.wgsl").into(),
            ),
        });

        let build_pipeline = |layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let premul_fwd_pipeline  = build_pipeline(&chirp_io_pipeline_layout, &chirp_shader, "stft_chirp_premul_fwd");
        let premul_inv_pipeline  = build_pipeline(&chirp_io_pipeline_layout, &chirp_shader, "stft_chirp_premul_inv");
        let pointmul_pipeline    = build_pipeline(&chirp_pm_pipeline_layout, &chirp_shader, "stft_chirp_pointmul");
        let postmul_fwd_pipeline = build_pipeline(&chirp_io_pipeline_layout, &chirp_shader, "stft_chirp_postmul_fwd");
        let postmul_inv_pipeline = build_pipeline(&chirp_io_pipeline_layout, &chirp_shader, "stft_chirp_postmul_inv");
            let pointmul_fwd_pipeline = build_pipeline(&chirp_pm_pipeline_layout, &chirp_shader, "stft_chirp_pointmul_fwd");

        let chirp_bitrev_pipeline        = build_pipeline(&radix2_pipeline_layout, &chirp_fft_shader, "chirp_fft_bitrev");
        let chirp_fwd_butterfly_pipeline = build_pipeline(&radix2_pipeline_layout, &chirp_fft_shader, "chirp_fft_butterfly_fwd");
        let chirp_inv_butterfly_pipeline = build_pipeline(&radix2_pipeline_layout, &chirp_fft_shader, "chirp_fft_butterfly_inv");
        let chirp_scale_pipeline         = build_pipeline(&radix2_pipeline_layout, &chirp_fft_shader, "chirp_fft_scale");

        // ── Precompute Radix-2 stage bind groups ──────────────────────────────
        // Forward sub-FFT: bitrev + log₂M butterfly stages.
        // Inverse sub-FFT: bitrev + log₂M butterfly stages + scale stage.
        let fwd_stage_count = 1 + log2_m as usize;
        let inv_stage_count = fwd_stage_count + 1;
        let total_stage_count = fwd_stage_count + inv_stage_count;
        let mut radix2_param_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(total_stage_count);
        let mut radix2_fwd_bgs: Vec<wgpu::BindGroup> = Vec::with_capacity(fwd_stage_count);
        let mut radix2_inv_bgs: Vec<wgpu::BindGroup> = Vec::with_capacity(inv_stage_count);

        let push_radix2_bg = |data: [u32; 4], bgs: &mut Vec<wgpu::BindGroup>,
                                   bufs: &mut Vec<wgpu::Buffer>| {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-stft-wgpu chirp radix2 stage params"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu chirp radix2 stage params BG"),
                layout: &chirp_radix2_params_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            bufs.push(buf);
            bgs.push(bg);
        };

        // Forward sub-FFT bind groups: [m, 0, 0, frame_count] for bitrev,
        // then [m, stage, 0, frame_count] for each butterfly stage.
        push_radix2_bg([m as u32, 0, 0, frame_count as u32], &mut radix2_fwd_bgs, &mut radix2_param_bufs);
        for s in 0..log2_m {
            push_radix2_bg([m as u32, s, 0, frame_count as u32], &mut radix2_fwd_bgs, &mut radix2_param_bufs);
        }

        // Inverse sub-FFT bind groups: same as forward + one scale stage.
        push_radix2_bg([m as u32, 0, 1, frame_count as u32], &mut radix2_inv_bgs, &mut radix2_param_bufs);
        for s in 0..log2_m {
            push_radix2_bg([m as u32, s, 1, frame_count as u32], &mut radix2_inv_bgs, &mut radix2_param_bufs);
        }
        // Scale stage (1/M normalisation): reuse same params layout.
        push_radix2_bg([m as u32, 0, 1, frame_count as u32], &mut radix2_inv_bgs, &mut radix2_param_bufs);

        let _ = queue; // queue used for construction in future if needed

        Self {
            _h_fft_re: h_fft_re_buf,
            _h_fft_im: h_fft_im_buf,
            chirp_re_buf,
            chirp_im_buf,
            chirp_data_bgl,
            chirp_data_bg,
            chirp_params_bgl,
            _chirp_params_buf: chirp_params_buf,
            chirp_params_bg,
            io_bgl,
            radix2_pipeline_layout,
            radix2_fwd_bgs,
            radix2_inv_bgs,
            _radix2_param_bufs: radix2_param_bufs,
            premul_fwd_pipeline,
            premul_inv_pipeline,
            pointmul_pipeline,
            postmul_fwd_pipeline,
            postmul_inv_pipeline,
            chirp_bitrev_pipeline,
            chirp_fwd_butterfly_pipeline,
            chirp_inv_butterfly_pipeline,
            chirp_scale_pipeline,
                        pointmul_fwd_pipeline,
            n: n as u32,
            m: m as u32,
            frame_count: frame_count as u32,
            log2_m,
        }
    }
}

// ─── Helper: create a storage bind group layout entry ─────────────────────────
fn bgl_storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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
