// Short-Time Fourier Transform Bluestein/Chirp-Z kernel for non-power-of-two frame lengths.
//
// Implements Bluestein's identity (Rabiner, Schafer & Rader, 1969; Bluestein, 1970):
//
//   X[k] = W^{k²/2} · Σ_{n=0}^{N-1} (x[n]·W^{n²/2}) · W^{-(k-n)²/2}
//
// where W = exp(−2πi/N) for the forward DFT and W = exp(+2πi/N) for the IDFT.
//
// The inner sum is a convolution y = premul(x) * h, where h[n] = W^{-n²/2}.
// The convolution is evaluated via a zero-padded Radix-2 FFT of length M ≥ 2N−1,
// M = 2^⌈log₂(2N−1)⌉.
//
// ## Forward dispatch sequence (encoded in one CommandEncoder after the OLA pass):
//   Pass A (stft_chirp_premul_fwd):  x[n] ← hann(n,N)·x[n] · exp(+πi·n²/N)
//                                    padded region [N, M) zeroed per batch row.
//   Pass B: Radix-2 forward FFT over M (reuses stft_fwd_bitrev / stft_fwd_butterfly pipelines
//           operating on the chirp_re / chirp_im working buffers).
//   Pass C (stft_chirp_pointmul):    pointwise multiply working buffers by h_fft.
//   Pass D: Radix-2 inverse FFT over M (reuses stft_bitrev / stft_butterfly pipelines).
//   Pass E (stft_chirp_postmul_fwd): y[k] ← y[k] · exp(+πi·k²/N)  (k < N; truncate).
//                                    Write interleaved ComplexValue output.
//
// ## Inverse dispatch sequence:
//   Pass A (stft_chirp_premul_inv):  conjugate twiddle exp(−πi·n²/N).
//   Passes B–D: identical to forward.
//   Pass E (stft_chirp_postmul_inv): conjugate postmul exp(−πi·k²/N); scale by 1/N.
//                                    Write frame_data (f32, synthesis window applied here).
//
// ## Bind group layout:
//   group 0 — chirp data buffers:
//     binding 0: read_write — chirp_re  f32[frame_count · M]  (working real part)
//     binding 1: read_write — chirp_im  f32[frame_count · M]  (working imag part)
//     binding 2: read       — h_fft_re  f32[M]                (precomputed chirp kernel, real)
//     binding 3: read       — h_fft_im  f32[M]                (precomputed chirp kernel, imag)
//   group 1 — chirp parameters:
//     binding 0: uniform — StftChirpParams (32 bytes; 8 × u32)
//
//   group 2 — signal / output buffers (for pass A and pass E):
//     binding 0: read-only storage — signal_data f32[signal_len]  (forward premul pass)
//                OR read-only storage — spectrum    f32[2·Nf·N]    (inverse premul pass)
//     binding 1: read_write storage — output_data ComplexValue[Nf·N] (forward postmul pass)
//                OR read_write storage — frame_data  f32[Nf·N]       (inverse postmul pass)
//
// Formal bases:
//   Rabiner, Schafer & Rader (1969) "The Chirp z-Transform Algorithm".
//   Bluestein (1970) "A linear filtering approach to the computation of DFT".
//   DFT forward twiddle: W_N^{nk} = exp(−2πi·nk/N); forward premul twiddle = exp(+πi·n²/N).
//   DFT inverse twiddle: W_N^{-nk} = exp(+2πi·nk/N); inverse premul twiddle = exp(−πi·n²/N).

const TAU:   f32 = 6.28318530717958647692;
const PI:    f32 = 3.14159265358979323846;

struct ComplexValue {
    re: f32,
    im: f32,
}

/// Parameters for the Bluestein/Chirp-Z STFT passes.
///
/// Layout: 8 × u32 = 32 bytes, satisfying WGPU uniform alignment.
struct StftChirpParams {
    frame_count: u32,   // number of STFT frames (batch dimension)
    frame_len:   u32,   // original transform length N (non-power-of-two)
    chirp_len:   u32,   // padded Radix-2 length M ≥ 2N−1
    hop_len:     u32,   // analysis hop size in samples (forward premul only)
    signal_len:  u32,   // total signal length (forward premul only; unused in inverse)
    _pad0:       u32,
    _pad1:       u32,
    _pad2:       u32,
}

// Group 0: chirp working buffers and precomputed kernel.
@group(0) @binding(0) var<storage, read_write> chirp_re:  array<f32>;
@group(0) @binding(1) var<storage, read_write> chirp_im:  array<f32>;
@group(0) @binding(2) var<storage, read>       h_fft_re:  array<f32>;
@group(0) @binding(3) var<storage, read>       h_fft_im:  array<f32>;

// Group 1: chirp parameters.
@group(1) @binding(0) var<uniform> cp: StftChirpParams;

// Group 2: signal / output buffers (bound per-pass; see pass A and pass E).
@group(2) @binding(0) var<storage, read>       input_data:  array<f32>;   // signal (fwd) or interleaved spectrum (inv)
@group(2) @binding(1) var<storage, read_write> output_data: array<ComplexValue>;  // complex output (fwd postmul)

// ─── Helper: Hann analysis window ─────────────────────────────────────────────
fn hann_analysis(n: u32, frame_len: u32) -> f32 {
    if frame_len <= 1u {
        return 1.0;
    }
    return 0.5 - 0.5 * cos(TAU * f32(n) / f32(frame_len - 1u));
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass A (forward): stft_chirp_premul_fwd
//
// For each (frame_m, n) in [0, frame_count) × [0, chirp_len):
//   if n < frame_len:
//     center   = frame_m · hop_len
//     sig_idx  = center − frame_len/2 + n  (centred analysis window)
//     w_n      = hann_analysis(n, frame_len) · signal_data[sig_idx]  (0 if out of range)
//     arg      = π·n²/N
//     chirp_re[frame_m·M + n] = w_n · cos(arg)
//     chirp_im[frame_m·M + n] = w_n · sin(arg)
//   else:
//     chirp_re[frame_m·M + n] = 0.0
//     chirp_im[frame_m·M + n] = 0.0
//
// Total invocations: frame_count · chirp_len.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn stft_chirp_premul_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let total: u32 = cp.frame_count * cp.chirp_len;
    if idx >= total {
        return;
    }
    let frame_m: u32 = idx / cp.chirp_len;
    let n:       u32 = idx % cp.chirp_len;

    if n >= cp.frame_len {
        chirp_re[idx] = 0.0;
        chirp_im[idx] = 0.0;
        return;
    }

    // Centred analysis window: center = frame_m · hop_len; offset = n − frame_len/2.
    let center:  u32 = frame_m * cp.hop_len;
    let half:    u32 = cp.frame_len / 2u;
    var sig_re:  f32 = 0.0;
    if n + center >= half && n + center - half < cp.signal_len {
        let sig_idx: u32 = n + center - half;
        sig_re = hann_analysis(n, cp.frame_len) * input_data[sig_idx];
    }

    // Forward Bluestein premultiply: exp(+πi·n²/N).
    let arg:    f32 = PI * f32(n) * f32(n) / f32(cp.frame_len);
    let c:      f32 = cos(arg);
    let s:      f32 = sin(arg);
    chirp_re[idx] = sig_re * c;
    chirp_im[idx] = sig_re * s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass A (inverse): stft_chirp_premul_inv
//
// For each (frame_m, n) in [0, frame_count) × [0, chirp_len):
//   if n < frame_len:
//     re = input_data[2·(frame_m·frame_len + n)]
//     im = input_data[2·(frame_m·frame_len + n) + 1]
//     arg      = π·n²/N
//     // Inverse Bluestein premultiply: exp(−πi·n²/N) = conj(exp(+πi·n²/N)).
//     chirp_re[frame_m·M + n] = re · cos(arg) + im · sin(arg)
//     chirp_im[frame_m·M + n] = im · cos(arg) − re · sin(arg)
//   else: zero.
//
// Total invocations: frame_count · chirp_len.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn stft_chirp_premul_inv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let total: u32 = cp.frame_count * cp.chirp_len;
    if idx >= total {
        return;
    }
    let frame_m: u32 = idx / cp.chirp_len;
    let n:       u32 = idx % cp.chirp_len;

    if n >= cp.frame_len {
        chirp_re[idx] = 0.0;
        chirp_im[idx] = 0.0;
        return;
    }

    let spec_idx: u32 = 2u * (frame_m * cp.frame_len + n);
    let re: f32 = input_data[spec_idx];
    let im: f32 = input_data[spec_idx + 1u];

    // Inverse Bluestein premultiply: exp(−πi·n²/N) = (cos(arg), −sin(arg)).
    let arg: f32 = PI * f32(n) * f32(n) / f32(cp.frame_len);
    let c:   f32 = cos(arg);
    let s:   f32 = sin(arg);
    chirp_re[idx] =  re * c + im * s;
    chirp_im[idx] = -re * s + im * c;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass C: stft_chirp_pointmul
//
// Pointwise multiply the working buffers by the precomputed chirp kernel H (in DFT domain).
//
// For each j in [0, frame_count · chirp_len):
//   local_j = j % chirp_len
//   (chirp_re[j], chirp_im[j]) ← (chirp_re[j] + i·chirp_im[j]) · (h_fft_re[local_j] + i·h_fft_im[local_j])
//
// Total invocations: frame_count · chirp_len.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn stft_chirp_pointmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:     u32 = gid.x;
    let total:   u32 = cp.frame_count * cp.chirp_len;
    if idx >= total {
        return;
    }
    let local_j: u32 = idx % cp.chirp_len;
    let a_re: f32 = chirp_re[idx];
    let a_im: f32 = chirp_im[idx];
    let h_re: f32 = h_fft_re[local_j];
    let h_im: f32 = h_fft_im[local_j];
    chirp_re[idx] = a_re * h_re - a_im * h_im;
    chirp_im[idx] = a_re * h_im + a_im * h_re;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass E (forward): stft_chirp_postmul_fwd
//
// Postmultiply and write interleaved ComplexValue output (truncating to N outputs per frame).
//
// For each (frame_m, k) in [0, frame_count) × [0, frame_len):
//   y_re = chirp_re[frame_m·M + k]
//   y_im = chirp_im[frame_m·M + k]
//   arg  = π·k²/N
//   // Forward Bluestein postmultiply: exp(+πi·k²/N).
//   out_re = y_re · cos(arg) − y_im · sin(arg)
//   out_im = y_re · sin(arg) + y_im · cos(arg)
//   output_data[frame_m·N + k] = (out_re, out_im)
//
// Total invocations: frame_count · frame_len.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn stft_chirp_postmul_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:     u32 = gid.x;
    let total:   u32 = cp.frame_count * cp.frame_len;
    if idx >= total {
        return;
    }
    let frame_m: u32 = idx / cp.frame_len;
    let k:       u32 = idx % cp.frame_len;
    let chirp_idx: u32 = frame_m * cp.chirp_len + k;

    let y_re: f32 = chirp_re[chirp_idx];
    let y_im: f32 = chirp_im[chirp_idx];
    let arg:  f32 = PI * f32(k) * f32(k) / f32(cp.frame_len);
    let c:    f32 = cos(arg);
    let s:    f32 = sin(arg);
    output_data[idx] = ComplexValue(y_re * c - y_im * s, y_re * s + y_im * c);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass E (inverse): stft_chirp_postmul_inv
//
// Postmultiply with conjugate twiddle, scale by 1/N, apply Hann synthesis window,
// and write to frame_data (f32, real part only — imaginary discarded after IDFT).
//
// For each (frame_m, k) in [0, frame_count) × [0, frame_len):
//   y_re = chirp_re[frame_m·M + k]
//   y_im = chirp_im[frame_m·M + k]
//   arg  = π·k²/N
//   // Inverse Bluestein postmultiply: exp(−πi·k²/N) = conj.
//   x_re = (y_re · cos(arg) + y_im · sin(arg)) / (M · N)   [1/M from IFFT, 1/N from IDFT]
//   frame_data[frame_m·N + k] = hann(k, N) · x_re
//
// Total invocations: frame_count · frame_len.
// ─────────────────────────────────────────────────────────────────────────────

// Group 2 (inverse variant): frame_data f32 output for OLA pass.
// Declared in a separate entry point binding via a separate bind group layout in Rust;
// the WGSL global below shadows output_data for the inverse postmul entry only.
@group(2) @binding(1) var<storage, read_write> frame_data_out: array<f32>;

fn hann_synthesis(k: u32, frame_len: u32) -> f32 {
    if frame_len <= 1u {
        return 1.0;
    }
    return 0.5 - 0.5 * cos(TAU * f32(k) / f32(frame_len - 1u));
}

@compute @workgroup_size(256, 1, 1)
fn stft_chirp_postmul_inv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:     u32 = gid.x;
    let total:   u32 = cp.frame_count * cp.frame_len;
    if idx >= total {
        return;
    }
    let frame_m: u32 = idx / cp.frame_len;
    let k:       u32 = idx % cp.frame_len;
    let chirp_idx: u32 = frame_m * cp.chirp_len + k;

    let y_re: f32 = chirp_re[chirp_idx];
    let y_im: f32 = chirp_im[chirp_idx];
    let arg:  f32 = PI * f32(k) * f32(k) / f32(cp.frame_len);
    let c:    f32 = cos(arg);
    let s:    f32 = sin(arg);
    // Inverse postmultiply: exp(−πi·k²/N): (y_re·c + y_im·s, y_im·c − y_re·s).
    // Take real part; imaginary is zero for real-valued signals.
    // Scale: 1/M (absorbed by the Radix-2 IFFT) is handled by the stft_fwd_bitrev/stft_butterfly
    // Radix-2 paths which do NOT apply scale for the chirp sub-FFTs; the scale 1/N is applied
    // explicitly here to recover the IDFT normalisation.
    let x_re: f32 = (y_re * c + y_im * s) / f32(cp.frame_len);
    frame_data_out[idx] = hann_synthesis(k, cp.frame_len) * x_re;
}
