// Short-Time Fourier Transform inverse kernel.
//
// Pass 1 (stft_inverse_frames): Per (frame m, local sample j):
//   frame_data[m·N + j] = (1/N) · Re{ Σ_k X[m,k]·exp(+2πi·k·j/N) } · w[j]
// where w[j] = 0.5 − 0.5·cos(2π·j/(N−1)) (Hann synthesis window), N = frame_len.
//
// IDFT real part expansion:
//   Re{ (re_k + i·im_k)·(cos θ + i·sin θ) } = re_k·cos θ − im_k·sin θ
//   where θ = 2π·k·j / N.
//
// Pass 2 (stft_inverse_ola): Per output sample n:
//   y[n] = Σ_m frame_data[m·N + (n − start_m)] / Σ_m w[n − start_m]²
// where start_m = m·hop_len − N/2, terms outside [0, N) are skipped.
//
// Formal basis: WOLA identity (Allen–Rabiner 1977, Theorem 1).
//
// Binding layout (shared between both passes):
//   @binding(0): read-only storage  — Pass 1: interleaved spectrum f32 pairs
//                                     Pass 2: frame_data f32 values
//   @binding(1): read_write storage — Pass 1: frame_data f32 values
//                                     Pass 2: output signal f32 values
//   @binding(2): uniform            — StftParams (16 bytes; layout matches Rust StftParams)

// StftParams must be 16 bytes to satisfy WGPU uniform alignment.
// Field order matches the Rust #[repr(C)] StftParams struct byte-for-byte.
struct StftParams {
    signal_len:  u32,   // byte offset 0
    frame_len:   u32,   // byte offset 4
    hop_len:     u32,   // byte offset 8
    frame_count: u32,   // byte offset 12
}

@group(0) @binding(0) var<storage, read>       input_data:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform>             params:      StftParams;

const TAU: f32 = 6.28318530717958647692;

// Hann synthesis window: w[j] = 0.5 − 0.5·cos(TAU·j / (N−1)).
// Returns 1.0 for N ≤ 1 (degenerate single-sample frame).
fn hann(j: u32, frame_len: u32) -> f32 {
    if frame_len <= 1u {
        return 1.0;
    }
    return 0.5 - 0.5 * cos(TAU * f32(j) / f32(frame_len - 1u));
}

// Pass 1: per-(frame, sample) IDFT with Hann window.
//
// Linear index: gid.x = frame_m * frame_len + j.
// Total invocations needed: frame_count * frame_len.
//
// input_data holds interleaved spectrum:
//   input_data[2*(m*frame_len+k)+0] = Re{X[m,k]}
//   input_data[2*(m*frame_len+k)+1] = Im{X[m,k]}
@compute @workgroup_size(64, 1, 1)
fn stft_inverse_frames(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_idx: u32 = gid.x;
    let total:      u32 = params.frame_count * params.frame_len;
    if linear_idx >= total {
        return;
    }

    let frame_m: u32 = linear_idx / params.frame_len;
    let j:       u32 = linear_idx % params.frame_len;
    let fl:      u32 = params.frame_len;

    // IDFT real part: (1/N) · Σ_k [re_k·cos(2π·k·j/N) − im_k·sin(2π·k·j/N)]
    var re_acc: f32 = 0.0;
    for (var k: u32 = 0u; k < fl; k = k + 1u) {
        let spec_base: u32 = 2u * (frame_m * fl + k);
        let re_k:      f32 = input_data[spec_base];
        let im_k:      f32 = input_data[spec_base + 1u];
        let angle:     f32 = TAU * f32(k) * f32(j) / f32(fl);
        re_acc = re_acc + re_k * cos(angle) - im_k * sin(angle);
    }

    let w: f32 = hann(j, fl);
    output_data[linear_idx] = (re_acc / f32(fl)) * w;
}

// Pass 2: weighted overlap-add reconstruction.
//
// Each invocation handles one output sample n ∈ [0, signal_len).
// input_data holds frame_data produced by Pass 1.
@compute @workgroup_size(64, 1, 1)
fn stft_inverse_ola(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n: u32 = gid.x;
    if n >= params.signal_len {
        return;
    }

    let fl:   u32 = params.frame_len;
    let half: i32 = i32(fl / 2u);

    var overlap: f32 = 0.0;
    var weight:  f32 = 0.0;

    for (var m: u32 = 0u; m < params.frame_count; m = m + 1u) {
        // start_m = m·hop_len − floor(frame_len / 2)  (signed)
        let start_m: i32 = i32(m) * i32(params.hop_len) - half;
        let local_n: i32 = i32(n) - start_m;
        if local_n >= 0 && u32(local_n) < fl {
            let j: u32 = u32(local_n);
            let w: f32 = hann(j, fl);
            overlap = overlap + input_data[m * fl + j];
            weight  = weight  + w * w;
        }
    }

    // select(false_val, true_val, cond): guard against zero denominator.
    // When weight = 0, select returns 0.0 regardless of overlap / weight.
    output_data[n] = select(0.0, overlap / weight, weight > 0.0);
}
