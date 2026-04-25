// Short-Time Fourier Transform forward kernel.
//
// Each invocation computes one complex output element X[frame_m, bin_k]:
//
//   X[m, k] = sum_{n=0}^{frame_len-1}  w[n] * x[m*hop - frame_len/2 + n]
//             * exp(-2*pi*i*k*n / frame_len)
//
// where w[n] = 0.5 - 0.5 * cos(2*pi*n / (frame_len - 1)) (Hann window)
// and out-of-bounds signal samples are zero-padded.
//
// Linear index: gid.x = frame_m * frame_len + bin_k
// Total invocations needed: frame_count * frame_len.

struct ComplexValue {
    re: f32,
    im: f32,
}

// StftParams must be 16 bytes to satisfy WGPU uniform alignment.
struct StftParams {
    signal_len:  u32,
    frame_len:   u32,
    hop_len:     u32,
    frame_count: u32,
}

@group(0) @binding(0) var<storage, read>       signal_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<ComplexValue>;
@group(0) @binding(2) var<uniform>             params:      StftParams;

const TAU: f32 = 6.28318530717958647692;

// Hann analysis window: w[n] = 0.5 - 0.5 * cos(TAU * n / (frame_len - 1)).
// Returns 1.0 for frame_len == 1 (degenerate single-sample frame).
fn hann(n: u32, frame_len: u32) -> f32 {
    if frame_len <= 1u {
        return 1.0;
    }
    return 0.5 - 0.5 * cos(TAU * f32(n) / f32(frame_len - 1u));
}

@compute @workgroup_size(64, 1, 1)
fn stft_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_idx: u32 = gid.x;
    let total: u32 = params.frame_count * params.frame_len;
    if linear_idx >= total {
        return;
    }

    let frame_m: u32 = linear_idx / params.frame_len;
    let bin_k:   u32 = linear_idx % params.frame_len;
    let fl:      u32 = params.frame_len;

    // Signed center position: frame_m * hop_len
    // Frame start (signed): center - frame_len / 2
    let center: i32 = i32(frame_m) * i32(params.hop_len);
    let half:   i32 = i32(fl) / 2;

    var re_acc: f32 = 0.0;
    var im_acc: f32 = 0.0;

    for (var n: u32 = 0u; n < fl; n = n + 1u) {
        let sig_idx: i32 = center - half + i32(n);

        var sample: f32 = 0.0;
        if sig_idx >= 0 && u32(sig_idx) < params.signal_len {
            sample = signal_data[u32(sig_idx)];
        }

        let w: f32     = hann(n, fl);
        let angle: f32 = -TAU * f32(bin_k) * f32(n) / f32(fl);
        let ws: f32    = w * sample;

        re_acc = re_acc + ws * cos(angle);
        im_acc = im_acc + ws * sin(angle);
    }

    output_data[linear_idx].re = re_acc;
    output_data[linear_idx].im = im_acc;
}
