struct ComplexValue {
    re: f32,
    im: f32,
}

struct MellinParams {
    signal_len: u32,
    samples: u32,
    signal_min: f32,
    signal_max: f32,
    log_min: f32,
    log_max: f32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0)
var<storage, read> input_signal: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_values: array<f32>;

@group(0) @binding(2)
var<uniform> params: MellinParams;

@group(0) @binding(0)
var<storage, read> log_samples: array<f32>;

@group(0) @binding(1)
var<storage, read_write> spectrum_out: array<ComplexValue>;

const TAU: f32 = 6.28318530717958647692;

@compute @workgroup_size(64, 1, 1)
fn mellin_resample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.samples {
        return;
    }

    if params.signal_len == 0u {
        output_values[i] = 0.0;
        return;
    }

    let log_step = select(
        0.0,
        (params.log_max - params.log_min) / f32(params.samples - 1u),
        params.samples > 1u
    );
    let current_scale = exp(params.log_min + f32(i) * log_step);
    let domain_width = params.signal_max - params.signal_min;
    if current_scale < params.signal_min || current_scale > params.signal_max || domain_width <= 0.0 {
        output_values[i] = 0.0;
        return;
    }

    if params.signal_len == 1u {
        output_values[i] = input_signal[0u];
        return;
    }

    let fraction = (current_scale - params.signal_min) / domain_width;
    let exact_idx = fraction * f32(params.signal_len - 1u);
    let lower_idx = u32(floor(exact_idx));
    let upper_idx = min(lower_idx + 1u, params.signal_len - 1u);
    let weight = exact_idx - f32(lower_idx);
    output_values[i] = input_signal[lower_idx] * (1.0 - weight) + input_signal[upper_idx] * weight;
}

@compute @workgroup_size(64, 1, 1)
fn mellin_spectrum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.samples {
        return;
    }

    let du = select(
        1.0,
        (params.log_max - params.log_min) / f32(params.samples - 1u),
        params.samples > 1u
    );
    let factor = -TAU / f32(params.samples);
    var sum_re = 0.0;
    var sum_im = 0.0;
    for (var n: u32 = 0u; n < params.samples; n = n + 1u) {
        let angle = factor * f32(k * n);
        let sample = log_samples[n];
        sum_re = sum_re + sample * cos(angle);
        sum_im = sum_im + sample * sin(angle);
    }
    spectrum_out[k].re = du * sum_re;
    spectrum_out[k].im = du * sum_im;
}
