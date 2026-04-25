struct ComplexValue {
    re: f32,
    im: f32,
}

struct HilbertParams {
    len: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0)
var<storage, read_write> inout_a: array<ComplexValue>;

@group(0) @binding(1)
var<storage, read_write> inout_b: array<ComplexValue>;

@group(0) @binding(2)
var<uniform> params: HilbertParams;

const TAU: f32 = 6.28318530717958647692;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

@compute @workgroup_size(64, 1, 1)
fn hilbert_forward_dft(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.len {
        return;
    }

    let factor = -TAU / f32(params.len);
    var acc = vec2<f32>(0.0, 0.0);
    for (var n: u32 = 0u; n < params.len; n = n + 1u) {
        let angle = factor * f32(k * n);
        let twiddle = vec2<f32>(cos(angle), sin(angle));
        let sample = vec2<f32>(inout_a[n].re, 0.0);
        acc = acc + cmul(sample, twiddle);
    }
    inout_b[k].re = acc.x;
    inout_b[k].im = acc.y;
}

@compute @workgroup_size(64, 1, 1)
fn hilbert_apply_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.len {
        return;
    }

    let positive_end = (params.len + 1u) / 2u;
    var scale = 0.0;
    if k == 0u || ((params.len & 1u) == 0u && k == params.len / 2u) {
        scale = 1.0;
    } else if k < positive_end {
        scale = 2.0;
    }
    inout_b[k].re = inout_b[k].re * scale;
    inout_b[k].im = inout_b[k].im * scale;
}

@compute @workgroup_size(64, 1, 1)
fn hilbert_inverse_dft(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if n >= params.len {
        return;
    }

    let factor = TAU / f32(params.len);
    let scale = 1.0 / f32(params.len);
    var acc = vec2<f32>(0.0, 0.0);
    for (var k: u32 = 0u; k < params.len; k = k + 1u) {
        let angle = factor * f32(k * n);
        let twiddle = vec2<f32>(cos(angle), sin(angle));
        let coefficient = vec2<f32>(inout_a[k].re, inout_a[k].im);
        acc = acc + cmul(coefficient, twiddle);
    }
    let original = inout_b[n].re;
    inout_b[n].re = original;
    inout_b[n].im = acc.y * scale;
}
