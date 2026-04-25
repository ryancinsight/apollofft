struct ComplexValue {
    re: f32,
    im: f32,
}

struct CztParams {
    input_len: u32,
    output_len: u32,
    a_re: f32,
    a_im: f32,
    w_re: f32,
    w_im: f32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0)
var<storage, read> input_data: array<ComplexValue>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<ComplexValue>;

@group(0) @binding(2)
var<uniform> params: CztParams;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

fn complex_pow_real(z: vec2<f32>, exponent: f32) -> vec2<f32> {
    let radius = length(z);
    let angle = atan2(z.y, z.x);
    let scaled_radius = pow(radius, exponent);
    let scaled_angle = exponent * angle;
    return vec2<f32>(
        scaled_radius * cos(scaled_angle),
        scaled_radius * sin(scaled_angle)
    );
}

@compute @workgroup_size(64, 1, 1)
fn czt_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.output_len {
        return;
    }

    let a = vec2<f32>(params.a_re, params.a_im);
    let w = vec2<f32>(params.w_re, params.w_im);
    var acc = vec2<f32>(0.0, 0.0);

    for (var n: u32 = 0u; n < params.input_len; n = n + 1u) {
        let x = vec2<f32>(input_data[n].re, input_data[n].im);
        let a_pow = complex_pow_real(a, -f32(n));
        let w_pow = complex_pow_real(w, f32(n * k));
        let weight = cmul(a_pow, w_pow);
        acc = acc + cmul(x, weight);
    }

    output_data[k].re = acc.x;
    output_data[k].im = acc.y;
}
