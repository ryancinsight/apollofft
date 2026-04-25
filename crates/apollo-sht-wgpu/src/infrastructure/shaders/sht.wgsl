struct Complex32 {
    re: f32,
    im: f32,
}

struct ShtParams {
    output_count: u32,
    reduction_count: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0)
var<storage, read> input_values: array<Complex32>;

@group(0) @binding(1)
var<storage, read> basis_values: array<Complex32>;

@group(0) @binding(2)
var<storage, read_write> output_values: array<Complex32>;

@group(0) @binding(3)
var<uniform> params: ShtParams;

fn complex_mul(a: Complex32, b: Complex32) -> Complex32 {
    return Complex32(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

fn matrix_sum(row: u32) -> Complex32 {
    var sum = Complex32(0.0, 0.0);
    for (var col: u32 = 0u; col < params.reduction_count; col = col + 1u) {
        let basis = basis_values[row * params.reduction_count + col];
        let term = complex_mul(input_values[col], basis);
        sum = Complex32(sum.re + term.re, sum.im + term.im);
    }
    return sum;
}

@compute @workgroup_size(64, 1, 1)
fn sht_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.output_count {
        return;
    }
    output_values[row] = matrix_sum(row);
}

@compute @workgroup_size(64, 1, 1)
fn sht_inverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.output_count {
        return;
    }
    output_values[row] = matrix_sum(row);
}
