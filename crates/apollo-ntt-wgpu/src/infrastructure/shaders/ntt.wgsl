struct NttParams {
    len: u32,
    modulus: u32,
    root: u32,
    mode: u32,
}

@group(0) @binding(0)
var<storage, read> input_values: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output_values: array<u32>;

@group(0) @binding(2)
var<uniform> params: NttParams;

fn mod_add(lhs: u32, rhs: u32, modulus: u32) -> u32 {
    let sum = lhs + rhs;
    if sum >= modulus {
        return sum - modulus;
    }
    return sum;
}

fn mod_mul(lhs: u32, rhs: u32, modulus: u32) -> u32 {
    var a = lhs % modulus;
    var b = rhs;
    var result = 0u;
    loop {
        if b == 0u {
            break;
        }
        if (b & 1u) == 1u {
            result = mod_add(result, a, modulus);
        }
        a = mod_add(a, a, modulus);
        b = b >> 1u;
    }
    return result;
}

fn mod_pow(base_in: u32, exp_in: u32, modulus: u32) -> u32 {
    var base = base_in % modulus;
    var exp = exp_in;
    var result = 1u;
    loop {
        if exp == 0u {
            break;
        }
        if (exp & 1u) == 1u {
            result = mod_mul(result, base, modulus);
        }
        base = mod_mul(base, base, modulus);
        exp = exp >> 1u;
    }
    return result;
}

fn mod_inv(value: u32, modulus: u32) -> u32 {
    return mod_pow(value, modulus - 2u, modulus);
}

@compute @workgroup_size(64, 1, 1)
fn ntt_transform(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.len {
        return;
    }

    var omega = params.root;
    if params.mode == 1u {
        omega = mod_inv(params.root, params.modulus);
    }

    var sum = 0u;
    for (var j: u32 = 0u; j < params.len; j = j + 1u) {
        let exponent = (j * k) % params.len;
        let twiddle = mod_pow(omega, exponent, params.modulus);
        let term = mod_mul(input_values[j] % params.modulus, twiddle, params.modulus);
        sum = mod_add(sum, term, params.modulus);
    }

    if params.mode == 1u {
        let n_inv = mod_inv(params.len, params.modulus);
        sum = mod_mul(sum, n_inv, params.modulus);
    }

    output_values[k] = sum;
}
