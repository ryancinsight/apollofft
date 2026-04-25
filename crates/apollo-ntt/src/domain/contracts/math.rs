//! Modular primitives and structural manipulations.

/// Verify the supported NTT length contract.
#[must_use]
pub fn is_valid_length(n: usize) -> bool {
    n > 0 && n.is_power_of_two()
}

/// Modular exponentiation.
#[must_use]
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, modulus);
        }
        base = mod_mul(base, base, modulus);
        exp >>= 1;
    }
    result
}

/// Modular multiplication with 128-bit widening.
#[must_use]
pub fn mod_mul(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    ((lhs as u128 * rhs as u128) % modulus as u128) as u64
}

/// Modular inverse via Fermat's little theorem for prime modulus.
#[must_use]
pub fn mod_inv(value: u64, modulus: u64) -> u64 {
    mod_pow(value, modulus - 2, modulus)
}

/// Add modulo `modulus`.
#[must_use]
pub fn mod_add(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    ((lhs as u128 + rhs as u128) % modulus as u128) as u64
}

/// Subtract modulo `modulus`.
#[must_use]
pub fn mod_sub(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    if lhs >= rhs {
        lhs - rhs
    } else {
        modulus - (rhs - lhs)
    }
}

/// Bit-reversal permutation in place.
pub fn bit_reverse_permute(data: &mut [u64]) {
    let n = data.len();
    assert!(
        is_valid_length(n),
        "NTT length must be a non-zero power of two"
    );
    let mut j = 0usize;
    for i in 1..n - 1 {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}
