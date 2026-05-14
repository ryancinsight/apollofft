use crate::application::execution::kernel::radix_shape::prime_factors;

pub(crate) fn primitive_root(p: usize) -> usize {
    // Static lookup table for common FFT prime sizes to avoid runtime factorization
    match p {
        2 => 1,
        3 => 2,
        5 => 2,
        7 => 3,
        11 => 2,
        13 => 2,
        17 => 3,
        19 => 2,
        23 => 5,
        29 => 2,
        31 => 3,
        37 => 2,
        41 => 6,
        43 => 3,
        47 => 5,
        53 => 2,
        59 => 2,
        61 => 2,
        67 => 2,
        71 => 7,
        73 => 5,
        79 => 3,
        83 => 2,
        89 => 3,
        97 => 5,
        10007 => 5,
        // Dynamic fusion fallback
        _ => primitive_root_dynamic(p),
    }
}

fn primitive_root_dynamic(p: usize) -> usize {
    if p == 2 {
        return 1;
    }
    let phi = p - 1;
    let mut factors = prime_factors(phi);
    factors.dedup(); // Only unique prime factors of P-1

    for g in 2..p {
        let mut is_primitive = true;
        for &q in &factors {
            if mod_pow(g, phi / q, p) == 1 {
                is_primitive = false;
                break;
            }
        }
        if is_primitive {
            return g;
        }
    }
    unreachable!("Prime must have a primitive root")
}

fn mod_pow(mut base: usize, mut exp: usize, modulo: usize) -> usize {
    let mut res = 1;
    base %= modulo;
    while exp > 0 {
        if exp % 2 == 1 {
            res = (res * base) % modulo;
        }
        base = (base * base) % modulo;
        exp /= 2;
    }
    res
}
