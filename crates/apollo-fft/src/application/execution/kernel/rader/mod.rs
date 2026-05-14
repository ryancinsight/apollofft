//! Rader's Algorithm for prime-length FFTs

pub(crate) mod generator;

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

// Rader's algorithm for prime lengths
/// Rader's Algorithm for prime N
pub(crate) fn rader_fft<F: MixedRadixScalar>(data: &mut [F::Complex], inverse: bool) {
    let n = data.len();
    debug_assert!(crate::application::execution::kernel::radix_shape::is_prime(n));

    let g = generator::primitive_root(n);
    let g_inv = mod_inverse(g, n);

    let x0 = data[0];

    F::with_rader_scratch(n - 1, |scratch| {
        // Construct permuted a directly into scratch FIRST before we overwrite data
        let mut curr = 1;
        for q in 0..(n - 1) {
            scratch[q] = data[curr];
            curr = (curr * g) % n;
        }

        // Compute sum_x and b directly into data[1..N] to save space
        let mut sum_x = F::complex(0.0, 0.0);
        let sign = if inverse { 1.0 } else { -1.0 };
        let mut curr_inv = 1;
        for q in 0..(n - 1) {
            sum_x = sum_x + data[1 + q];
            let angle = sign * std::f64::consts::TAU * (curr_inv as f64) / (n as f64);
            curr_inv = (curr_inv * g_inv) % n;
            data[1 + q] = F::complex(angle.cos(), angle.sin());
        }

        // Circular convolution of scratch (a) and data[1..N] (b)
        circular_convolution_inplace::<F>(scratch, &mut data[1..]);

        data[0] = x0 + sum_x;

        // Permute result back into data using g_inv_pow
        // We need to write a[q] to data[g_inv_pow[q]]. However, we can't do this in-place easily
        // because data[1..N] already holds the result (it was the second argument to convolution,
        // wait, circular_convolution_inplace overwrites the FIRST argument `a`).
        // So `scratch` holds the convolved output! data[1..N] is trashed by convolution, which is fine!

        curr_inv = 1;
        for q in 0..(n - 1) {
            data[curr_inv] = x0 + scratch[q];
            curr_inv = (curr_inv * g_inv) % n;
        }
    });
}

fn mod_inverse(a: usize, m: usize) -> usize {
    let mut m0 = m as i64;
    let mut y = 0i64;
    let mut x = 1i64;
    let mut a_i64 = a as i64;

    if m == 1 {
        return 0;
    }

    while a_i64 > 1 {
        let q = a_i64 / m0;
        let mut t = m0;
        m0 = a_i64 % m0;
        a_i64 = t;
        t = y;
        y = x - q * y;
        x = t;
    }

    if x < 0 {
        x += m as i64;
    }
    x as usize
}

// Deleted dead allocating function

fn circular_convolution_inplace<F: MixedRadixScalar>(a: &mut [F::Complex], b: &mut [F::Complex]) {
    let l = a.len();
    debug_assert_eq!(l, b.len());

    // For small or highly smooth lengths, we could use the inplace FFT directly.
    // However, to guarantee O(N log N) performance and avoid recursive Rader on large primes,
    // we use zero-padded linear convolution with a power-of-two FFT.
    // M must be >= 2L - 1
    let m = (2 * l - 1).next_power_of_two();

    F::with_rader_padded_scratch(m, |scratch_a| {
        F::with_rader_padded_scratch(m, |scratch_b| {
            // Pad a into scratch_a
            scratch_a[..l].copy_from_slice(a);
            for i in l..m {
                scratch_a[i] = F::complex(0.0, 0.0);
            }

            // Pad b into scratch_b
            scratch_b[..l].copy_from_slice(b);
            for i in l..m {
                scratch_b[i] = F::complex(0.0, 0.0);
            }

            // Forward Stockham FFTs (since M is power of 2)
            crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(scratch_a);
            crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(scratch_b);

            // Pointwise multiplication using precision-optimized method
            F::pointwise_mul(&mut scratch_a[..m], &scratch_b[..m]);

            // Inverse normalized FFT
            crate::application::execution::kernel::mixed_radix::inverse_inplace::<F>(scratch_a);

            // Alias linear convolution back into circular convolution of length L
            for n in 0..l {
                let tail = if n + l < m {
                    scratch_a[n + l]
                } else {
                    F::complex(0.0, 0.0)
                };
                a[n] = scratch_a[n] + tail;
            }
        });
    });
}
