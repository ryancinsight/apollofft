//! Rader's Algorithm for prime-length FFTs.
//!
//! ## Circular convolution via direct DFT-{N-1}
//!
//! The standard Rader decomposition rewrites a length-N prime DFT as a
//! length-(N-1) circular convolution.  The precomputed kernel spectrum has
//! length N-1, and the convolution is computed via forward DFT-{N-1},
//! pointwise multiply, and normalized inverse DFT-{N-1}.  The DFT-{N-1} call
//! routes through the composite/PFA path (e.g., DFT-18 = 2×DFT-9 for N=19),
//! which is far shorter than the previous next_power_of_two(2*(N-1)-1) zero-
//! padded path (e.g., DFT-64 for N=19..31).

pub(crate) mod generator;

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use std::sync::Arc;

/// Rader's algorithm for prime N.
///
/// # Precondition
/// `data.len()` must be prime.
pub(crate) fn rader_fft<F: MixedRadixScalar>(data: &mut [F::Complex], inverse: bool) {
    let n = data.len();
    debug_assert!(crate::application::execution::kernel::radix_shape::is_prime(n));

    let g = generator::primitive_root(n);
    let g_inv = mod_inverse(g, n);
    let kernel_spectrum = F::cached_rader_spectrum(n, inverse, g_inv);
    let perm = cached_permutation(n, g, g_inv);

    let x0 = data[0];
    let l = n - 1;

    // Circular convolution of length N-1 via direct FFT (no zero-padding).
    // Rader's algorithm: X[g^{-k}] = x0 + circ_conv(y, h)[k], where
    //   y[q] = x[g^q mod N] and h is the precomputed kernel (DFT of W_N twiddles).
    // kernel_spectrum = DFT_{N-1}(h), length N-1.
    F::with_rader_padded_scratch(l, |padded| {
        // Fused gather: x[g^q mod n] → padded[q], accumulate x0 complement.
        let mut sum_x = F::complex(0.0, 0.0);
        for (q, &(input_idx, _)) in perm.iter().enumerate() {
            let v = data[input_idx];
            padded[q] = v;
            sum_x = sum_x + v;
        }

        // Forward DFT-{N-1}: routes through composite/PFA (far smaller than
        // the previous next_power_of_two(2*(N-1)-1) zero-padded path).
        crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(padded);

        F::pointwise_mul(padded, kernel_spectrum.as_ref());

        // Normalized inverse DFT-{N-1}: result is the circular convolution.
        crate::application::execution::kernel::mixed_radix::inverse_inplace::<F>(padded);

        data[0] = x0 + sum_x;
        for (q, &(_, output_idx)) in perm.iter().enumerate() {
            data[output_idx] = x0 + padded[q];
        }
    });
}

// ── Permutation cache ─────────────────────────────────────────────────────────

/// Returns cached pairs `(g^q mod n, g_inv^q mod n)` for `q = 0..n-1`.
///
/// The first element is the gather index (input permutation) and the second
/// is the scatter index (output permutation) for the Rader convolution.
fn cached_permutation(n: usize, g: usize, g_inv: usize) -> Arc<[(usize, usize)]> {
    crate::application::execution::kernel::mixed_radix::caches::cached_rader_perm(
        (n, g, g_inv),
        |(n, g, g_inv)| build_permutation(n, g, g_inv),
    )
}

fn build_permutation(n: usize, g: usize, g_inv: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(n - 1);
    let mut input_idx = 1usize;
    let mut output_idx = 1usize;
    for _ in 0..(n - 1) {
        pairs.push((input_idx, output_idx));
        input_idx = (input_idx * g) % n;
        output_idx = (output_idx * g_inv) % n;
    }
    pairs
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
