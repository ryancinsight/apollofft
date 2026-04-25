//! Dense unitary quantum Fourier transform kernel.
//!
//! Forward entry M[k,j] = exp(2*pi*i*j*k/n) / sqrt(n).
//! Inverse entry M[k,j] = exp(-2*pi*i*j*k/n) / sqrt(n).
//! Both maps are unitary (norm-preserving) in exact arithmetic.

use num_complex::Complex64;

/// Forward dense QFT over a contiguous amplitude vector using precomputed twiddle factors.
///
/// `twiddles[k] = exp(2*pi*i*k/n)`. The entry `twiddles[(row*col) % n]` gives
/// `exp(2*pi*i*row*col/n)` without trigonometric calls at transform time.
#[must_use]
pub fn qft_forward_dense(input: &[Complex64], twiddles: &[Complex64]) -> Vec<Complex64> {
    qft_dense(input, twiddles, true)
}

/// Inverse dense QFT over a contiguous amplitude vector using precomputed twiddle factors.
#[must_use]
pub fn qft_inverse_dense(input: &[Complex64], twiddles: &[Complex64]) -> Vec<Complex64> {
    qft_dense(input, twiddles, false)
}

fn qft_dense(input: &[Complex64], twiddles: &[Complex64], forward: bool) -> Vec<Complex64> {
    let n = input.len();
    assert!(n > 0, "QFT length must be non-zero");
    let scale = 1.0 / (n as f64).sqrt();
    (0..n)
        .map(|row| {
            let sum: Complex64 = input
                .iter()
                .enumerate()
                .map(|(col, &value)| {
                    let tw = twiddles[(row * col) % n];
                    let twiddle = if forward { tw } else { tw.conj() };
                    value * twiddle
                })
                .sum();
            sum * scale
        })
        .collect()
}
