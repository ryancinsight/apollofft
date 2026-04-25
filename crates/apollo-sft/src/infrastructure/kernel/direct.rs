//! Direct DFT kernel retained as a mathematical reference.
//!
//! **Note:** This function is O(N^2) and is no longer called in production
//! code. Forward and inverse sparse execution use Apollo FFT kernels
//! (`O(N log N)`) instead. This module is preserved as a ground-truth reference
//! for verification cross-checks.

#[cfg(test)]
use num_complex::Complex64;

/// Execute the dense DFT or inverse DFT over a complex slice.
///
/// # Mathematical contract
///
/// Forward:
/// X_k = sum_n x_n exp(-2*pi*i*k*n/N).
///
/// Inverse:
/// x_n = (1/N) sum_k X_k exp(2*pi*i*k*n/N).
#[must_use]
#[cfg(test)]
pub(crate) fn dft(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = input.len();
    let mut output = vec![Complex64::new(0.0, 0.0); n];
    if n == 0 {
        return output;
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let tau = std::f64::consts::TAU;
    let scale = if inverse { 1.0 / n as f64 } else { 1.0 };

    for (k, out) in output.iter_mut().enumerate() {
        let mut sum = Complex64::new(0.0, 0.0);
        for (n_idx, &x_n) in input.iter().enumerate() {
            let angle = sign * tau * (k as f64) * (n_idx as f64) / (n as f64);
            let twiddle = Complex64::new(angle.cos(), angle.sin());
            sum += x_n * twiddle;
        }
        *out = sum * scale;
    }

    output
}
