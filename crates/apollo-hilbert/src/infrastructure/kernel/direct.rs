//! Direct frequency-domain Hilbert kernel.
//!
//! The Hilbert transform is computed via the analytic signal mask applied in the
//! frequency domain. The forward and inverse DFT steps delegate to
//! `apollo_fft::fft_1d_array` and `apollo_fft::ifft_1d_complex`, which use the
//! O(N log N) radix-2/Bluestein strategy, replacing the former private O(N²)
//! direct-summation kernels and eliminating the SSOT violation documented in
//! `apollo-radon::infrastructure::kernel::filter`.

use crate::domain::contracts::error::{HilbertError, HilbertResult};
use ndarray::Array1;
use num_complex::Complex64;

/// Compute the Hilbert quadrature component of a real signal.
pub fn hilbert_transform(signal: &[f64]) -> HilbertResult<Vec<f64>> {
    Ok(analytic_signal(signal)?
        .into_iter()
        .map(|value| value.im)
        .collect())
}

/// Compute the analytic signal `x[n] + i H{x}[n]`.
///
/// # Theorem: Analytic Signal via Frequency-Domain Mask
///
/// For a real signal x ∈ ℝᴺ, the analytic signal z ∈ ℂᴺ is defined by
/// doubling the positive-frequency components of the DFT spectrum and zeroing
/// the negative-frequency components, then applying the IDFT:
///
/// ```text
/// Z[k] = { X[0]        k = 0
///         { 2 X[k]     1 ≤ k < N/2
///         { X[N/2]     k = N/2 (even N only)
///         { 0          N/2 < k < N
/// z[n] = IDFT(Z)[n];  re(z[n]) ← x[n]  (Hartley–Zygmund constraint)
/// ```
///
/// The Hilbert quadrature is `H{x}[n] = im(z[n])`.
///
/// # Proof sketch
///
/// The analytic signal z is the unique complex extension of x whose
/// negative-frequency content vanishes. Doubling positive frequencies preserves
/// the convolution-with-signum interpretation of the Hilbert transform
/// (`H{x} = IDFT(−i·sgn(k)·X[k])`). The DC and Nyquist bins are unscaled so
/// that `re(IDFT(Z)) = x` exactly, and the real-part is then forced from the
/// original input to eliminate rounding accumulation across the FFT/IFFT pair.
///
/// Reference: *The Analytic Signal*, Gabor D., J. IEE, 93(3):429–441, 1946.
///
/// # Complexity
///
/// O(N log N) — one O(N log N) FFT, O(N) mask application, one O(N log N) IFFT.
/// The previous O(N²) direct-summation implementation has been replaced.
pub fn analytic_signal(signal: &[f64]) -> HilbertResult<Vec<Complex64>> {
    if signal.is_empty() {
        return Err(HilbertError::EmptySignal);
    }

    let arr = Array1::from_iter(signal.iter().copied());
    let mut spectrum: Vec<Complex64> = apollo_fft::fft_1d_array(&arr).to_vec();
    apply_analytic_mask(&mut spectrum);
    let spectrum_arr = Array1::from_vec(spectrum);
    let mut analytic: Vec<Complex64> = apollo_fft::ifft_1d_complex(&spectrum_arr).to_vec();

    // Force the real part to equal the original input to eliminate IFFT rounding.
    for (sample, original) in analytic.iter_mut().zip(signal.iter()) {
        sample.re = *original;
    }
    Ok(analytic)
}

fn apply_analytic_mask(spectrum: &mut [Complex64]) {
    let len = spectrum.len();
    let positive_end = (len + 1) / 2;
    for (k, value) in spectrum.iter_mut().enumerate() {
        let scale = if k == 0 || (len % 2 == 0 && k == len / 2) {
            1.0
        } else if k < positive_end {
            2.0
        } else {
            0.0
        };
        *value *= scale;
    }
}
