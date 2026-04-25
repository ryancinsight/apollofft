//! Direct frequency-domain Hilbert kernel.

use crate::domain::contracts::error::{HilbertError, HilbertResult};
use num_complex::Complex64;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 128;

/// Compute the Hilbert quadrature component of a real signal.
pub fn hilbert_transform(signal: &[f64]) -> HilbertResult<Vec<f64>> {
    Ok(analytic_signal(signal)?
        .into_iter()
        .map(|value| value.im)
        .collect())
}

/// Compute the analytic signal `x[n] + i H{x}[n]`.
pub fn analytic_signal(signal: &[f64]) -> HilbertResult<Vec<Complex64>> {
    if signal.is_empty() {
        return Err(HilbertError::EmptySignal);
    }

    let mut spectrum = forward_dft_real(signal);
    apply_analytic_mask(&mut spectrum);
    let mut analytic = inverse_dft_complex(&spectrum);

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

fn forward_dft_real(signal: &[f64]) -> Vec<Complex64> {
    let len = signal.len();
    let factor = -std::f64::consts::TAU / len as f64;
    if len >= PAR_THRESHOLD {
        (0..len)
            .into_par_iter()
            .map(|k| dft_real_coefficient(signal, factor, k))
            .collect()
    } else {
        (0..len)
            .map(|k| dft_real_coefficient(signal, factor, k))
            .collect()
    }
}

fn inverse_dft_complex(spectrum: &[Complex64]) -> Vec<Complex64> {
    let len = spectrum.len();
    let factor = std::f64::consts::TAU / len as f64;
    let scale = 1.0 / len as f64;
    if len >= PAR_THRESHOLD {
        (0..len)
            .into_par_iter()
            .map(|n| idft_complex_sample(spectrum, factor, scale, n))
            .collect()
    } else {
        (0..len)
            .map(|n| idft_complex_sample(spectrum, factor, scale, n))
            .collect()
    }
}

fn dft_real_coefficient(signal: &[f64], factor: f64, k: usize) -> Complex64 {
    signal
        .iter()
        .enumerate()
        .map(|(n, sample)| Complex64::from_polar(*sample, factor * k as f64 * n as f64))
        .sum()
}

fn idft_complex_sample(spectrum: &[Complex64], factor: f64, scale: f64, n: usize) -> Complex64 {
    scale
        * spectrum
            .iter()
            .enumerate()
            .map(|(k, coefficient)| {
                coefficient * Complex64::from_polar(1.0, factor * k as f64 * n as f64)
            })
            .sum::<Complex64>()
}
