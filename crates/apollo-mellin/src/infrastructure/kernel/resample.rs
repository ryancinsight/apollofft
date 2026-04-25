//! Mellin transform kernels: log-resampling, trapezoidal-rule moments, and direct DFT spectrum.
//!
//! ## Mathematical foundation
//!
//! The Mellin transform of f(r) on (0, inf) is M(s) = integral_0^inf f(r) r^(s-1) dr.
//!
//! **Theorem (Mellin-Fourier substitution)**: Under the substitution r = e^u,
//! the Mellin transform becomes M(s) = integral f(e^u) e^{su} du where g(u) = f(e^u).
//! Along the imaginary axis s = i*xi this is the Fourier transform of g:
//! M(i*xi) = integral g(u) e^{i*xi*u} du.
//!
//! **Consequence**: The Mellin spectrum on the imaginary axis equals the Fourier
//! transform of the log-resampled signal. calculate_log_resample performs the
//! r to e^u substitution; log_frequency_spectrum then applies the discrete Fourier sum.

use num_complex::Complex64;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 256;

/// Interpolate a positive-domain signal onto logarithmically spaced samples.
///
/// The output sample `i` evaluates the input at
/// `exp(log(min_scale) + i * du)` using linear interpolation in the original
/// positive coordinate. Values outside `[signal_min, signal_max]` map to zero.
pub fn calculate_log_resample(
    signal: &[f64],
    signal_min: f64,
    signal_max: f64,
    output: &mut [f64],
    min_scale: f64,
    max_scale: f64,
) {
    let samples = output.len();
    if samples == 0 {
        return;
    }

    let log_min = min_scale.ln();
    let log_max = max_scale.ln();
    let step = if samples > 1 {
        (log_max - log_min) / (samples as f64 - 1.0)
    } else {
        0.0
    };

    let signal_len = signal.len();
    if signal_len == 0 {
        output.fill(0.0);
        return;
    }

    let domain_width = signal_max - signal_min;

    let eval = |i: usize| -> f64 {
        let current_scale = (log_min + i as f64 * step).exp();

        if current_scale < signal_min || current_scale > signal_max || domain_width <= 0.0 {
            return 0.0;
        }

        let fraction = (current_scale - signal_min) / domain_width;
        let exact_idx = fraction * (signal_len as f64 - 1.0);

        let lower_idx = exact_idx.floor() as usize;
        let upper_idx = (lower_idx + 1).min(signal_len - 1);

        let weight = exact_idx - lower_idx as f64;

        signal[lower_idx] * (1.0 - weight) + signal[upper_idx] * weight
    };

    if samples >= PAR_THRESHOLD {
        output.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = eval(i);
        });
    } else {
        output.iter_mut().enumerate().for_each(|(i, val)| {
            *val = eval(i);
        });
    }
}

/// Evaluate the real Mellin moment `int f(r) r^(exponent - 1) dr`.
#[must_use]
pub fn mellin_moment(signal: &[f64], signal_min: f64, signal_max: f64, exponent: f64) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    if signal.len() == 1 {
        return signal[0] * moment_antiderivative(signal_min, signal_max, exponent);
    }

    let step = (signal_max - signal_min) / (signal.len() as f64 - 1.0);
    signal
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            let coordinate = signal_min + index as f64 * step;
            let weight = if index == 0 || index + 1 == signal.len() {
                0.5
            } else {
                1.0
            };
            weight * sample * coordinate.powf(exponent - 1.0)
        })
        .sum::<f64>()
        * step
}

/// Compute the direct log-frequency Mellin spectrum from log-domain samples.
///
/// The returned coefficient `k` is `du * sum_j g[j] exp(-2 pi i k j / N)`,
/// where `g[j] = f(exp(u_j))`. This is the DFT form of the imaginary-axis
/// Mellin transform after the substitution `r = exp(u)`.
#[must_use]
pub fn log_frequency_spectrum(log_samples: &[f64], log_min: f64, log_max: f64) -> Vec<Complex64> {
    let len = log_samples.len();
    if len == 0 {
        return Vec::new();
    }
    let du = if len > 1 {
        (log_max - log_min) / (len as f64 - 1.0)
    } else {
        1.0
    };
    let factor = -std::f64::consts::TAU / len as f64;
    let dft_coeff = |k: usize| -> Complex64 {
        du * log_samples
            .iter()
            .enumerate()
            .map(|(n, sample)| Complex64::from_polar(*sample, factor * k as f64 * n as f64))
            .sum::<Complex64>()
    };
    if len >= PAR_THRESHOLD {
        (0..len).into_par_iter().map(dft_coeff).collect()
    } else {
        (0..len).map(dft_coeff).collect()
    }
}

fn moment_antiderivative(min: f64, max: f64, exponent: f64) -> f64 {
    if exponent.abs() < f64::EPSILON {
        max.ln() - min.ln()
    } else {
        (max.powf(exponent) - min.powf(exponent)) / exponent
    }
}
