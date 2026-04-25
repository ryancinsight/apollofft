//! Projection-domain filters for filtered backprojection.

use num_complex::Complex64;

/// Apply the unwindowed ramp filter `|omega|` to one projection.
#[must_use]
pub fn ramp_filter_projection(projection: &[f64], detector_spacing: f64) -> Vec<f64> {
    let mut output = vec![0.0; projection.len()];
    ramp_filter_projection_into(projection, detector_spacing, &mut output);
    output
}

/// Apply the unwindowed ramp filter into caller-owned storage.
///
/// This avoids per-angle output allocation in filtered backprojection while
/// preserving the same direct DFT mathematical contract as
/// `ramp_filter_projection`.
pub fn ramp_filter_projection_into(projection: &[f64], detector_spacing: f64, output: &mut [f64]) {
    assert_eq!(
        projection.len(),
        output.len(),
        "ramp filter output length mismatch"
    );
    if projection.is_empty() {
        return;
    }

    let mut spectrum = forward_dft_real(projection);
    for (k, coefficient) in spectrum.iter_mut().enumerate() {
        *coefficient *= ramp_frequency(k, projection.len(), detector_spacing);
    }
    inverse_dft_real_into(&spectrum, output);
}

fn ramp_frequency(k: usize, len: usize, detector_spacing: f64) -> f64 {
    let signed_bin = if k <= len / 2 {
        k as f64
    } else {
        k as f64 - len as f64
    };
    std::f64::consts::TAU * signed_bin.abs() / (len as f64 * detector_spacing)
}

/// Direct O(N^2) forward DFT on a real signal.
///
/// ## SSOT note
/// This function duplicates a private kernel also present in apollo-hilbert.
/// A future refactor should extract shared DFT kernels into apollo-fft internal API.
fn forward_dft_real(signal: &[f64]) -> Vec<Complex64> {
    let len = signal.len();
    let factor = -std::f64::consts::TAU / len as f64;
    (0..len)
        .map(|k| {
            signal
                .iter()
                .enumerate()
                .map(|(n, sample)| Complex64::from_polar(*sample, factor * k as f64 * n as f64))
                .sum()
        })
        .collect()
}

fn inverse_dft_real_into(spectrum: &[Complex64], output: &mut [f64]) {
    let len = spectrum.len();
    debug_assert_eq!(len, output.len());
    let factor = std::f64::consts::TAU / len as f64;
    let scale = 1.0 / len as f64;
    for (n, value) in output.iter_mut().enumerate() {
        *value = scale
            * spectrum
                .iter()
                .enumerate()
                .map(|(k, coefficient)| {
                    coefficient * Complex64::from_polar(1.0, factor * k as f64 * n as f64)
                })
                .sum::<Complex64>()
                .re;
    }
}
