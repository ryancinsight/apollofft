//! Continuous wavelet transform analysis kernels.

use crate::domain::metadata::wavelet::ContinuousWavelet;

/// Evaluate the selected mother wavelet at normalized time `t`.
#[must_use]
pub fn mother_wavelet(wavelet: ContinuousWavelet, t: f64) -> f64 {
    match wavelet {
        ContinuousWavelet::Ricker => {
            let normalization = 2.0 / (3.0_f64.sqrt() * std::f64::consts::PI.powf(0.25));
            normalization * (1.0 - t * t) * (-0.5 * t * t).exp()
        }
        ContinuousWavelet::Morlet { omega0 } => {
            let correction = (-0.5 * omega0 * omega0).exp();
            std::f64::consts::PI.powf(-0.25)
                * ((omega0 * t).cos() - correction)
                * (-0.5 * t * t).exp()
        }
    }
}

/// Compute one real-valued CWT coefficient.
#[must_use]
pub fn coefficient(signal: &[f64], wavelet: ContinuousWavelet, scale: f64, shift: usize) -> f64 {
    let inv_sqrt_scale = 1.0 / scale.sqrt();
    signal
        .iter()
        .enumerate()
        .map(|(index, &sample)| {
            let normalized_time = (index as f64 - shift as f64) / scale;
            sample * inv_sqrt_scale * mother_wavelet(wavelet, normalized_time)
        })
        .sum()
}
