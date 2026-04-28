use crate::application::execution::plan::stft::dimension_1d::StftPlan;
use crate::domain::contracts::error::StftResult;
use ndarray::Array1;
use num_complex::Complex64;

/// Forward STFT convenience wrapper.
///
/// Constructs a `StftPlan` with the given parameters and calls `forward`.
pub fn stft(
    signal: &Array1<f64>,
    frame_len: usize,
    hop_len: usize,
) -> StftResult<Array1<Complex64>> {
    StftPlan::new(frame_len, hop_len)?.forward(signal)
}

/// Inverse STFT convenience wrapper.
///
/// Constructs a `StftPlan` with the given parameters and calls `inverse`.
pub fn istft(
    spectrum: &Array1<Complex64>,
    frame_len: usize,
    hop_len: usize,
    signal_len: usize,
) -> StftResult<Array1<f64>> {
    StftPlan::new(frame_len, hop_len)?.inverse(spectrum, signal_len)
}
