//! Reusable continuous wavelet transform plan.

use crate::domain::contracts::error::{WaveletError, WaveletResult};
use crate::domain::metadata::wavelet::ContinuousWavelet;
use crate::domain::spectrum::coefficients::CwtCoefficients;
use crate::infrastructure::kernel::continuous::coefficient;
use ndarray::Array2;
use rayon::prelude::*;

/// Reusable real-valued 1D CWT plan.
#[derive(Debug, Clone, PartialEq)]
pub struct CwtPlan {
    len: usize,
    scales: Vec<f64>,
    wavelet: ContinuousWavelet,
}

impl CwtPlan {
    /// Create a CWT plan for a real-valued signal length and scale list.
    pub fn new(len: usize, scales: Vec<f64>, wavelet: ContinuousWavelet) -> WaveletResult<Self> {
        if len == 0 {
            return Err(WaveletError::EmptySignal);
        }
        if scales.is_empty() {
            return Err(WaveletError::EmptyScales);
        }
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err(WaveletError::InvalidScale);
        }
        Ok(Self {
            len,
            scales,
            wavelet,
        })
    }

    /// Return signal length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Return true when signal length is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return scales.
    #[must_use]
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Return mother wavelet descriptor.
    #[must_use]
    pub const fn wavelet(&self) -> ContinuousWavelet {
        self.wavelet
    }

    /// Execute the CWT. Output shape is `(scales, signal_len)`.
    pub fn transform(&self, signal: &[f64]) -> WaveletResult<CwtCoefficients> {
        if signal.len() != self.len {
            return Err(WaveletError::LengthMismatch);
        }
        // Parallelize over the scale dimension; each scale row is independent.
        let rows: Vec<Vec<f64>> = self
            .scales
            .par_iter()
            .map(|&scale| {
                (0..self.len)
                    .map(|shift| coefficient(signal, self.wavelet, scale, shift))
                    .collect()
            })
            .collect();
        let values = Array2::from_shape_fn((self.scales.len(), self.len), |(s, b)| rows[s][b]);
        Ok(CwtCoefficients::new(self.scales.clone(), values))
    }
}
