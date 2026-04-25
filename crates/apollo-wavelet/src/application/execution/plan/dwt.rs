//! Reusable discrete wavelet transform plan.

use crate::domain::contracts::error::{WaveletError, WaveletResult};
use crate::domain::metadata::wavelet::DiscreteWavelet;
use crate::domain::spectrum::coefficients::DwtCoefficients;
use crate::infrastructure::kernel::discrete::{analysis_stage_into, synthesis_stage_into};

/// Reusable 1D orthogonal DWT plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DwtPlan {
    len: usize,
    levels: usize,
    wavelet: DiscreteWavelet,
}

impl DwtPlan {
    /// Create a DWT plan for a power-of-two signal length.
    pub fn new(len: usize, levels: usize, wavelet: DiscreteWavelet) -> WaveletResult<Self> {
        if len == 0 {
            return Err(WaveletError::EmptySignal);
        }
        if !len.is_power_of_two() {
            return Err(WaveletError::NonPowerOfTwoLength);
        }
        if levels == 0 {
            return Err(WaveletError::EmptyLevelCount);
        }
        if levels > len.trailing_zeros() as usize {
            return Err(WaveletError::LevelExceedsLength);
        }
        Ok(Self {
            len,
            levels,
            wavelet,
        })
    }

    /// Return signal length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return true when signal length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Return decomposition level count.
    #[must_use]
    pub const fn levels(self) -> usize {
        self.levels
    }

    /// Return wavelet family.
    #[must_use]
    pub const fn wavelet(self) -> DiscreteWavelet {
        self.wavelet
    }

    /// Execute a multilevel forward DWT.
    pub fn forward(&self, signal: &[f64]) -> WaveletResult<DwtCoefficients> {
        if signal.len() != self.len {
            return Err(WaveletError::LengthMismatch);
        }
        let mut current = signal.to_vec();
        let mut details = Vec::with_capacity(self.levels);
        for _ in 0..self.levels {
            let half = current.len() / 2;
            let mut approximation = vec![0.0; half];
            let mut detail = vec![0.0; half];
            analysis_stage_into(&current, self.wavelet, &mut approximation, &mut detail);
            details.push(detail);
            current = approximation;
        }
        Ok(DwtCoefficients::new(
            self.len,
            self.levels,
            current,
            details,
        ))
    }

    /// Execute inverse multilevel DWT.
    pub fn inverse(&self, coefficients: &DwtCoefficients) -> WaveletResult<Vec<f64>> {
        if coefficients.len() != self.len || coefficients.levels() != self.levels {
            return Err(WaveletError::CoefficientShapeMismatch);
        }
        let mut current = coefficients.approximation().to_vec();
        for detail in coefficients.details().iter().rev() {
            let n = current.len() * 2;
            let mut output = vec![0.0; n];
            synthesis_stage_into(&current, detail, self.wavelet, &mut output);
            current = output;
        }
        Ok(current)
    }
}
