//! Reusable Mellin plan metadata surface.

use crate::domain::contracts::error::{MellinError, MellinResult};
use crate::domain::metadata::scale::MellinScaleConfig;
use crate::infrastructure::kernel::resample::{
    calculate_log_resample, log_frequency_spectrum, mellin_moment,
};
use num_complex::Complex64;

/// Dense Mellin log-frequency spectrum.
#[derive(Debug, Clone, PartialEq)]
pub struct MellinSpectrum {
    values: Vec<Complex64>,
}

impl MellinSpectrum {
    /// Create spectrum storage from computed values.
    #[must_use]
    pub fn new(values: Vec<Complex64>) -> Self {
        Self { values }
    }

    /// Borrow spectrum coefficients.
    #[must_use]
    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    /// Return coefficient count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Return true when no coefficients are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Reusable Mellin transform plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MellinPlan {
    config: MellinScaleConfig,
}

impl MellinPlan {
    /// Create a validated Mellin transform plan.
    pub fn new(samples: usize, min_scale: f64, max_scale: f64) -> MellinResult<Self> {
        Ok(Self {
            config: MellinScaleConfig::new(samples, min_scale, max_scale)?,
        })
    }

    /// Return the validated scale configuration.
    #[must_use]
    pub const fn config(self) -> MellinScaleConfig {
        self.config
    }

    /// Resample an input signal onto this plan's logarithmic scale grid.
    pub fn forward_resample(
        &self,
        signal: &[f64],
        signal_min: f64,
        signal_max: f64,
        output: &mut [f64],
    ) -> MellinResult<()> {
        validate_signal_domain(signal, signal_min, signal_max)?;
        validate_output_len(output.len(), self.config.samples())?;

        calculate_log_resample(
            signal,
            signal_min,
            signal_max,
            output,
            self.config.min_scale(),
            self.config.max_scale(),
        );

        Ok(())
    }

    /// Evaluate the real Mellin moment `M(s) = int f(r) r^(s-1) dr`.
    pub fn moment(
        &self,
        signal: &[f64],
        signal_min: f64,
        signal_max: f64,
        exponent: f64,
    ) -> MellinResult<f64> {
        validate_signal_domain(signal, signal_min, signal_max)?;
        if !exponent.is_finite() {
            return Err(MellinError::InvalidExponent);
        }
        Ok(mellin_moment(signal, signal_min, signal_max, exponent))
    }

    /// Compute the direct log-frequency Mellin spectrum over this plan's scale grid.
    pub fn forward_spectrum(
        &self,
        signal: &[f64],
        signal_min: f64,
        signal_max: f64,
    ) -> MellinResult<MellinSpectrum> {
        let mut log_samples = vec![0.0; self.config.samples()];
        self.forward_resample(signal, signal_min, signal_max, &mut log_samples)?;
        Ok(MellinSpectrum::new(log_frequency_spectrum(
            &log_samples,
            self.config.min_scale().ln(),
            self.config.max_scale().ln(),
        )))
    }
}

fn validate_output_len(actual: usize, expected: usize) -> MellinResult<()> {
    if actual != expected {
        return Err(MellinError::LengthMismatch);
    }
    Ok(())
}

fn validate_signal_domain(signal: &[f64], signal_min: f64, signal_max: f64) -> MellinResult<()> {
    if signal.is_empty() {
        return Err(MellinError::EmptySignal);
    }
    if !signal_min.is_finite() || !signal_max.is_finite() || signal_min <= 0.0 || signal_max <= 0.0
    {
        return Err(MellinError::InvalidSignalBound);
    }
    if signal_min >= signal_max {
        return Err(MellinError::InvalidSignalOrder);
    }
    Ok(())
}
