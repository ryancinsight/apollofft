//! Reusable Hilbert transform plan.

use crate::domain::contracts::error::{HilbertError, HilbertResult};
use crate::domain::metadata::length::SignalLength;
use crate::domain::signal::analytic::AnalyticSignal;
use crate::infrastructure::kernel::direct::{analytic_signal, hilbert_transform};

/// Reusable 1D Hilbert transform plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HilbertPlan {
    length: SignalLength,
}

impl HilbertPlan {
    /// Create a Hilbert transform plan for a non-empty signal length.
    pub fn new(len: usize) -> HilbertResult<Self> {
        Ok(Self {
            length: SignalLength::new(len)?,
        })
    }

    /// Return validated signal length.
    #[must_use]
    pub const fn length(self) -> SignalLength {
        self.length
    }

    /// Return signal length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.length.get()
    }

    /// Return true when signal length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.length.is_empty()
    }

    /// Compute the analytic signal `x + i H{x}`.
    pub fn analytic_signal(&self, signal: &[f64]) -> HilbertResult<AnalyticSignal> {
        if signal.len() != self.len() {
            return Err(HilbertError::LengthMismatch);
        }
        Ok(AnalyticSignal::new(analytic_signal(signal)?))
    }

    /// Compute only the Hilbert quadrature component.
    pub fn transform(&self, signal: &[f64]) -> HilbertResult<Vec<f64>> {
        if signal.len() != self.len() {
            return Err(HilbertError::LengthMismatch);
        }
        hilbert_transform(signal)
    }

    /// Compute the instantaneous envelope from the analytic signal.
    pub fn envelope(&self, signal: &[f64]) -> HilbertResult<Vec<f64>> {
        Ok(self.analytic_signal(signal)?.envelope())
    }

    /// Compute the wrapped instantaneous phase from the analytic signal.
    pub fn phase(&self, signal: &[f64]) -> HilbertResult<Vec<f64>> {
        Ok(self.analytic_signal(signal)?.phase())
    }
}
