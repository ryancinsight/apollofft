//! Analytic signal storage and derived envelope/phase observables.

use num_complex::Complex64;

/// Dense analytic signal `z[n] = x[n] + i H{x}[n]`.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalyticSignal {
    values: Vec<Complex64>,
}

impl AnalyticSignal {
    /// Create analytic-signal storage from computed values.
    #[must_use]
    pub fn new(values: Vec<Complex64>) -> Self {
        Self { values }
    }

    /// Return sample count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Return true when no samples are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Borrow complex analytic samples.
    #[must_use]
    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    /// Return the real original signal component.
    #[must_use]
    pub fn real(&self) -> Vec<f64> {
        self.values.iter().map(|value| value.re).collect()
    }

    /// Return the Hilbert quadrature component.
    #[must_use]
    pub fn quadrature(&self) -> Vec<f64> {
        self.values.iter().map(|value| value.im).collect()
    }

    /// Return the instantaneous envelope `|z[n]|`.
    #[must_use]
    pub fn envelope(&self) -> Vec<f64> {
        self.values.iter().map(|value| value.norm()).collect()
    }

    /// Return the wrapped instantaneous phase `atan2(imag, real)`.
    #[must_use]
    pub fn phase(&self) -> Vec<f64> {
        self.values.iter().map(|value| value.arg()).collect()
    }

    /// Consume storage and return complex samples.
    #[must_use]
    pub fn into_values(self) -> Vec<Complex64> {
        self.values
    }
}
