//! Dense real Hartley spectrum coefficients.

/// Dense real DHT spectrum.
#[derive(Debug, Clone, PartialEq)]
pub struct HartleySpectrum {
    values: Vec<f64>,
}

impl HartleySpectrum {
    /// Create spectrum storage from computed coefficient values.
    #[must_use]
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
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

    /// Borrow spectrum coefficients.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Consume storage and return coefficients.
    #[must_use]
    pub fn into_values(self) -> Vec<f64> {
        self.values
    }
}
