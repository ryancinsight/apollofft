//! Reusable Discrete Hartley Transform plan.

use crate::domain::contracts::error::{DhtError, DhtResult};
use crate::domain::metadata::length::HartleyLength;
use crate::domain::spectrum::coefficients::HartleySpectrum;
use crate::infrastructure::kernel::direct::transform_real;
use crate::infrastructure::kernel::fast::dht_fast;

const FAST_KERNEL_THRESHOLD: usize = 512;

/// Reusable 1D real-to-real DHT plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DhtPlan {
    length: HartleyLength,
}

impl DhtPlan {
    /// Create a DHT plan for a non-empty signal length.
    pub fn new(len: usize) -> DhtResult<Self> {
        Ok(Self {
            length: HartleyLength::new(len)?,
        })
    }

    /// Return validated transform length.
    #[must_use]
    pub const fn length(self) -> HartleyLength {
        self.length
    }

    /// Return transform length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.length.get()
    }

    /// Return true when transform length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.length.is_empty()
    }

    /// Execute the unnormalized forward DHT.
    pub fn forward(&self, signal: &[f64]) -> DhtResult<HartleySpectrum> {
        let mut output = vec![0.0; self.len()];
        self.forward_into(signal, &mut output)?;
        Ok(HartleySpectrum::new(output))
    }

    /// Execute the unnormalized forward DHT into a zero-allocation buffer.
    pub fn forward_into(&self, signal: &[f64], output: &mut [f64]) -> DhtResult<()> {
        if signal.len() != self.len() || output.len() != self.len() {
            return Err(DhtError::LengthMismatch);
        }
        if self.len() >= FAST_KERNEL_THRESHOLD {
            dht_fast(signal, output);
            Ok(())
        } else {
            transform_real(signal, output)
        }
    }

    /// Execute the inverse DHT by reusing the forward kernel and applying `1 / N`.
    pub fn inverse(&self, spectrum: &HartleySpectrum) -> DhtResult<Vec<f64>> {
        let mut output = vec![0.0; self.len()];
        self.inverse_into(spectrum.values(), &mut output)?;
        Ok(output)
    }

    /// Execute the inverse DHT securely into a zero-allocation buffer.
    pub fn inverse_into(&self, spectrum: &[f64], output: &mut [f64]) -> DhtResult<()> {
        if spectrum.len() != self.len() || output.len() != self.len() {
            return Err(DhtError::LengthMismatch);
        }
        self.forward_into(spectrum, output)?;
        let scale = 1.0 / self.len() as f64;
        output.iter_mut().for_each(|value| *value *= scale);
        Ok(())
    }

    /// Apply one raw unnormalized DHT pass.
    pub fn transform_unscaled(&self, input: &[f64]) -> DhtResult<Vec<f64>> {
        let mut output = vec![0.0; self.len()];
        self.forward_into(input, &mut output)?;
        Ok(output)
    }
}
