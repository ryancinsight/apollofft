//! Reusable dense quantum Fourier transform plan.
//!
//! For a state vector x in C^n, the forward QFT is
//! X_k = (1/sqrt(n)) sum_j x_j exp(2*pi*i*j*k/n). The inverse is the
//! conjugate transpose with the negative phase. Both maps are unitary.
//!
//! Twiddle factors exp(2*pi*i*k/n) for k=0..n are precomputed at plan
//! construction time and reused across all forward and inverse calls.

use crate::domain::contracts::error::{QftError, QftResult};
use crate::domain::state::dimension::QuantumStateDimension;
use crate::infrastructure::kernel::dense::{qft_forward_dense, qft_inverse_dense};
use ndarray::Array1;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Reusable QFT plan with precomputed twiddle factors.
///
/// `twiddles[k] = exp(2*pi*i*k/n)` for `k = 0..n`. The kernel indexes
/// `twiddles[(row*col) % n]` to obtain `exp(2*pi*i*row*col/n)` without
/// trigonometric evaluation per transform element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QftPlan {
    dimension: QuantumStateDimension,
    /// Precomputed twiddle factors: twiddles[k] = exp(2*pi*i*k/n).
    twiddles: Vec<Complex64>,
}

impl QftPlan {
    /// Create a QFT plan for a validated quantum state dimension.
    pub fn new(dimension: QuantumStateDimension) -> Self {
        let n = dimension.len();
        let twiddles: Vec<Complex64> = (0..n)
            .map(|k| {
                let angle = std::f64::consts::TAU * k as f64 / n as f64;
                Complex64::new(angle.cos(), angle.sin())
            })
            .collect();
        Self {
            dimension,
            twiddles,
        }
    }

    /// Return the plan length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.dimension.len()
    }

    /// Return true when the plan length is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dimension.is_empty()
    }

    /// Forward QFT of a complex amplitude vector.
    pub fn forward(&self, input: &Array1<Complex64>) -> QftResult<Array1<Complex64>> {
        if input.len() != self.len() {
            return Err(QftError::LengthMismatch);
        }
        Ok(Array1::from_vec(qft_forward_dense(
            input.as_slice().expect("QFT input must be contiguous"),
            &self.twiddles,
        )))
    }

    /// Inverse QFT of a complex amplitude vector.
    pub fn inverse(&self, input: &Array1<Complex64>) -> QftResult<Array1<Complex64>> {
        if input.len() != self.len() {
            return Err(QftError::LengthMismatch);
        }
        Ok(Array1::from_vec(qft_inverse_dense(
            input.as_slice().expect("QFT input must be contiguous"),
            &self.twiddles,
        )))
    }

    /// Forward QFT executed in place.
    pub fn forward_inplace(&self, data: &mut Array1<Complex64>) -> QftResult<()> {
        let transformed = self.forward(data)?;
        *data = transformed;
        Ok(())
    }

    /// Inverse QFT executed in place.
    pub fn inverse_inplace(&self, data: &mut Array1<Complex64>) -> QftResult<()> {
        let transformed = self.inverse(data)?;
        *data = transformed;
        Ok(())
    }
}

/// Convenience wrapper for forward QFT.
pub fn qft(input: &Array1<Complex64>) -> QftResult<Array1<Complex64>> {
    QftPlan::new(QuantumStateDimension::new(input.len())?).forward(input)
}

/// Convenience wrapper for inverse QFT.
pub fn iqft(input: &Array1<Complex64>) -> QftResult<Array1<Complex64>> {
    QftPlan::new(QuantumStateDimension::new(input.len())?).inverse(input)
}
