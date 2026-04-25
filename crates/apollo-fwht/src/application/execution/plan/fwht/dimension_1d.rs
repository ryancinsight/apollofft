//! 1D Fast Walsh-Hadamard Transform plan.

use crate::application::execution::kernel::direct::wht_inplace;
use crate::domain::contracts::error::FwhtError;
use ndarray::Array1;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Reusable FWHT plan.
///
/// Stores the validated transform length. All methods validate input length
/// and return Err(FwhtError::LengthMismatch) instead of panicking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FwhtPlan {
    n: usize,
}

impl FwhtPlan {
    /// Create a validated FWHT plan.
    ///
    /// # Errors
    /// Returns Err(FwhtError::EmptyInput) if n == 0.
    /// Returns Err(FwhtError::NonPowerOfTwo) if n is not a power of two.
    pub fn new(n: usize) -> Result<Self, FwhtError> {
        if n == 0 {
            return Err(FwhtError::EmptyInput);
        }
        if !n.is_power_of_two() {
            return Err(FwhtError::NonPowerOfTwo);
        }
        Ok(Self { n })
    }

    /// Return the transform length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.n
    }

    /// Return true when the plan length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.n == 0
    }

    /// Forward WHT of a real-valued vector. O(N log N).
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when input.len() != self.n.
    pub fn forward(&self, input: &Array1<f64>) -> Result<Array1<f64>, FwhtError> {
        if input.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        let mut data = input.clone();
        self.forward_inplace(&mut data)?;
        Ok(data)
    }

    /// Forward WHT in-place. O(N log N) butterfly operations.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when data.len() != self.n.
    pub fn forward_inplace(&self, data: &mut Array1<f64>) -> Result<(), FwhtError> {
        if data.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        wht_inplace(data.as_slice_mut().expect("Array must be contiguous"));
        Ok(())
    }

    /// Inverse WHT of a real-valued spectrum. Applies WHT then divides by N.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when input.len() != self.n.
    pub fn inverse(&self, input: &Array1<f64>) -> Result<Array1<f64>, FwhtError> {
        if input.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        let mut data = input.clone();
        self.inverse_inplace(&mut data)?;
        Ok(data)
    }

    /// Inverse WHT in-place. Applies WHT then divides by N.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when data.len() != self.n.
    pub fn inverse_inplace(&self, data: &mut Array1<f64>) -> Result<(), FwhtError> {
        if data.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        wht_inplace(data.as_slice_mut().expect("Array must be contiguous"));
        let scale = 1.0 / self.n as f64;
        data.mapv_inplace(|value| value * scale);
        Ok(())
    }

    /// Forward WHT of a complex-valued vector.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when input.len() != self.n.
    pub fn forward_complex(
        &self,
        input: &Array1<Complex64>,
    ) -> Result<Array1<Complex64>, FwhtError> {
        if input.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        let mut data = input.clone();
        self.forward_complex_inplace(&mut data)?;
        Ok(data)
    }

    /// Forward complex WHT in-place.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when data.len() != self.n.
    pub fn forward_complex_inplace(&self, data: &mut Array1<Complex64>) -> Result<(), FwhtError> {
        if data.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        wht_inplace(data.as_slice_mut().expect("Array must be contiguous"));
        Ok(())
    }

    /// Inverse WHT of a complex-valued spectrum. Applies WHT then divides by N.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when input.len() != self.n.
    pub fn inverse_complex(
        &self,
        input: &Array1<Complex64>,
    ) -> Result<Array1<Complex64>, FwhtError> {
        if input.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        let mut data = input.clone();
        self.inverse_complex_inplace(&mut data)?;
        Ok(data)
    }

    /// Inverse complex WHT in-place.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when data.len() != self.n.
    pub fn inverse_complex_inplace(&self, data: &mut Array1<Complex64>) -> Result<(), FwhtError> {
        if data.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        wht_inplace(data.as_slice_mut().expect("Array must be contiguous"));
        let scale = 1.0 / self.n as f64;
        data.mapv_inplace(|value| value * scale);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    #[test]
    fn two_point_transform_matches_reference() {
        let plan = FwhtPlan::new(2).expect("valid plan");
        let input = Array1::from_vec(vec![1.0, 3.0]);
        let output = plan.forward(&input).expect("forward");
        assert_relative_eq!(output[0], 4.0, epsilon = 1.0e-12);
        assert_relative_eq!(output[1], -2.0, epsilon = 1.0e-12);
    }

    #[test]
    fn roundtrip_recovers_input() {
        let plan = FwhtPlan::new(8).expect("valid plan");
        let input = Array1::from_vec(vec![1.0, -2.0, 3.5, 0.25, -1.5, 2.0, 0.0, 4.0]);
        let fwd = plan.forward(&input).expect("forward");
        let recovered = plan.inverse(&fwd).expect("inverse");
        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn complex_roundtrip_recovers_input() {
        let plan = FwhtPlan::new(4).expect("valid plan");
        let input = Array1::from_vec(vec![
            Complex64::new(1.0, -1.0),
            Complex64::new(2.0, 0.5),
            Complex64::new(-0.75, 0.25),
            Complex64::new(0.125, -0.625),
        ]);
        let fwd = plan.forward_complex(&input).expect("forward_complex");
        let recovered = plan.inverse_complex(&fwd).expect("inverse_complex");
        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn rejects_invalid_lengths() {
        assert!(matches!(FwhtPlan::new(0), Err(FwhtError::EmptyInput)));
        assert!(matches!(FwhtPlan::new(3), Err(FwhtError::NonPowerOfTwo)));
    }

    #[test]
    fn length_mismatch_returns_error() {
        let plan = FwhtPlan::new(4).expect("valid plan");
        let wrong = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(matches!(
            plan.forward(&wrong),
            Err(FwhtError::LengthMismatch)
        ));
        assert!(matches!(
            plan.inverse(&wrong),
            Err(FwhtError::LengthMismatch)
        ));
        let wrong_c = Array1::from_vec(vec![Complex64::new(1.0, 0.0); 3]);
        assert!(matches!(
            plan.forward_complex(&wrong_c),
            Err(FwhtError::LengthMismatch)
        ));
        assert!(matches!(
            plan.inverse_complex(&wrong_c),
            Err(FwhtError::LengthMismatch)
        ));
    }

    #[test]
    fn single_element_is_identity() {
        let plan = FwhtPlan::new(1).expect("valid plan");
        let input = Array1::from_vec(vec![42.0f64]);
        let fwd = plan.forward(&input).expect("forward");
        assert_relative_eq!(fwd[0], 42.0, epsilon = 1.0e-12);
        let inv = plan.inverse(&fwd).expect("inverse");
        assert_relative_eq!(inv[0], 42.0, epsilon = 1.0e-12);
    }

    #[test]
    fn involution_property() {
        // WHT(WHT(x)) = N * x
        let plan = FwhtPlan::new(8).expect("valid plan");
        let input = Array1::from_vec(vec![1.0, -2.0, 3.5, 0.25, -1.5, 2.0, 0.0, 4.0]);
        let fwd1 = plan.forward(&input).expect("fwd1");
        let fwd2 = plan.forward(&fwd1).expect("fwd2");
        for (actual, expected) in fwd2.iter().zip(input.iter()) {
            assert_relative_eq!(*actual, *expected * 8.0, epsilon = 1.0e-10);
        }
    }

    proptest::proptest! {
        #[test]
        fn roundtrip_holds_for_random_power_of_two_lengths(
            power in 1usize..12,
            samples in prop::collection::vec(-10.0f64..10.0f64, 1usize..4096)
        ) {
            let n = 1usize << power;
            let input = Array1::from_vec(
                samples.into_iter().cycle().take(n).collect::<Vec<_>>()
            );
            let plan = FwhtPlan::new(n).expect("valid plan");
            let fwd = plan.forward(&input).expect("forward");
            let recovered = plan.inverse(&fwd).expect("inverse");
            for (actual, expected) in recovered.iter().zip(input.iter()) {
                prop_assert!((actual - expected).abs() < 1.0e-10);
            }
        }
    }
}
