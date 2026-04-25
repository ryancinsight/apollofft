//! 1D Fractional Fourier Transform Plan

use crate::application::execution::kernel::direct::direct_frft_forward_into;
use crate::domain::contracts::error::FrftError;
use ndarray::Array1;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use std::f64::consts::FRAC_PI_2;

/// Direct fractional Fourier transform plan.
///
/// ## Integer order degenerate cases
///
/// When `alpha = order*pi/2` is a multiple of `pi`, the cotangent and cosecant
/// terms are singular. These fields are unused when the transform
/// degenerates to identity (order divisible by 4), reversal (order = 2 mod 4), or
/// centered DFT/IDFT (order = 1 or 3 mod 4). The direct kernel
/// handles those cases before reading cotangent or cosecant state.
///
/// The order `a` maps to rotation angle `alpha = a*pi/2`. Integer orders are
/// handled by exact identity, corrected centered DFT, reversal, and centered inverse DFT cases.
/// Non-integer orders evaluate the standard cotangent/cosecant FrFT kernel on centered
/// discrete coordinates.
///
/// # Theorem
///
/// The fractional Fourier transform is the order-`a` rotation of a signal in
/// the time-frequency plane by `alpha = a*pi/2`. Integer orders reduce to:
/// identity for `a = 0 mod 4`, centered unitary DFT for `a = 1 mod 4`,
/// reversal for `a = 2 mod 4`, and centered unitary inverse DFT for
/// `a = 3 mod 4`.
///
/// # Proof sketch
///
/// The continuous FrFT kernel factors into chirp terms containing `cot(alpha)`
/// and `csc(alpha)`. At integer quarter rotations, the limiting operators are
/// exactly the identity, Fourier transform, parity reversal, and inverse
/// Fourier transform. The implementation dispatches those singular limits
/// explicitly and uses finite cotangent/cosecant state only for non-integer
/// orders.
///
/// # Complexity
///
/// The direct kernel costs `O(n^2)` time. `forward_into` and `inverse_into`
/// write into caller-owned buffers and use `O(1)` auxiliary storage beyond the
/// output.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrftPlan {
    n: usize,
    order: f64,
    cot: f64,
    csc: f64,
    scale: Complex64,
}

impl FrftPlan {
    /// Create a validated FrFT plan.
    pub fn new(n: usize, order: f64) -> Result<Self, FrftError> {
        if n == 0 {
            return Err(FrftError::EmptySignal);
        }
        if !order.is_finite() {
            return Err(FrftError::NonFiniteOrder);
        }
        let reduced = order.rem_euclid(4.0);
        let alpha = reduced * FRAC_PI_2;
        let integer_rotation = (reduced - reduced.round()).abs() < 1.0e-12;
        let (cot, csc, scale) = if integer_rotation {
            (0.0, 0.0, Complex64::new(1.0, 0.0))
        } else {
            let sin_alpha = alpha.sin();
            let cot = alpha.cos() / sin_alpha;
            let csc = 1.0 / sin_alpha;
            (
                cot,
                csc,
                (1.0 - Complex64::i() * cot).sqrt() / (n as f64).sqrt(),
            )
        };

        Ok(Self {
            n,
            order,
            cot,
            csc,
            scale,
        })
    }

    /// Return the transform length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.n
    }

    /// Return whether the plan length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.n == 0
    }

    /// Return the fractional order.
    #[must_use]
    pub const fn order(self) -> f64 {
        self.order
    }

    /// Execute the forward FrFT.
    pub fn forward(&self, input: &Array1<Complex64>) -> Result<Array1<Complex64>, FrftError> {
        let mut output = Array1::<Complex64>::zeros(self.n);
        self.forward_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute the forward FrFT into a pre-allocated output buffer.
    pub fn forward_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<Complex64>,
    ) -> Result<(), FrftError> {
        if input.len() != self.n {
            return Err(FrftError::LengthMismatch {
                input: input.len(),
                plan: self.n,
            });
        }
        if output.len() != self.n {
            return Err(FrftError::LengthMismatch {
                input: output.len(),
                plan: self.n,
            });
        }
        direct_frft_forward_into(
            input.as_slice().expect("Array must be contiguous"),
            output.as_slice_mut().expect("Array must be contiguous"),
            self.order,
            self.cot,
            self.csc,
            self.scale,
        );
        Ok(())
    }

    /// Execute the inverse FrFT, equivalent to a forward FrFT of order `-a`.
    pub fn inverse(&self, input: &Array1<Complex64>) -> Result<Array1<Complex64>, FrftError> {
        let mut output = Array1::<Complex64>::zeros(self.n);
        self.inverse_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute the inverse FrFT into a pre-allocated output buffer.
    pub fn inverse_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<Complex64>,
    ) -> Result<(), FrftError> {
        let inverse_plan = Self::new(self.n, -self.order)?;
        inverse_plan.forward_into(input, output)
    }
}

/// Execute a single forward fractional Fourier transform on a 1D array.
///
/// This provides a zero-setup convenience path executing the explicit mathematical
/// definition directly without retaining a plan struct.
pub fn frft(input: &Array1<Complex64>, order: f64) -> Result<Array1<Complex64>, FrftError> {
    FrftPlan::new(input.len(), order)?.forward(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_order_zero_is_identity() {
        let input = Array1::from_vec(vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)]);
        assert_eq!(frft(&input, 0.0).expect("frft"), input);
    }

    #[test]
    fn exact_centered_continuity_at_boundary() {
        let n: usize = 16;
        let input = Array1::from_shape_fn(n, |i| Complex64::new((i as f64 * 0.1).sin(), 0.0));
        let boundary = frft(&input, 1.0).unwrap();
        let near_limit = frft(&input, 0.9999999999).unwrap();

        for (a, b) in boundary.iter().zip(near_limit.iter()) {
            assert!((a.re - b.re).abs() < 1.0e-6);
            assert!((a.im - b.im).abs() < 1.0e-6);
        }
    }

    #[test]
    fn integer_order_one_inverse_recovers_input() {
        let n: usize = 8;
        let plan = FrftPlan::new(n, 1.0).expect("valid plan");
        let input = Array1::from_shape_fn(n, |i| {
            Complex64::new((i as f64 * 0.17).cos(), (i as f64 * 0.23).sin())
        });
        let recovered = plan
            .inverse(&plan.forward(&input).expect("forward"))
            .expect("inverse");

        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert!((actual.re - expected.re).abs() < 1.0e-12);
            assert!((actual.im - expected.im).abs() < 1.0e-12);
        }
    }

    #[test]
    fn inverse_into_matches_allocating_inverse() {
        let n: usize = 8;
        let plan = FrftPlan::new(n, 3.0).expect("valid plan");
        let input = Array1::from_shape_fn(n, |i| Complex64::new(i as f64 - 2.0, i as f64 * 0.5));
        let expected = plan.inverse(&input).expect("inverse");
        let mut actual = Array1::<Complex64>::zeros(n);
        plan.inverse_into(&input, &mut actual)
            .expect("inverse_into");

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual.re - expected.re).abs() < 1.0e-12);
            assert!((actual.im - expected.im).abs() < 1.0e-12);
        }
    }

    #[test]
    fn rejects_invalid_plan() {
        assert_eq!(FrftPlan::new(0, 1.0), Err(FrftError::EmptySignal));
        assert_eq!(FrftPlan::new(4, f64::NAN), Err(FrftError::NonFiniteOrder));
    }
    #[test]
    fn frft_order_1_matches_dft() {
        use std::f64::consts::PI;
        // FrFT order=1 implements a centered unitary DFT:
        //   X[k] = (1/sqrt(N)) sum_j x[j] exp(-2*pi*i*(j-c)*(k-c)/N), c=(N-1)/2
        let n = 8usize;
        let input: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.31).sin(), 0.0))
            .collect();
        let input_arr = Array1::from_vec(input.clone());
        let result = frft(&input_arr, 1.0).unwrap();
        let center = (n as f64 - 1.0) * 0.5;
        let scale = 1.0 / (n as f64).sqrt();
        for k in 0..n {
            let u = k as f64 - center;
            let expected: Complex64 = input
                .iter()
                .enumerate()
                .map(|(j, &val)| {
                    let x = j as f64 - center;
                    let angle = -2.0 * PI * x * u / n as f64;
                    val * Complex64::new(angle.cos(), angle.sin())
                })
                .sum::<Complex64>()
                * scale;
            assert!(
                (result[k] - expected).norm() < 1e-12,
                "FrFT(1) != centered DFT reference at k={}",
                k
            );
        }
    }

    #[test]
    fn frft_order_4_is_identity() {
        // 4.0.rem_euclid(4.0) = 0 => identity path in direct_frft_forward_into.
        let n = 8usize;
        let input: Array1<Complex64> = Array1::from_shape_fn(n, |i| {
            Complex64::new((i as f64 * 0.5).cos(), (i as f64 * 0.3).sin())
        });
        let plan = FrftPlan::new(n, 4.0).unwrap();
        let result = plan.forward(&input).unwrap();
        for (orig, out) in input.iter().zip(result.iter()) {
            assert!(
                (orig - out).norm() < 1e-12,
                "FrFT(4) is not identity: diff = {}",
                (orig - out).norm()
            );
        }
    }

    #[test]
    fn frft_order_2_is_reversal() {
        // 2.0.rem_euclid(4.0) = 2 => reversal path: output[k] = input[N-1-k].
        let n = 8usize;
        let input: Array1<Complex64> =
            Array1::from_shape_fn(n, |i| Complex64::new(i as f64 + 1.0, 0.0));
        let plan = FrftPlan::new(n, 2.0).unwrap();
        let result = plan.forward(&input).unwrap();
        for k in 0..n {
            let expected = input[n - 1 - k];
            assert!(
                (result[k] - expected).norm() < 1e-12,
                "FrFT(2) reversal failed at k={}: got {:?}, expected {:?}",
                k,
                result[k],
                expected
            );
        }
    }
}
