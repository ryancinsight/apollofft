//! 1D FFT plan.

use super::{RealFftData, AXIS_SCRATCH, AXIS_SCRATCH_32, VOLUME_COMPLEX_BUF};
use crate::types::{PrecisionProfile, Shape1D};
use half::f16;
use ndarray::{Array1, Zip};
use num_complex::Complex32;
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Reusable 1D FFT plan.
///
/// The plan caches forward and inverse `rustfft` instances. The transform pair
/// is linear, the forward transform is unnormalized, and the inverse applies
/// `1 / n` when returning a real-valued signal.
///
/// # Theorem
///
/// Let `F` denote the forward discrete Fourier transform implemented by this
/// plan and let `F⁻¹` denote the inverse path that applies `1 / n`
/// normalization. For every real signal `x ∈ Rⁿ`, the plan satisfies
/// `F⁻¹(F(x)) = x`, `F(ax + by) = aF(x) + bF(y)`, and Parseval's identity
/// `Σ |x_j|² = (1 / n) Σ |X_k|²`.
///
/// **Proof sketch**
///
/// `rustfft` computes the standard unnormalized DFT and its inverse over the
/// same Vandermonde basis. The inverse path multiplies by `1 / n`, which
/// yields the exact inverse in exact arithmetic. Linearity follows from matrix
/// linearity of the DFT. Parseval follows because the DFT matrix is unitary up
/// to the factor `√n`; applying the FFTW-compatible normalization on inverse
/// preserves the `1 / n` energy relationship used in the tests below.
///
/// Assumptions:
/// - input and output arrays are contiguous standard-layout `ndarray` buffers
/// - transform length `n > 0`, enforced by [`Shape1D`]
/// - floating-point roundoff is bounded by the precision profile
///
/// Failure modes:
/// - mismatched caller-supplied buffer lengths panic with an explicit assertion
/// - non-contiguous buffers panic when `ndarray` cannot expose a contiguous slice
///
/// Enforced by:
/// - `roundtrip_holds_for_random_real_signals`
/// - `forward_is_linear`
/// - `parseval_identity_holds`
/// - `typed_precision_profiles_remain_bounded_for_quantized_inputs`
pub struct FftPlan1D {
    n: usize,
    precision: PrecisionProfile,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
    fft_scratch_len: usize,
    ifft_scratch_len: usize,
    fft_f32: Arc<dyn Fft<f32>>,
    ifft_f32: Arc<dyn Fft<f32>>,
    fft_f32_scratch_len: usize,
    ifft_f32_scratch_len: usize,
}

impl std::fmt::Debug for FftPlan1D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan1D").field("n", &self.n).finish()
    }
}

impl FftPlan1D {
    /// Create a new 1D plan.
    #[must_use]
    pub fn new(shape: Shape1D) -> Self {
        Self::with_precision(shape, PrecisionProfile::HIGH_ACCURACY_F64)
    }

    /// Create a new 1D plan with an explicit precision profile.
    #[must_use]
    pub fn with_precision(shape: Shape1D, precision: PrecisionProfile) -> Self {
        let n = shape.n;
        let mut planner = FftPlanner::new();
        let mut planner_f32 = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);
        let fft_f32 = planner_f32.plan_fft_forward(n);
        let ifft_f32 = planner_f32.plan_fft_inverse(n);
        Self {
            n,
            precision,
            fft_scratch_len: fft.get_inplace_scratch_len(),
            ifft_scratch_len: ifft.get_inplace_scratch_len(),
            fft_f32_scratch_len: fft_f32.get_inplace_scratch_len(),
            ifft_f32_scratch_len: ifft_f32.get_inplace_scratch_len(),
            fft,
            ifft,
            fft_f32,
            ifft_f32,
        }
    }

    /// Return the plan length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Return the validated shape owned by this plan.
    #[must_use]
    pub fn shape(&self) -> Shape1D {
        Shape1D { n: self.n }
    }

    /// Return the precision profile used by this plan.
    #[must_use]
    pub fn precision_profile(&self) -> PrecisionProfile {
        self.precision
    }

    /// Whether the plan length is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Forward transform of a real signal.
    #[must_use]
    pub fn forward(&self, input: &Array1<f64>) -> Array1<Complex64> {
        self.forward_real_to_complex(input)
    }

    /// Inverse transform returning a real signal.
    #[must_use]
    pub fn inverse(&self, input: &Array1<Complex64>) -> Array1<f64> {
        self.inverse_complex_to_real(input)
    }

    /// Forward transform of a real signal using generic storage dispatch.
    #[must_use]
    pub fn forward_typed<T: RealFftData>(&self, input: &Array1<T>) -> Array1<T::Spectrum> {
        T::forward_1d(self, input)
    }

    /// Inverse transform of a complex spectrum using generic storage dispatch.
    #[must_use]
    pub fn inverse_typed<T: RealFftData>(&self, input: &Array1<T::Spectrum>) -> Array1<T> {
        T::inverse_1d(self, input)
    }

    /// Forward transform of a real signal.
    #[must_use]
    pub fn forward_real_to_complex(&self, input: &Array1<f64>) -> Array1<Complex64> {
        let mut data = input.mapv(|value| Complex64::new(value, 0.0));
        AXIS_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let len = self.fft_scratch_len;
            if scratch.len() < len {
                scratch.resize(len, Complex64::default());
            }
            self.fft.process_with_scratch(
                data.as_slice_mut().expect("Array must be contiguous"),
                &mut scratch[..len],
            );
        });
        data
    }

    /// Forward transform of a real signal into a caller-supplied complex buffer.
    pub fn forward_real_to_complex_into(
        &self,
        input: &Array1<f64>,
        output: &mut Array1<Complex64>,
    ) {
        assert_eq!(input.len(), self.n, "forward input length mismatch");
        assert_eq!(output.len(), self.n, "forward output length mismatch");
        Zip::from(&mut *output).and(input).for_each(|out, &value| {
            *out = Complex64::new(value, 0.0);
        });
        AXIS_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let len = self.fft_scratch_len;
            if scratch.len() < len {
                scratch.resize(len, Complex64::default());
            }
            self.fft.process_with_scratch(
                output.as_slice_mut().expect("Array must be contiguous"),
                &mut scratch[..len],
            );
        });
    }

    /// Compatibility alias for `forward_real_to_complex_into`.
    pub fn forward_into(&self, input: &Array1<f64>, output: &mut Array1<Complex64>) {
        self.forward_real_to_complex_into(input, output);
    }

    /// Inverse transform returning a real-valued signal.
    #[must_use]
    pub fn inverse_complex_to_real(&self, input: &Array1<Complex64>) -> Array1<f64> {
        let mut output = Array1::<f64>::zeros(self.n);
        self.inverse_complex_to_real_with_workspace(input, &mut output);
        output
    }

    /// Inverse transform into caller-owned real and scratch buffers.
    pub fn inverse_complex_to_real_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<f64>,
        scratch: &mut Array1<Complex64>,
    ) {
        assert_eq!(input.len(), self.n, "inverse input length mismatch");
        assert_eq!(output.len(), self.n, "inverse output length mismatch");
        assert_eq!(scratch.len(), self.n, "inverse scratch length mismatch");
        scratch.assign(input);
        AXIS_SCRATCH.with(|cell| {
            let mut fft_scratch = cell.borrow_mut();
            let len = self.ifft_scratch_len;
            if fft_scratch.len() < len {
                fft_scratch.resize(len, Complex64::default());
            }
            self.ifft.process_with_scratch(
                scratch.as_slice_mut().expect("Array must be contiguous"),
                &mut fft_scratch[..len],
            );
        });
        let norm = 1.0 / self.n as f64;
        Zip::from(output).and(scratch).for_each(|out, value| {
            *out = value.re * norm;
        });
    }

    fn inverse_complex_to_real_with_workspace(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<f64>,
    ) {
        assert_eq!(input.len(), self.n, "inverse input length mismatch");
        assert_eq!(output.len(), self.n, "inverse output length mismatch");
        let input_slice = input.as_slice().expect("Array must be contiguous");
        let output_slice = output.as_slice_mut().expect("Array must be contiguous");
        VOLUME_COMPLEX_BUF.with(|cell| {
            let mut workspace = cell.borrow_mut();
            if workspace.len() < self.n {
                workspace.resize(self.n, Complex64::default());
            }
            workspace[..self.n].copy_from_slice(input_slice);
            AXIS_SCRATCH.with(|scratch_cell| {
                let mut scratch = scratch_cell.borrow_mut();
                let len = self.ifft_scratch_len;
                if scratch.len() < len {
                    scratch.resize(len, Complex64::default());
                }
                self.ifft
                    .process_with_scratch(&mut workspace[..self.n], &mut scratch[..len]);
            });
            let norm = 1.0 / self.n as f64;
            for (out, value) in output_slice.iter_mut().zip(workspace[..self.n].iter()) {
                *out = value.re * norm;
            }
        });
    }

    /// Compatibility alias for `inverse_complex_to_real_into`.
    pub fn inverse_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<f64>,
        scratch: &mut Array1<Complex64>,
    ) {
        self.inverse_complex_to_real_into(input, output, scratch);
    }

    /// Forward transform of a complex signal in-place.
    pub fn forward_complex_inplace(&self, data: &mut Array1<Complex64>) {
        assert_eq!(data.len(), self.n, "complex forward length mismatch");
        AXIS_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let len = self.fft_scratch_len;
            if scratch.len() < len {
                scratch.resize(len, Complex64::default());
            }
            self.fft.process_with_scratch(
                data.as_slice_mut().expect("Array must be contiguous"),
                &mut scratch[..len],
            );
        });
    }

    /// Inverse transform of a complex signal in-place without normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array1<Complex64>) {
        assert_eq!(data.len(), self.n, "complex inverse length mismatch");
        AXIS_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let len = self.ifft_scratch_len;
            if scratch.len() < len {
                scratch.resize(len, Complex64::default());
            }
            self.ifft.process_with_scratch(
                data.as_slice_mut().expect("Array must be contiguous"),
                &mut scratch[..len],
            );
        });
    }

    /// Forward transform of a real signal stored as `f32`.
    #[must_use]
    pub(crate) fn forward_f32(&self, input: &Array1<f32>) -> Array1<Complex32> {
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            let mut data = input.mapv(|value| Complex32::new(value, 0.0));
            AXIS_SCRATCH_32.with(|cell| {
                let mut scratch = cell.borrow_mut();
                let len = self.fft_f32_scratch_len;
                if scratch.len() < len {
                    scratch.resize(len, Complex32::default());
                }
                self.fft_f32.process_with_scratch(
                    data.as_slice_mut().expect("Array must be contiguous"),
                    &mut scratch[..len],
                );
            });
            data
        } else {
            let promoted = input.mapv(f64::from);
            self.forward_real_to_complex(&promoted)
                .mapv(|value| Complex32::new(value.re as f32, value.im as f32))
        }
    }

    /// Inverse transform of an `f32`-storage complex spectrum.
    #[must_use]
    pub(crate) fn inverse_f32(&self, input: &Array1<Complex32>) -> Array1<f32> {
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            let mut data = input.clone();
            AXIS_SCRATCH_32.with(|cell| {
                let mut scratch = cell.borrow_mut();
                let len = self.ifft_f32_scratch_len;
                if scratch.len() < len {
                    scratch.resize(len, Complex32::default());
                }
                self.ifft_f32.process_with_scratch(
                    data.as_slice_mut().expect("Array must be contiguous"),
                    &mut scratch[..len],
                );
            });
            let norm = 1.0 / self.n as f32;
            data.mapv(|value| value.re * norm)
        } else {
            let promoted =
                input.mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im)));
            self.inverse_complex_to_real(&promoted)
                .mapv(|value| value as f32)
        }
    }

    /// Forward transform of a real signal stored as `f16`.
    #[must_use]
    pub(crate) fn forward_f16(&self, input: &Array1<f16>) -> Array1<Complex32> {
        if self.precision == PrecisionProfile::MIXED_PRECISION_F16_F32 {
            let mut data = input.mapv(|value| Complex32::new(value.to_f32(), 0.0));
            AXIS_SCRATCH_32.with(|cell| {
                let mut scratch = cell.borrow_mut();
                let len = self.fft_f32_scratch_len;
                if scratch.len() < len {
                    scratch.resize(len, Complex32::default());
                }
                self.fft_f32.process_with_scratch(
                    data.as_slice_mut().expect("Array must be contiguous"),
                    &mut scratch[..len],
                );
            });
            data
        } else {
            let promoted = input.mapv(|value| f64::from(value.to_f32()));
            self.forward_real_to_complex(&promoted)
                .mapv(|value| Complex32::new(value.re as f32, value.im as f32))
        }
    }

    /// Inverse transform of a complex spectrum to `f16` storage.
    #[must_use]
    pub(crate) fn inverse_f16(&self, input: &Array1<Complex32>) -> Array1<f16> {
        if self.precision == PrecisionProfile::MIXED_PRECISION_F16_F32 {
            let mut data = input.clone();
            AXIS_SCRATCH_32.with(|cell| {
                let mut scratch = cell.borrow_mut();
                let len = self.ifft_f32_scratch_len;
                if scratch.len() < len {
                    scratch.resize(len, Complex32::default());
                }
                self.ifft_f32.process_with_scratch(
                    data.as_slice_mut().expect("Array must be contiguous"),
                    &mut scratch[..len],
                );
            });
            let norm = 1.0 / self.n as f32;
            data.mapv(|value| f16::from_f32(value.re * norm))
        } else {
            let promoted =
                input.mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im)));
            self.inverse_complex_to_real(&promoted)
                .mapv(|value| f16::from_f32(value as f32))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    fn energy_time_domain(signal: &Array1<f64>) -> f64 {
        signal.iter().map(|value| value * value).sum()
    }

    fn energy_frequency_domain(spectrum: &Array1<Complex64>) -> f64 {
        spectrum.iter().map(|value| value.norm_sqr()).sum()
    }

    #[test]
    fn forward_is_linear() {
        let shape = Shape1D::new(16).expect("shape");
        let plan = FftPlan1D::new(shape);
        let lhs = Array1::from_iter((0..16).map(|i| (i as f64 * 0.2).sin()));
        let rhs = Array1::from_iter((0..16).map(|i| (i as f64 * 0.3).cos()));
        let alpha = 1.75;
        let beta = -0.25;
        let combined = &lhs * alpha + &rhs * beta;

        let combined_hat = plan.forward_real_to_complex(&combined);
        let lhs_hat = plan.forward_real_to_complex(&lhs);
        let rhs_hat = plan.forward_real_to_complex(&rhs);

        for (actual, expected) in combined_hat.iter().zip(
            lhs_hat
                .iter()
                .zip(rhs_hat.iter())
                .map(|(x, y)| *x * alpha + *y * beta),
        ) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn parseval_identity_holds() {
        let shape = Shape1D::new(32).expect("shape");
        let plan = FftPlan1D::new(shape);
        let signal = Array1::from_iter((0..32).map(|i| {
            let x = i as f64;
            (0.17 * x).sin() + 0.5 * (0.41 * x).cos()
        }));

        let spectrum = plan.forward_real_to_complex(&signal);
        let time_energy = energy_time_domain(&signal);
        let spectral_energy = energy_frequency_domain(&spectrum) / shape.n as f64;

        assert_relative_eq!(time_energy, spectral_energy, epsilon = 1.0e-10);
    }

    #[test]
    fn caller_owned_paths_match_allocating_paths() {
        let shape = Shape1D::new(24).expect("shape");
        let plan = FftPlan1D::new(shape);
        let signal = Array1::from_iter((0..24).map(|i| (i as f64 * 0.1).sin()));

        let expected_hat = plan.forward_real_to_complex(&signal);
        let mut actual_hat = Array1::<Complex64>::zeros(shape.n);
        plan.forward_real_to_complex_into(&signal, &mut actual_hat);
        for (expected, actual) in expected_hat.iter().zip(actual_hat.iter()) {
            assert_relative_eq!(expected.re, actual.re, epsilon = 1.0e-12);
            assert_relative_eq!(expected.im, actual.im, epsilon = 1.0e-12);
        }

        let expected_signal = plan.inverse_complex_to_real(&expected_hat);
        let mut actual_signal = Array1::<f64>::zeros(shape.n);
        let mut scratch = Array1::<Complex64>::zeros(shape.n);
        plan.inverse_complex_to_real_into(&expected_hat, &mut actual_signal, &mut scratch);
        for (expected, actual) in expected_signal.iter().zip(actual_signal.iter()) {
            assert_relative_eq!(expected, actual, epsilon = 1.0e-12);
        }
    }

    #[test]
    #[should_panic(expected = "forward output length mismatch")]
    fn forward_rejects_mismatched_output_length() {
        let shape = Shape1D::new(8).expect("shape");
        let plan = FftPlan1D::new(shape);
        let signal = Array1::from_elem(shape.n, 1.0);
        let mut output = Array1::<Complex64>::zeros(shape.n - 1);
        plan.forward_real_to_complex_into(&signal, &mut output);
    }

    #[test]
    #[should_panic(expected = "inverse scratch length mismatch")]
    fn inverse_rejects_mismatched_scratch_length() {
        let shape = Shape1D::new(8).expect("shape");
        let plan = FftPlan1D::new(shape);
        let spectrum = Array1::<Complex64>::zeros(shape.n);
        let mut output = Array1::<f64>::zeros(shape.n);
        let mut scratch = Array1::<Complex64>::zeros(shape.n - 1);
        plan.inverse_complex_to_real_into(&spectrum, &mut output, &mut scratch);
    }

    #[test]
    fn typed_precision_profiles_remain_bounded_for_quantized_inputs() {
        let shape = Shape1D::new(32).expect("shape");
        let reference = Array1::from_iter((0..32).map(|i| {
            let x = i as f64;
            (0.09 * x).sin() + 0.25 * (0.71 * x).cos()
        }));

        let high_plan = FftPlan1D::with_precision(shape, PrecisionProfile::HIGH_ACCURACY_F64);
        let low_plan = FftPlan1D::with_precision(shape, PrecisionProfile::LOW_PRECISION_F32);
        let mixed_plan =
            FftPlan1D::with_precision(shape, PrecisionProfile::MIXED_PRECISION_F16_F32);

        let mixed_input = reference.mapv(|value| f16::from_f32(value as f32));
        let quantized_reference = mixed_input.mapv(|value| f64::from(value.to_f32()));
        let low_input = mixed_input.mapv(|value| value.to_f32());

        let low_recovered: Array1<f32> =
            low_plan.inverse_typed(&low_plan.forward_typed(&low_input));
        let low_roundtrip = low_recovered.mapv(f64::from);
        let mixed_recovered: Array1<f16> =
            mixed_plan.inverse_typed(&mixed_plan.forward_typed(&mixed_input));
        let mixed_roundtrip: Array1<f64> = mixed_recovered.mapv(|value| f64::from(value.to_f32()));
        let high_roundtrip = high_plan.inverse(&high_plan.forward(&reference));

        let low_error: f64 = low_roundtrip
            .iter()
            .zip(quantized_reference.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .sum();
        let mixed_error: f64 = mixed_roundtrip
            .iter()
            .zip(quantized_reference.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .sum();
        let high_error: f64 = high_roundtrip
            .iter()
            .zip(reference.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .sum();

        assert!(high_error <= 1.0e-12);
        assert!(low_error <= 1.0e-4);
        assert!(mixed_error <= low_error + 1.0e-6);
    }

    proptest! {
        #[test]
        fn roundtrip_holds_for_random_real_signals(
            signal in (1usize..33).prop_flat_map(|len| {
                prop::collection::vec(-10.0f64..10.0f64, len)
            })
        ) {
            let shape = Shape1D::new(signal.len()).expect("shape");
            let plan = FftPlan1D::new(shape);
            let input = Array1::from_vec(signal);
            let recovered = plan.inverse_complex_to_real(&plan.forward_real_to_complex(&input));
            for (expected, actual) in input.iter().zip(recovered.iter()) {
                prop_assert!((expected - actual).abs() < 1.0e-9);
            }
        }
    }
}
