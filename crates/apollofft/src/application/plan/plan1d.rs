//! 1D FFT plan.

use ndarray::{Array1, Zip};
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Reusable 1D FFT plan.
///
/// The plan caches forward and inverse `rustfft` instances. The transform pair
/// is linear, the forward transform is unnormalized, and the inverse applies
/// `1 / n` when returning a real-valued signal.
pub struct FftPlan1D {
    n: usize,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
}

impl std::fmt::Debug for FftPlan1D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan1D").field("n", &self.n).finish()
    }
}

impl FftPlan1D {
    /// Create a new 1D plan.
    #[must_use]
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::new();
        Self {
            n,
            fft: planner.plan_fft_forward(n),
            ifft: planner.plan_fft_inverse(n),
        }
    }

    /// Return the plan length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
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

    /// Forward transform of a real signal.
    #[must_use]
    pub fn forward_real_to_complex(&self, input: &Array1<f64>) -> Array1<Complex64> {
        let mut data = input.mapv(|value| Complex64::new(value, 0.0));
        self.fft
            .process(data.as_slice_mut().expect("Array must be contiguous"));
        data
    }

    /// Forward transform of a real signal into a caller-supplied complex buffer.
    pub fn forward_real_to_complex_into(
        &self,
        input: &Array1<f64>,
        output: &mut Array1<Complex64>,
    ) {
        Zip::from(&mut *output).and(input).for_each(|out, &value| {
            *out = Complex64::new(value, 0.0);
        });
        self.fft
            .process(output.as_slice_mut().expect("Array must be contiguous"));
    }

    /// Compatibility alias for `forward_real_to_complex_into`.
    pub fn forward_into(&self, input: &Array1<f64>, output: &mut Array1<Complex64>) {
        self.forward_real_to_complex_into(input, output);
    }

    /// Inverse transform returning a real-valued signal.
    #[must_use]
    pub fn inverse_complex_to_real(&self, input: &Array1<Complex64>) -> Array1<f64> {
        let mut data = input.clone();
        self.ifft
            .process(data.as_slice_mut().expect("Array must be contiguous"));
        let norm = 1.0 / self.n as f64;
        data.mapv(|value| value.re * norm)
    }

    /// Inverse transform into caller-owned real and scratch buffers.
    pub fn inverse_complex_to_real_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<f64>,
        scratch: &mut Array1<Complex64>,
    ) {
        scratch.assign(input);
        self.ifft
            .process(scratch.as_slice_mut().expect("Array must be contiguous"));
        let norm = 1.0 / self.n as f64;
        Zip::from(output).and(scratch).for_each(|out, value| {
            *out = value.re * norm;
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
        self.fft
            .process(data.as_slice_mut().expect("Array must be contiguous"));
    }

    /// Inverse transform of a complex signal in-place without normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array1<Complex64>) {
        self.ifft
            .process(data.as_slice_mut().expect("Array must be contiguous"));
    }
}
