//! 1D FFT plan.

use super::{RealFftData, AXIS_SCRATCH, AXIS_SCRATCH_32, VOLUME_COMPLEX_BUF};
use crate::types::PrecisionProfile;
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
    pub fn new(n: usize) -> Self {
        Self::with_precision(n, PrecisionProfile::HIGH_ACCURACY_F64)
    }

    /// Create a new 1D plan with an explicit precision profile.
    #[must_use]
    pub fn with_precision(n: usize, precision: PrecisionProfile) -> Self {
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
        match self.precision {
            PrecisionProfile::LOW_PRECISION_F32 => {
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
            }
            _ => {
                let promoted = input.mapv(f64::from);
                self.forward_real_to_complex(&promoted)
                    .mapv(|value| Complex32::new(value.re as f32, value.im as f32))
            }
        }
    }

    /// Inverse transform of an `f32`-storage complex spectrum.
    #[must_use]
    pub(crate) fn inverse_f32(&self, input: &Array1<Complex32>) -> Array1<f32> {
        match self.precision {
            PrecisionProfile::LOW_PRECISION_F32 => {
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
            }
            _ => {
                let promoted = input.mapv(|value| Complex64::new(value.re as f64, value.im as f64));
                self.inverse_complex_to_real(&promoted)
                    .mapv(|value| value as f32)
            }
        }
    }

    /// Forward transform of a real signal stored as `f16`.
    #[must_use]
    pub(crate) fn forward_f16(&self, input: &Array1<f16>) -> Array1<Complex32> {
        match self.precision {
            PrecisionProfile::MIXED_PRECISION_F16_F32 => {
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
            }
            _ => {
                let promoted = input.mapv(|value| f64::from(value.to_f32()));
                self.forward_real_to_complex(&promoted)
                    .mapv(|value| Complex32::new(value.re as f32, value.im as f32))
            }
        }
    }

    /// Inverse transform of a complex spectrum to `f16` storage.
    #[must_use]
    pub(crate) fn inverse_f16(&self, input: &Array1<Complex32>) -> Array1<f16> {
        match self.precision {
            PrecisionProfile::MIXED_PRECISION_F16_F32 => {
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
            }
            _ => {
                let promoted = input.mapv(|value| Complex64::new(value.re as f64, value.im as f64));
                self.inverse_complex_to_real(&promoted)
                    .mapv(|value| f16::from_f32(value as f32))
            }
        }
    }
}
