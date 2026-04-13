//! 2D FFT plan.

use super::{
    RealFftData, AXIS_BUF, AXIS_BUF_32, AXIS_SCRATCH, AXIS_SCRATCH_32, VOLUME_COMPLEX_BUF,
};
use crate::types::PrecisionProfile;
use half::f16;
use ndarray::{Array2, ArrayBase, ArrayViewMut2, Axis, DataMut, Ix2, Zip};
use num_complex::Complex32;
use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Reusable 2D FFT plan.
///
/// The 2D DFT is separable, so this plan applies one-dimensional FFT passes
/// along each axis. Rows are transformed in-place and columns use a thread-local
/// contiguous scratch buffer to avoid repeated heap allocation.
pub struct FftPlan2D {
    nx: usize,
    ny: usize,
    precision: PrecisionProfile,
    fft_x: Arc<dyn Fft<f64>>,
    fft_y: Arc<dyn Fft<f64>>,
    ifft_x: Arc<dyn Fft<f64>>,
    ifft_y: Arc<dyn Fft<f64>>,
    fft_x_scratch_len: usize,
    fft_y_scratch_len: usize,
    ifft_x_scratch_len: usize,
    ifft_y_scratch_len: usize,
    fft_x_f32: Arc<dyn Fft<f32>>,
    fft_y_f32: Arc<dyn Fft<f32>>,
    ifft_x_f32: Arc<dyn Fft<f32>>,
    ifft_y_f32: Arc<dyn Fft<f32>>,
    fft_x_f32_scratch_len: usize,
    fft_y_f32_scratch_len: usize,
    ifft_x_f32_scratch_len: usize,
    ifft_y_f32_scratch_len: usize,
}

impl std::fmt::Debug for FftPlan2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan2D")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .finish()
    }
}

impl FftPlan2D {
    /// Create a new 2D plan.
    #[must_use]
    pub fn new(nx: usize, ny: usize) -> Self {
        Self::with_precision(nx, ny, PrecisionProfile::HIGH_ACCURACY_F64)
    }

    /// Create a new 2D plan with an explicit precision profile.
    #[must_use]
    pub fn with_precision(nx: usize, ny: usize, precision: PrecisionProfile) -> Self {
        let mut planner = FftPlanner::new();
        let mut planner_f32 = FftPlanner::<f32>::new();
        let fft_x = planner.plan_fft_forward(nx);
        let fft_y = planner.plan_fft_forward(ny);
        let ifft_x = planner.plan_fft_inverse(nx);
        let ifft_y = planner.plan_fft_inverse(ny);
        let fft_x_f32 = planner_f32.plan_fft_forward(nx);
        let fft_y_f32 = planner_f32.plan_fft_forward(ny);
        let ifft_x_f32 = planner_f32.plan_fft_inverse(nx);
        let ifft_y_f32 = planner_f32.plan_fft_inverse(ny);
        Self {
            nx,
            ny,
            precision,
            fft_x_scratch_len: fft_x.get_inplace_scratch_len(),
            fft_y_scratch_len: fft_y.get_inplace_scratch_len(),
            ifft_x_scratch_len: ifft_x.get_inplace_scratch_len(),
            ifft_y_scratch_len: ifft_y.get_inplace_scratch_len(),
            fft_x_f32_scratch_len: fft_x_f32.get_inplace_scratch_len(),
            fft_y_f32_scratch_len: fft_y_f32.get_inplace_scratch_len(),
            ifft_x_f32_scratch_len: ifft_x_f32.get_inplace_scratch_len(),
            ifft_y_f32_scratch_len: ifft_y_f32.get_inplace_scratch_len(),
            fft_x,
            fft_y,
            ifft_x,
            ifft_y,
            fft_x_f32,
            fft_y_f32,
            ifft_x_f32,
            ifft_y_f32,
        }
    }

    /// Return the precision profile used by this plan.
    #[must_use]
    pub fn precision_profile(&self) -> PrecisionProfile {
        self.precision
    }

    /// Forward transform of a real array.
    #[must_use]
    pub fn forward(&self, input: &Array2<f64>) -> Array2<Complex64> {
        self.forward_real_to_complex(input)
    }

    /// Inverse transform returning a real array.
    #[must_use]
    pub fn inverse(&self, input: &Array2<Complex64>) -> Array2<f64> {
        self.inverse_complex_to_real(input)
    }

    /// Forward transform of a real array using generic storage dispatch.
    #[must_use]
    pub fn forward_typed<T: RealFftData>(&self, input: &Array2<T>) -> Array2<T::Spectrum> {
        T::forward_2d(self, input)
    }

    /// Inverse transform of a complex spectrum using generic storage dispatch.
    #[must_use]
    pub fn inverse_typed<T: RealFftData>(&self, input: &Array2<T::Spectrum>) -> Array2<T> {
        T::inverse_2d(self, input)
    }

    /// Forward transform of a real array.
    #[must_use]
    pub fn forward_real_to_complex(&self, input: &Array2<f64>) -> Array2<Complex64> {
        let mut data = input.mapv(|value| Complex64::new(value, 0.0));
        self.forward_complex_inplace(&mut data);
        data
    }

    /// Forward transform of a real array into a caller-supplied complex buffer.
    pub fn forward_real_to_complex_into(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<Complex64>,
    ) {
        Zip::from(&mut *output).and(input).for_each(|out, &value| {
            *out = Complex64::new(value, 0.0);
        });
        self.forward_complex_inplace(output);
    }

    /// Compatibility alias for `forward_real_to_complex_into`.
    pub fn forward_into(&self, input: &Array2<f64>, output: &mut Array2<Complex64>) {
        self.forward_real_to_complex_into(input, output);
    }

    /// Inverse transform returning a real array.
    #[must_use]
    pub fn inverse_complex_to_real(&self, input: &Array2<Complex64>) -> Array2<f64> {
        let mut output = Array2::<f64>::zeros((self.nx, self.ny));
        self.inverse_complex_to_real_with_workspace(input, &mut output);
        output
    }

    /// Inverse transform into caller-owned real and scratch buffers.
    pub fn inverse_complex_to_real_into(
        &self,
        input: &Array2<Complex64>,
        output: &mut Array2<f64>,
        scratch: &mut Array2<Complex64>,
    ) {
        scratch.assign(input);
        self.inverse_complex_inplace(scratch);
        let norm = 1.0 / (self.nx * self.ny) as f64;
        Zip::from(output).and(scratch).for_each(|out, value| {
            *out = value.re * norm;
        });
    }

    fn inverse_complex_to_real_with_workspace(
        &self,
        input: &Array2<Complex64>,
        output: &mut Array2<f64>,
    ) {
        assert_eq!(
            input.dim(),
            (self.nx, self.ny),
            "inverse input shape mismatch"
        );
        assert_eq!(
            output.dim(),
            (self.nx, self.ny),
            "inverse output shape mismatch"
        );
        let total = self.nx * self.ny;
        let input_slice = input
            .as_slice_memory_order()
            .expect("Array must be contiguous");
        let output_slice = output
            .as_slice_memory_order_mut()
            .expect("Array must be contiguous");
        VOLUME_COMPLEX_BUF.with(|cell| {
            let mut workspace = cell.borrow_mut();
            if workspace.len() < total {
                workspace.resize(total, Complex64::default());
            }
            workspace[..total].copy_from_slice(input_slice);
            let mut view = ArrayViewMut2::from_shape((self.nx, self.ny), &mut workspace[..total])
                .expect("workspace shape must match plan dimensions");
            self.inverse_complex_inplace_impl(&mut view);
            let norm = 1.0 / total as f64;
            for (out, value) in output_slice.iter_mut().zip(workspace[..total].iter()) {
                *out = value.re * norm;
            }
        });
    }

    /// Compatibility alias for `inverse_complex_to_real_into`.
    pub fn inverse_into(
        &self,
        input: &Array2<Complex64>,
        output: &mut Array2<f64>,
        scratch: &mut Array2<Complex64>,
    ) {
        self.inverse_complex_to_real_into(input, output, scratch);
    }

    /// Forward transform of a complex array in-place.
    pub fn forward_complex_inplace(&self, data: &mut Array2<Complex64>) {
        let fft_x = Arc::clone(&self.fft_x);
        let fft_y = Arc::clone(&self.fft_y);
        data.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                let buffer = row.as_slice_mut().expect("row must be contiguous");
                AXIS_SCRATCH.with(|cell| {
                    let mut scratch = cell.borrow_mut();
                    let len = self.fft_x_scratch_len;
                    if scratch.len() < len {
                        scratch.resize(len, Complex64::default());
                    }
                    fft_x.process_with_scratch(buffer, &mut scratch[..len]);
                });
            });
        let ny = self.ny;
        data.axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut column| {
                AXIS_BUF.with(|cell| {
                    let mut buffer = cell.borrow_mut();
                    if buffer.len() < ny {
                        buffer.resize(ny, Complex64::default());
                    }
                    for (index, &value) in column.iter().enumerate() {
                        buffer[index] = value;
                    }
                    AXIS_SCRATCH.with(|scratch_cell| {
                        let mut scratch = scratch_cell.borrow_mut();
                        let len = self.fft_y_scratch_len;
                        if scratch.len() < len {
                            scratch.resize(len, Complex64::default());
                        }
                        fft_y.process_with_scratch(&mut buffer[..ny], &mut scratch[..len]);
                    });
                    for (index, value) in column.iter_mut().enumerate() {
                        *value = buffer[index];
                    }
                });
            });
    }

    /// Inverse transform of a complex array in-place without normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array2<Complex64>) {
        self.inverse_complex_inplace_impl(data);
    }

    fn inverse_complex_inplace_impl<S>(&self, data: &mut ArrayBase<S, Ix2>)
    where
        S: DataMut<Elem = Complex64>,
    {
        let ifft_x = Arc::clone(&self.ifft_x);
        let ifft_y = Arc::clone(&self.ifft_y);
        let ny = self.ny;
        data.axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut column| {
                AXIS_BUF.with(|cell| {
                    let mut buffer = cell.borrow_mut();
                    if buffer.len() < ny {
                        buffer.resize(ny, Complex64::default());
                    }
                    for (index, &value) in column.iter().enumerate() {
                        buffer[index] = value;
                    }
                    AXIS_SCRATCH.with(|scratch_cell| {
                        let mut scratch = scratch_cell.borrow_mut();
                        let len = self.ifft_y_scratch_len;
                        if scratch.len() < len {
                            scratch.resize(len, Complex64::default());
                        }
                        ifft_y.process_with_scratch(&mut buffer[..ny], &mut scratch[..len]);
                    });
                    for (index, value) in column.iter_mut().enumerate() {
                        *value = buffer[index];
                    }
                });
            });
        data.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                let buffer = row.as_slice_mut().expect("row must be contiguous");
                AXIS_SCRATCH.with(|cell| {
                    let mut scratch = cell.borrow_mut();
                    let len = self.ifft_x_scratch_len;
                    if scratch.len() < len {
                        scratch.resize(len, Complex64::default());
                    }
                    ifft_x.process_with_scratch(buffer, &mut scratch[..len]);
                });
            });
    }

    /// Forward transform of a real array stored as `f32`.
    #[must_use]
    pub(crate) fn forward_f32(&self, input: &Array2<f32>) -> Array2<Complex32> {
        match self.precision {
            PrecisionProfile::LOW_PRECISION_F32 => {
                let mut data = input.mapv(|value| Complex32::new(value, 0.0));
                self.forward_complex_inplace_f32(&mut data);
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
    pub(crate) fn inverse_f32(&self, input: &Array2<Complex32>) -> Array2<f32> {
        match self.precision {
            PrecisionProfile::LOW_PRECISION_F32 => {
                let mut data = input.clone();
                self.inverse_complex_inplace_f32(&mut data);
                let norm = 1.0 / (self.nx * self.ny) as f32;
                data.mapv(|value| value.re * norm)
            }
            _ => {
                let promoted = input.mapv(|value| Complex64::new(value.re as f64, value.im as f64));
                self.inverse_complex_to_real(&promoted)
                    .mapv(|value| value as f32)
            }
        }
    }

    /// Forward transform of a real array stored as `f16`.
    #[must_use]
    pub(crate) fn forward_f16(&self, input: &Array2<f16>) -> Array2<Complex32> {
        match self.precision {
            PrecisionProfile::MIXED_PRECISION_F16_F32 => {
                let mut data = input.mapv(|value| Complex32::new(value.to_f32(), 0.0));
                self.forward_complex_inplace_f32(&mut data);
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
    pub(crate) fn inverse_f16(&self, input: &Array2<Complex32>) -> Array2<f16> {
        match self.precision {
            PrecisionProfile::MIXED_PRECISION_F16_F32 => {
                let mut data = input.clone();
                self.inverse_complex_inplace_f32(&mut data);
                let norm = 1.0 / (self.nx * self.ny) as f32;
                data.mapv(|value| f16::from_f32(value.re * norm))
            }
            _ => {
                let promoted = input.mapv(|value| Complex64::new(value.re as f64, value.im as f64));
                self.inverse_complex_to_real(&promoted)
                    .mapv(|value| f16::from_f32(value as f32))
            }
        }
    }

    fn forward_complex_inplace_f32(&self, data: &mut Array2<Complex32>) {
        let fft_x = Arc::clone(&self.fft_x_f32);
        let fft_y = Arc::clone(&self.fft_y_f32);
        data.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                AXIS_SCRATCH_32.with(|cell| {
                    let mut scratch = cell.borrow_mut();
                    let len = self.fft_x_f32_scratch_len;
                    if scratch.len() < len {
                        scratch.resize(len, Complex32::default());
                    }
                    fft_x.process_with_scratch(
                        row.as_slice_mut().expect("row must be contiguous"),
                        &mut scratch[..len],
                    );
                });
            });
        let ny = self.ny;
        data.axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut column| {
                AXIS_BUF_32.with(|cell| {
                    let mut buffer = cell.borrow_mut();
                    if buffer.len() < ny {
                        buffer.resize(ny, Complex32::default());
                    }
                    for (index, &value) in column.iter().enumerate() {
                        buffer[index] = value;
                    }
                    AXIS_SCRATCH_32.with(|scratch_cell| {
                        let mut scratch = scratch_cell.borrow_mut();
                        let len = self.fft_y_f32_scratch_len;
                        if scratch.len() < len {
                            scratch.resize(len, Complex32::default());
                        }
                        fft_y.process_with_scratch(&mut buffer[..ny], &mut scratch[..len]);
                    });
                    for (index, value) in column.iter_mut().enumerate() {
                        *value = buffer[index];
                    }
                });
            });
    }

    fn inverse_complex_inplace_f32(&self, data: &mut Array2<Complex32>) {
        let ifft_x = Arc::clone(&self.ifft_x_f32);
        let ifft_y = Arc::clone(&self.ifft_y_f32);
        let ny = self.ny;
        data.axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut column| {
                AXIS_BUF_32.with(|cell| {
                    let mut buffer = cell.borrow_mut();
                    if buffer.len() < ny {
                        buffer.resize(ny, Complex32::default());
                    }
                    for (index, &value) in column.iter().enumerate() {
                        buffer[index] = value;
                    }
                    AXIS_SCRATCH_32.with(|scratch_cell| {
                        let mut scratch = scratch_cell.borrow_mut();
                        let len = self.ifft_y_f32_scratch_len;
                        if scratch.len() < len {
                            scratch.resize(len, Complex32::default());
                        }
                        ifft_y.process_with_scratch(&mut buffer[..ny], &mut scratch[..len]);
                    });
                    for (index, value) in column.iter_mut().enumerate() {
                        *value = buffer[index];
                    }
                });
            });
        data.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                AXIS_SCRATCH_32.with(|cell| {
                    let mut scratch = cell.borrow_mut();
                    let len = self.ifft_x_f32_scratch_len;
                    if scratch.len() < len {
                        scratch.resize(len, Complex32::default());
                    }
                    ifft_x.process_with_scratch(
                        row.as_slice_mut().expect("row must be contiguous"),
                        &mut scratch[..len],
                    );
                });
            });
    }
}
