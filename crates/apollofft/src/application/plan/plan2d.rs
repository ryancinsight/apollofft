//! 2D FFT plan.

use super::AXIS_BUF;
use ndarray::{Array2, Axis, Zip};
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
    fft_x: Arc<dyn Fft<f64>>,
    fft_y: Arc<dyn Fft<f64>>,
    ifft_x: Arc<dyn Fft<f64>>,
    ifft_y: Arc<dyn Fft<f64>>,
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
        let mut planner = FftPlanner::new();
        Self {
            nx,
            ny,
            fft_x: planner.plan_fft_forward(nx),
            fft_y: planner.plan_fft_forward(ny),
            ifft_x: planner.plan_fft_inverse(nx),
            ifft_y: planner.plan_fft_inverse(ny),
        }
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
        let mut data = input.clone();
        self.inverse_complex_inplace(&mut data);
        let norm = 1.0 / (self.nx * self.ny) as f64;
        data.mapv(|value| value.re * norm)
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
                fft_x.process(buffer);
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
                    fft_y.process(&mut buffer[..ny]);
                    for (index, value) in column.iter_mut().enumerate() {
                        *value = buffer[index];
                    }
                });
            });
    }

    /// Inverse transform of a complex array in-place without normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array2<Complex64>) {
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
                    ifft_y.process(&mut buffer[..ny]);
                    for (index, value) in column.iter_mut().enumerate() {
                        *value = buffer[index];
                    }
                });
            });
        data.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                let buffer = row.as_slice_mut().expect("row must be contiguous");
                ifft_x.process(buffer);
            });
    }
}
