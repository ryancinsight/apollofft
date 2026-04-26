//! 2D FFT plan.
//!
//! Apollo-owned 2D FFT implementation.
//!
//! The 2D DFT is separable, so this plan applies the in-repo auto-selected 1D
//! FFT kernel across rows and columns. The inverse path is normalized on each
//! inverse axis pass, which gives the standard `1 / (nx * ny)` inverse
//! normalization.
//!
//! # Mathematical contract
//!
//! For a complex input field `x in C^(nx x ny)`, the forward transform is
//!
//! `X[k,l] = sum_i sum_j x[i,j] exp(-2*pi*i*(k*i/nx + l*j/ny))`.
//!
//! The inverse transform is
//!
//! `x[i,j] = (1/(nx*ny)) sum_k sum_l X[k,l] exp(2*pi*i*(k*i/nx + l*j/ny))`.
//!
//! The implementation is linear and separable. Floating-point error follows
//! from the selected scalar precision and the selected 1D FFT kernel.
//!
//! # Complexity
//!
//! Let `C(n)` be the selected 1D FFT cost. The plan costs
//! `O(ny * C(nx) + nx * C(ny))`, with `C(n) = O(n log n)` for both radix-2 and
//! Bluestein plan paths. Contiguous innermost-axis passes mutate row chunks in
//! place, while non-contiguous passes gather lanes into scratch buffers before
//! scattering them back.

use crate::application::execution::kernel::{
    fft_forward_32, fft_forward_64, fft_inverse_32, fft_inverse_64,
};
use crate::application::execution::plan::fft::real_storage::RealFftData;
use crate::domain::metadata::precision::PrecisionProfile;
use crate::domain::metadata::shape::Shape2D;
use half::f16;
use ndarray::{Array2, Axis, Zip};
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;

/// Reusable 2D FFT plan.
pub struct FftPlan2D {
    nx: usize,
    ny: usize,
    precision: PrecisionProfile,
}

impl std::fmt::Debug for FftPlan2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan2D")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("precision", &self.precision)
            .finish()
    }
}

impl FftPlan2D {
    /// Create a new 2D plan.
    #[must_use]
    pub fn new(shape: Shape2D) -> Self {
        Self::with_precision(shape, PrecisionProfile::HIGH_ACCURACY_F64)
    }

    /// Create a new 2D plan with an explicit precision profile.
    #[must_use]
    pub fn with_precision(shape: Shape2D, precision: PrecisionProfile) -> Self {
        Self {
            nx: shape.nx,
            ny: shape.ny,
            precision,
        }
    }

    /// Return the precision profile used by this plan.
    #[must_use]
    pub fn precision_profile(&self) -> PrecisionProfile {
        self.precision
    }

    /// Return the validated shape owned by this plan.
    #[must_use]
    pub fn shape(&self) -> Shape2D {
        Shape2D {
            nx: self.nx,
            ny: self.ny,
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
        assert_eq!(
            input.dim(),
            (self.nx, self.ny),
            "forward input shape mismatch"
        );
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
        assert_eq!(
            input.dim(),
            (self.nx, self.ny),
            "forward input shape mismatch"
        );
        assert_eq!(
            output.dim(),
            (self.nx, self.ny),
            "forward output shape mismatch"
        );
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
        assert_eq!(
            scratch.dim(),
            (self.nx, self.ny),
            "inverse scratch shape mismatch"
        );
        scratch.assign(input);
        self.inverse_complex_inplace(scratch);
        Zip::from(output).and(scratch).for_each(|out, value| {
            *out = value.re;
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
        let mut transformed = input.clone();
        self.inverse_complex_inplace(&mut transformed);
        Zip::from(output).and(&transformed).for_each(|out, value| {
            *out = value.re;
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
        assert_eq!(
            data.dim(),
            (self.nx, self.ny),
            "complex forward shape mismatch"
        );
        self.axis_pass_complex(data, Axis(1), true);
        self.axis_pass_complex(data, Axis(0), true);
    }

    /// Inverse transform of a complex array in-place with normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array2<Complex64>) {
        assert_eq!(
            data.dim(),
            (self.nx, self.ny),
            "complex inverse shape mismatch"
        );
        self.axis_pass_complex(data, Axis(0), false);
        self.axis_pass_complex(data, Axis(1), false);
    }

    /// Forward transform of a real array stored as `f32`.
    #[must_use]
    pub(crate) fn forward_f32(&self, input: &Array2<f32>) -> Array2<Complex32> {
        assert_eq!(
            input.dim(),
            (self.nx, self.ny),
            "forward input shape mismatch"
        );
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            let mut data = input.mapv(|value| Complex32::new(value, 0.0));
            self.forward_complex_inplace_f32(&mut data);
            data
        } else {
            let promoted = input.mapv(f64::from);
            self.forward_real_to_complex(&promoted)
                .mapv(|value| Complex32::new(value.re as f32, value.im as f32))
        }
    }

    /// Inverse transform of an `f32`-storage complex spectrum.
    #[must_use]
    pub(crate) fn inverse_f32(&self, input: &Array2<Complex32>) -> Array2<f32> {
        assert_eq!(
            input.dim(),
            (self.nx, self.ny),
            "inverse input shape mismatch"
        );
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            let mut data = input.clone();
            self.inverse_complex_inplace_f32(&mut data);
            data.mapv(|value| value.re)
        } else {
            let promoted =
                input.mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im)));
            self.inverse_complex_to_real(&promoted)
                .mapv(|value| value as f32)
        }
    }

    /// Forward transform of a real array stored as `f16`.
    #[must_use]
    pub(crate) fn forward_f16(&self, input: &Array2<f16>) -> Array2<Complex32> {
        assert_eq!(
            input.dim(),
            (self.nx, self.ny),
            "forward input shape mismatch"
        );
        if self.precision == PrecisionProfile::MIXED_PRECISION_F16_F32 {
            let mut data = input.mapv(|value| Complex32::new(value.to_f32(), 0.0));
            self.forward_complex_inplace_f32(&mut data);
            data
        } else {
            let promoted = input.mapv(|value| f64::from(value.to_f32()));
            self.forward_real_to_complex(&promoted)
                .mapv(|value| Complex32::new(value.re as f32, value.im as f32))
        }
    }

    /// Inverse transform of a complex spectrum to `f16` storage.
    #[must_use]
    pub(crate) fn inverse_f16(&self, input: &Array2<Complex32>) -> Array2<f16> {
        assert_eq!(
            input.dim(),
            (self.nx, self.ny),
            "inverse input shape mismatch"
        );
        if self.precision == PrecisionProfile::MIXED_PRECISION_F16_F32 {
            let mut data = input.clone();
            self.inverse_complex_inplace_f32(&mut data);
            data.mapv(|value| f16::from_f32(value.re))
        } else {
            let promoted =
                input.mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im)));
            self.inverse_complex_to_real(&promoted)
                .mapv(|value| f16::from_f32(value as f32))
        }
    }

    fn axis_pass_complex(&self, data: &mut Array2<Complex64>, axis: Axis, forward: bool) {
        if axis.index() == 1 {
            self.axis1_pass_complex(data, forward);
            return;
        }
        if axis.index() == 0 {
            self.axis0_pass_complex(data, forward);
            return;
        }

        let mut lanes: Vec<Vec<Complex64>> = data
            .lanes(axis)
            .into_iter()
            .map(|lane| lane.to_vec())
            .collect();
        lanes.par_iter_mut().for_each(|lane| {
            if forward {
                fft_forward_64(lane);
            } else {
                fft_inverse_64(lane);
            }
        });
        for (mut lane, values) in data.lanes_mut(axis).into_iter().zip(lanes.into_iter()) {
            for (slot, value) in lane.iter_mut().zip(values.into_iter()) {
                *slot = value;
            }
        }
    }

    fn axis1_pass_complex(&self, data: &mut Array2<Complex64>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("2D complex data must be contiguous");
        data_slice.par_chunks_mut(self.ny).for_each(|lane| {
            if forward {
                fft_forward_64(lane);
            } else {
                fft_inverse_64(lane);
            }
        });
    }

    fn axis0_pass_complex(&self, data: &mut Array2<Complex64>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("2D complex data must be contiguous");
        let mut lanes = vec![Complex64::default(); self.nx * self.ny];
        for col in 0..self.ny {
            for row in 0..self.nx {
                lanes[col * self.nx + row] = data_slice[row * self.ny + col];
            }
        }

        lanes.par_chunks_mut(self.nx).for_each(|lane| {
            if forward {
                fft_forward_64(lane);
            } else {
                fft_inverse_64(lane);
            }
        });

        for col in 0..self.ny {
            for row in 0..self.nx {
                data_slice[row * self.ny + col] = lanes[col * self.nx + row];
            }
        }
    }

    fn forward_complex_inplace_f32(&self, data: &mut Array2<Complex32>) {
        self.axis_pass_complex_f32(data, Axis(1), true);
        self.axis_pass_complex_f32(data, Axis(0), true);
    }

    fn inverse_complex_inplace_f32(&self, data: &mut Array2<Complex32>) {
        self.axis_pass_complex_f32(data, Axis(0), false);
        self.axis_pass_complex_f32(data, Axis(1), false);
    }

    fn axis_pass_complex_f32(&self, data: &mut Array2<Complex32>, axis: Axis, forward: bool) {
        if axis.index() == 1 {
            self.axis1_pass_complex_f32(data, forward);
            return;
        }
        if axis.index() == 0 {
            self.axis0_pass_complex_f32(data, forward);
            return;
        }

        let mut lanes: Vec<Vec<Complex32>> = data
            .lanes(axis)
            .into_iter()
            .map(|lane| lane.to_vec())
            .collect();
        lanes.par_iter_mut().for_each(|lane| {
            if forward {
                fft_forward_32(lane);
            } else {
                fft_inverse_32(lane);
            }
        });
        for (mut lane, values) in data.lanes_mut(axis).into_iter().zip(lanes.into_iter()) {
            for (slot, value) in lane.iter_mut().zip(values.into_iter()) {
                *slot = value;
            }
        }
    }

    fn axis1_pass_complex_f32(&self, data: &mut Array2<Complex32>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("2D f32 complex data must be contiguous");
        data_slice.par_chunks_mut(self.ny).for_each(|lane| {
            if forward {
                fft_forward_32(lane);
            } else {
                fft_inverse_32(lane);
            }
        });
    }

    fn axis0_pass_complex_f32(&self, data: &mut Array2<Complex32>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("2D f32 complex data must be contiguous");
        let mut lanes = vec![Complex32::default(); self.nx * self.ny];
        for col in 0..self.ny {
            for row in 0..self.nx {
                lanes[col * self.nx + row] = data_slice[row * self.ny + col];
            }
        }

        lanes.par_chunks_mut(self.nx).for_each(|lane| {
            if forward {
                fft_forward_32(lane);
            } else {
                fft_inverse_32(lane);
            }
        });

        for col in 0..self.ny {
            for row in 0..self.nx {
                data_slice[row * self.ny + col] = lanes[col * self.nx + row];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn roundtrip_recovers_asymmetric_inputs() {
        for &(nx, ny) in &[(7, 13), (13, 7), (16, 24), (24, 16), (120, 360)] {
            let shape = Shape2D::new(nx, ny).expect("test dimensions are non-zero");
            let plan = FftPlan2D::new(shape);
            let input = Array2::from_shape_fn((nx, ny), |(i, j)| {
                let x = i as f64 / nx as f64;
                let y = j as f64 / ny as f64;
                (std::f64::consts::TAU * x).sin()
                    + 0.5 * (2.0 * std::f64::consts::TAU * y).cos()
                    + 0.125 * ((i * 3 + j * 5) as f64).sin()
            });

            let spectrum = plan.forward(&input);
            let recovered = plan.inverse(&spectrum);

            for ((i, j), expected) in input.indexed_iter() {
                assert_abs_diff_eq!(recovered[(i, j)], *expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn complex_inverse_inplace_is_normalized() {
        let nx = 5;
        let ny = 9;
        let shape = Shape2D::new(nx, ny).expect("test dimensions are non-zero");
        let plan = FftPlan2D::new(shape);
        let input = Array2::from_shape_fn((nx, ny), |(i, j)| {
            Complex64::new((i + 2 * j) as f64, (2 * i + j) as f64 / 3.0)
        });

        let mut spectrum = input.clone();
        plan.forward_complex_inplace(&mut spectrum);
        plan.inverse_complex_inplace(&mut spectrum);

        for ((i, j), expected) in input.indexed_iter() {
            assert_abs_diff_eq!(spectrum[(i, j)].re, expected.re, epsilon = 1e-9);
            assert_abs_diff_eq!(spectrum[(i, j)].im, expected.im, epsilon = 1e-9);
        }
    }
}
