//! 3D FFT plan.

use super::{AXIS_BUF, RFFT_REAL_BUF};
use crate::domain::error::{ApolloError, ApolloResult};
use ndarray::{Array3, Axis, Zip};
use num_complex::Complex64;
use rayon::prelude::*;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Reusable 3D FFT plan.
///
/// The three-dimensional DFT is separable, so Apollo applies one-dimensional
/// FFT passes across Z, then Y, then X for the forward transform, and the
/// reverse order for inverse transforms. Real-to-complex transforms exploit
/// Hermitian symmetry along the Z axis and expose the half-spectrum layout
/// used by PSTD-style solvers.
pub struct FftPlan3D {
    nx: usize,
    ny: usize,
    nz: usize,
    nz_c: usize,
    fft_x: Arc<dyn Fft<f64>>,
    fft_y: Arc<dyn Fft<f64>>,
    fft_z: Arc<dyn Fft<f64>>,
    ifft_x: Arc<dyn Fft<f64>>,
    ifft_y: Arc<dyn Fft<f64>>,
    ifft_z: Arc<dyn Fft<f64>>,
    rfft_z: Arc<dyn RealToComplex<f64>>,
    irfft_z: Arc<dyn ComplexToReal<f64>>,
}

impl std::fmt::Debug for FftPlan3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan3D")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .field("nz_c", &self.nz_c)
            .finish()
    }
}

impl FftPlan3D {
    /// Create a new 3D plan.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let mut planner = FftPlanner::new();
        let mut real_planner = RealFftPlanner::<f64>::new();
        let nz_c = nz / 2 + 1;
        Self {
            nx,
            ny,
            nz,
            nz_c,
            fft_x: planner.plan_fft_forward(nx),
            fft_y: planner.plan_fft_forward(ny),
            fft_z: planner.plan_fft_forward(nz),
            ifft_x: planner.plan_fft_inverse(nx),
            ifft_y: planner.plan_fft_inverse(ny),
            ifft_z: planner.plan_fft_inverse(nz),
            rfft_z: real_planner.plan_fft_forward(nz),
            irfft_z: real_planner.plan_fft_inverse(nz),
        }
    }

    /// Return the number of independent complex Z bins in the half-spectrum layout.
    #[must_use]
    pub fn nz_c(&self) -> usize {
        self.nz_c
    }

    /// Alias for `nz_c()` used by the Apollo API docs.
    #[must_use]
    pub fn nz_complex(&self) -> usize {
        self.nz_c()
    }

    /// Return the full real-domain shape owned by this plan.
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// Forward transform of a real 3D field.
    #[must_use]
    pub fn forward(&self, input: &Array3<f64>) -> Array3<Complex64> {
        self.forward_real_to_complex(input)
    }

    /// Inverse transform of a full complex 3D spectrum.
    #[must_use]
    pub fn inverse(&self, input: &Array3<Complex64>) -> Array3<f64> {
        self.inverse_complex_to_real(input)
    }

    /// Forward transform of a complex field.
    #[must_use]
    pub fn forward_complex(&self, input: &Array3<Complex64>) -> Array3<Complex64> {
        let mut output = input.clone();
        self.forward_complex_inplace(&mut output);
        output
    }

    /// Inverse transform of a complex field.
    #[must_use]
    pub fn inverse_complex(&self, input: &Array3<Complex64>) -> Array3<Complex64> {
        let mut output = input.clone();
        self.inverse_complex_inplace(&mut output);
        output
    }

    /// Forward transform of a real field.
    #[must_use]
    pub fn forward_real_to_complex(&self, input: &Array3<f64>) -> Array3<Complex64> {
        let mut output = Array3::<Complex64>::zeros((self.nx, self.ny, self.nz));
        self.forward_real_to_complex_into_full(input, &mut output);
        output
    }

    /// Forward transform of a real field into a full complex spectrum buffer.
    pub fn forward_real_to_complex_into(
        &self,
        input: &Array3<f64>,
        output: &mut Array3<Complex64>,
    ) {
        self.forward_real_to_complex_into_full(input, output);
    }

    /// Compatibility alias for `forward_real_to_complex_into`.
    pub fn forward_into(&self, input: &Array3<f64>, output: &mut Array3<Complex64>) {
        self.forward_real_to_complex_into_full(input, output);
    }

    /// Inverse transform of a full complex spectrum to a real field.
    #[must_use]
    pub fn inverse_complex_to_real(&self, input: &Array3<Complex64>) -> Array3<f64> {
        let mut output = Array3::<f64>::zeros((self.nx, self.ny, self.nz));
        let mut scratch = Array3::<Complex64>::zeros((self.nx, self.ny, self.nz));
        self.inverse_complex_to_real_into(input, &mut output, &mut scratch);
        output
    }

    /// Inverse transform into caller-owned output and scratch buffers.
    pub fn inverse_complex_to_real_into(
        &self,
        input: &Array3<Complex64>,
        output: &mut Array3<f64>,
        scratch: &mut Array3<Complex64>,
    ) {
        self.check_full_complex_shape(input.dim(), "inverse input");
        self.check_real_shape(output.dim(), "inverse output");
        self.check_full_complex_shape(scratch.dim(), "inverse scratch");
        scratch.assign(input);
        self.inverse_complex_inplace(scratch);
        let norm = 1.0 / (self.nx * self.ny * self.nz) as f64;
        Zip::from(output)
            .and(scratch.view())
            .par_for_each(|out, value| {
                *out = value.re * norm;
            });
    }

    /// Compatibility alias for `inverse_complex_to_real_into`.
    pub fn inverse_into(
        &self,
        input: &Array3<Complex64>,
        output: &mut Array3<f64>,
        scratch: &mut Array3<Complex64>,
    ) {
        self.inverse_complex_to_real_into(input, output, scratch);
    }

    /// In-place inverse transform writing a normalized real field to `output`.
    pub fn inverse_inplace(&self, data: &mut Array3<Complex64>, output: &mut Array3<f64>) {
        self.check_full_complex_shape(data.dim(), "inverse_inplace data");
        self.check_real_shape(output.dim(), "inverse_inplace output");
        self.inverse_complex_inplace(data);
        let norm = 1.0 / (self.nx * self.ny * self.nz) as f64;
        Zip::from(output)
            .and(data.view())
            .par_for_each(|out, value| {
                *out = value.re * norm;
            });
    }

    /// Forward real-to-complex transform into a half-spectrum buffer.
    ///
    /// The proof obligation is Hermitian symmetry for real-valued input. Because
    /// the negative-frequency half is the complex conjugate of the positive half,
    /// only `nz / 2 + 1` Z bins are stored.
    pub fn forward_r2c_into(&self, input: &Array3<f64>, output: &mut Array3<Complex64>) {
        self.check_real_shape(input.dim(), "forward_r2c_into input");
        assert_eq!(
            output.dim(),
            (self.nx, self.ny, self.nz_c),
            "forward_r2c_into: output shape mismatch"
        );
        assert!(
            input.is_standard_layout(),
            "forward_r2c_into: input must be C-contiguous"
        );
        assert!(
            output.is_standard_layout(),
            "forward_r2c_into: output must be C-contiguous"
        );

        let rfft_z = Arc::clone(&self.rfft_z);
        let fft_y = Arc::clone(&self.fft_y);
        let fft_x = Arc::clone(&self.fft_x);
        let nz = self.nz;
        let nz_c = self.nz_c;
        let ny = self.ny;
        let nx = self.nx;

        output
            .outer_iter_mut()
            .into_par_iter()
            .zip(input.outer_iter())
            .for_each(|(mut out_x, in_x)| {
                for (mut out_row, in_row) in out_x.outer_iter_mut().zip(in_x.outer_iter()) {
                    RFFT_REAL_BUF.with(|cell| {
                        let mut scratch = cell.borrow_mut();
                        if scratch.len() < nz {
                            scratch.resize(nz, 0.0);
                        }
                        for (dst, src) in scratch[..nz].iter_mut().zip(in_row.iter()) {
                            *dst = *src;
                        }
                        rfft_z
                            .process(
                                &mut scratch[..nz],
                                out_row
                                    .as_slice_mut()
                                    .expect("forward_r2c_into: z row must be contiguous"),
                            )
                            .expect("forward_r2c_into: rfft_z failed");
                    });
                }
            });

        output
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut x_slice| {
                for k in 0..nz_c {
                    AXIS_BUF.with(|cell| {
                        let mut buffer = cell.borrow_mut();
                        if buffer.len() < ny {
                            buffer.resize(ny, Complex64::default());
                        }
                        for j in 0..ny {
                            buffer[j] = x_slice[[j, k]];
                        }
                        fft_y.process(&mut buffer[..ny]);
                        for j in 0..ny {
                            x_slice[[j, k]] = buffer[j];
                        }
                    });
                }
            });

        output
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut y_slice| {
                for k in 0..nz_c {
                    AXIS_BUF.with(|cell| {
                        let mut buffer = cell.borrow_mut();
                        if buffer.len() < nx {
                            buffer.resize(nx, Complex64::default());
                        }
                        for i in 0..nx {
                            buffer[i] = y_slice[[i, k]];
                        }
                        fft_x.process(&mut buffer[..nx]);
                        for i in 0..nx {
                            y_slice[[i, k]] = buffer[i];
                        }
                    });
                }
            });
    }

    /// Inverse half-spectrum transform back to a real field.
    ///
    /// The algorithm applies inverse X and Y complex passes, then a final
    /// complex-to-real inverse along Z. The inverse is normalized by
    /// `1 / (nx * ny * nz)`.
    pub fn inverse_c2r_inplace(&self, data: &mut Array3<Complex64>, output: &mut Array3<f64>) {
        assert_eq!(
            data.dim(),
            (self.nx, self.ny, self.nz_c),
            "inverse_c2r_inplace: data shape mismatch"
        );
        self.check_real_shape(output.dim(), "inverse_c2r_inplace output");
        assert!(
            data.is_standard_layout(),
            "inverse_c2r_inplace: data must be C-contiguous"
        );
        assert!(
            output.is_standard_layout(),
            "inverse_c2r_inplace: output must be C-contiguous"
        );

        let ifft_x = Arc::clone(&self.ifft_x);
        let ifft_y = Arc::clone(&self.ifft_y);
        let irfft_z = Arc::clone(&self.irfft_z);
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nz_c = self.nz_c;

        data.axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut y_slice| {
                for k in 0..nz_c {
                    AXIS_BUF.with(|cell| {
                        let mut buffer = cell.borrow_mut();
                        if buffer.len() < nx {
                            buffer.resize(nx, Complex64::default());
                        }
                        for i in 0..nx {
                            buffer[i] = y_slice[[i, k]];
                        }
                        ifft_x.process(&mut buffer[..nx]);
                        for i in 0..nx {
                            y_slice[[i, k]] = buffer[i];
                        }
                    });
                }
            });

        data.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut x_slice| {
                for k in 0..nz_c {
                    AXIS_BUF.with(|cell| {
                        let mut buffer = cell.borrow_mut();
                        if buffer.len() < ny {
                            buffer.resize(ny, Complex64::default());
                        }
                        for j in 0..ny {
                            buffer[j] = x_slice[[j, k]];
                        }
                        ifft_y.process(&mut buffer[..ny]);
                        for j in 0..ny {
                            x_slice[[j, k]] = buffer[j];
                        }
                    });
                }
            });

        output
            .outer_iter_mut()
            .into_par_iter()
            .zip(data.outer_iter_mut())
            .for_each(|(mut out_x, mut data_x)| {
                for (mut out_row, mut data_row) in
                    out_x.outer_iter_mut().zip(data_x.outer_iter_mut())
                {
                    RFFT_REAL_BUF.with(|cell| {
                        let mut scratch = cell.borrow_mut();
                        if scratch.len() < nz {
                            scratch.resize(nz, 0.0);
                        }
                        {
                            let row = data_row
                                .as_slice_mut()
                                .expect("inverse_c2r_inplace: z row must be contiguous");
                            row[0].im = 0.0;
                            if nz % 2 == 0 {
                                row[nz_c - 1].im = 0.0;
                            }
                        }
                        irfft_z
                            .process(
                                data_row
                                    .as_slice_mut()
                                    .expect("inverse_c2r_inplace: z row must be contiguous"),
                                &mut scratch[..nz],
                            )
                            .expect("inverse_c2r_inplace: irfft_z failed");
                        let norm = 1.0 / (nx * ny * nz) as f64;
                        for (dst, src) in out_row.iter_mut().zip(scratch[..nz].iter()) {
                            *dst = *src * norm;
                        }
                    });
                }
            });
    }

    /// Forward two real fields in parallel into half-spectrum buffers.
    pub fn forward_r2c_2x_parallel(
        &self,
        first: (&Array3<f64>, &mut Array3<Complex64>),
        second: (&Array3<f64>, &mut Array3<Complex64>),
    ) {
        rayon::join(
            || self.forward_r2c_into(first.0, first.1),
            || self.forward_r2c_into(second.0, second.1),
        );
    }

    /// Inverse two half-spectrum fields in parallel into real buffers.
    pub fn inverse_c2r_2x_parallel(
        &self,
        first: (&mut Array3<Complex64>, &mut Array3<f64>),
        second: (&mut Array3<Complex64>, &mut Array3<f64>),
    ) {
        rayon::join(
            || self.inverse_c2r_inplace(first.0, first.1),
            || self.inverse_c2r_inplace(second.0, second.1),
        );
    }

    /// Forward three real fields in parallel into half-spectrum buffers.
    pub fn forward_r2c_3x_parallel(
        &self,
        first: (&Array3<f64>, &mut Array3<Complex64>),
        second: (&Array3<f64>, &mut Array3<Complex64>),
        third: (&Array3<f64>, &mut Array3<Complex64>),
    ) {
        rayon::scope(|scope| {
            scope.spawn(|_| self.forward_r2c_into(first.0, first.1));
            scope.spawn(|_| self.forward_r2c_into(second.0, second.1));
            self.forward_r2c_into(third.0, third.1);
        });
    }

    /// Inverse three half-spectrum fields in parallel into real buffers.
    pub fn inverse_c2r_3x_parallel(
        &self,
        first: (&mut Array3<Complex64>, &mut Array3<f64>),
        second: (&mut Array3<Complex64>, &mut Array3<f64>),
        third: (&mut Array3<Complex64>, &mut Array3<f64>),
    ) {
        rayon::scope(|scope| {
            scope.spawn(|_| self.inverse_c2r_inplace(first.0, first.1));
            scope.spawn(|_| self.inverse_c2r_inplace(second.0, second.1));
            self.inverse_c2r_inplace(third.0, third.1);
        });
    }

    /// Forward complex transform in-place.
    pub fn forward_complex_inplace(&self, data: &mut Array3<Complex64>) {
        self.check_full_complex_shape(data.dim(), "forward_complex_inplace");
        let fft_z = Arc::clone(&self.fft_z);
        let fft_y = Arc::clone(&self.fft_y);
        let fft_x = Arc::clone(&self.fft_x);
        let ny = self.ny;
        let nx = self.nx;
        let nz = self.nz;

        data.outer_iter_mut()
            .into_par_iter()
            .for_each(|mut x_slice| {
                for mut row in x_slice.outer_iter_mut() {
                    fft_z.process(row.as_slice_mut().expect("z row must be contiguous"));
                }
            });

        data.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut x_slice| {
                for k in 0..nz {
                    AXIS_BUF.with(|cell| {
                        let mut buffer = cell.borrow_mut();
                        if buffer.len() < ny {
                            buffer.resize(ny, Complex64::default());
                        }
                        for j in 0..ny {
                            buffer[j] = x_slice[[j, k]];
                        }
                        fft_y.process(&mut buffer[..ny]);
                        for j in 0..ny {
                            x_slice[[j, k]] = buffer[j];
                        }
                    });
                }
            });

        data.axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut y_slice| {
                for k in 0..nz {
                    AXIS_BUF.with(|cell| {
                        let mut buffer = cell.borrow_mut();
                        if buffer.len() < nx {
                            buffer.resize(nx, Complex64::default());
                        }
                        for i in 0..nx {
                            buffer[i] = y_slice[[i, k]];
                        }
                        fft_x.process(&mut buffer[..nx]);
                        for i in 0..nx {
                            y_slice[[i, k]] = buffer[i];
                        }
                    });
                }
            });
    }

    /// Inverse complex transform in-place without normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array3<Complex64>) {
        self.check_full_complex_shape(data.dim(), "inverse_complex_inplace");
        let ifft_x = Arc::clone(&self.ifft_x);
        let ifft_y = Arc::clone(&self.ifft_y);
        let ifft_z = Arc::clone(&self.ifft_z);
        let ny = self.ny;
        let nx = self.nx;
        let nz = self.nz;

        data.axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut y_slice| {
                for k in 0..nz {
                    AXIS_BUF.with(|cell| {
                        let mut buffer = cell.borrow_mut();
                        if buffer.len() < nx {
                            buffer.resize(nx, Complex64::default());
                        }
                        for i in 0..nx {
                            buffer[i] = y_slice[[i, k]];
                        }
                        ifft_x.process(&mut buffer[..nx]);
                        for i in 0..nx {
                            y_slice[[i, k]] = buffer[i];
                        }
                    });
                }
            });

        data.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut x_slice| {
                for k in 0..nz {
                    AXIS_BUF.with(|cell| {
                        let mut buffer = cell.borrow_mut();
                        if buffer.len() < ny {
                            buffer.resize(ny, Complex64::default());
                        }
                        for j in 0..ny {
                            buffer[j] = x_slice[[j, k]];
                        }
                        ifft_y.process(&mut buffer[..ny]);
                        for j in 0..ny {
                            x_slice[[j, k]] = buffer[j];
                        }
                    });
                }
            });

        data.outer_iter_mut()
            .into_par_iter()
            .for_each(|mut x_slice| {
                for mut row in x_slice.outer_iter_mut() {
                    ifft_z.process(row.as_slice_mut().expect("z row must be contiguous"));
                }
            });
    }

    /// Compute a spectral derivative along one Cartesian axis.
    ///
    /// The derivative theorem for the Fourier transform states that
    /// `F[∂f/∂x] = i k_x F[f]`. Apollo applies that multiplication in the
    /// spectral domain and then returns to the spatial domain.
    pub fn spectral_derivative(
        &self,
        field: &Array3<f64>,
        axis: usize,
    ) -> ApolloResult<Array3<f64>> {
        let mut spectrum = self.forward(field);
        match axis {
            0 => {
                let dk = 2.0 * std::f64::consts::PI / self.nx as f64;
                for i in 0..self.nx {
                    let k = if i <= self.nx / 2 {
                        i as f64 * dk
                    } else {
                        (i as f64 - self.nx as f64) * dk
                    };
                    let factor = Complex64::new(0.0, k);
                    spectrum
                        .index_axis_mut(Axis(0), i)
                        .par_mapv_inplace(|value| value * factor);
                }
            }
            1 => {
                let dk = 2.0 * std::f64::consts::PI / self.ny as f64;
                for j in 0..self.ny {
                    let k = if j <= self.ny / 2 {
                        j as f64 * dk
                    } else {
                        (j as f64 - self.ny as f64) * dk
                    };
                    let factor = Complex64::new(0.0, k);
                    spectrum
                        .index_axis_mut(Axis(1), j)
                        .par_mapv_inplace(|value| value * factor);
                }
            }
            2 => {
                let dk = 2.0 * std::f64::consts::PI / self.nz as f64;
                for k_index in 0..self.nz {
                    let k = if k_index <= self.nz / 2 {
                        k_index as f64 * dk
                    } else {
                        (k_index as f64 - self.nz as f64) * dk
                    };
                    let factor = Complex64::new(0.0, k);
                    spectrum
                        .index_axis_mut(Axis(2), k_index)
                        .par_mapv_inplace(|value| value * factor);
                }
            }
            _ => {
                return Err(ApolloError::validation(
                    "axis",
                    axis.to_string(),
                    "Axis must be 0, 1, or 2",
                ));
            }
        }

        Ok(self.inverse(&spectrum))
    }

    fn forward_real_to_complex_into_full(
        &self,
        input: &Array3<f64>,
        output: &mut Array3<Complex64>,
    ) {
        self.check_real_shape(input.dim(), "forward input");
        self.check_full_complex_shape(output.dim(), "forward output");
        Zip::from(&mut *output)
            .and(input)
            .par_for_each(|dst, &src| {
                *dst = Complex64::new(src, 0.0);
            });
        self.forward_complex_inplace(output);
    }

    fn check_real_shape(&self, shape: (usize, usize, usize), context: &str) {
        assert_eq!(
            shape,
            (self.nx, self.ny, self.nz),
            "{context}: expected ({}, {}, {}), got {:?}",
            self.nx,
            self.ny,
            self.nz,
            shape
        );
    }

    fn check_full_complex_shape(&self, shape: (usize, usize, usize), context: &str) {
        assert_eq!(
            shape,
            (self.nx, self.ny, self.nz),
            "{context}: expected ({}, {}, {}), got {:?}",
            self.nx,
            self.ny,
            self.nz,
            shape
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{Array1, Array3};
    use proptest::prelude::*;

    #[test]
    fn roundtrip_matches_for_delta() {
        let n = 4usize;
        let mut input = Array3::<f64>::zeros((n, n, n));
        input[[2, 2, 2]] = 1.0;
        let plan = FftPlan3D::new(n, n, n);
        let spectrum = plan.forward(&input);
        let recovered = plan.inverse(&spectrum);
        for (lhs, rhs) in input.iter().zip(recovered.iter()) {
            assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
        }
    }

    #[test]
    fn parseval_holds_in_1d_projection() {
        let n = 64usize;
        let plan = crate::application::plan::FftPlan1D::new(n);
        let data = Array1::from_shape_fn(n, |i| {
            (i as f64 * 0.37).sin() + 0.5 * (i as f64 * 1.1).cos()
        });
        let spectrum = plan.forward(&data);
        let spatial: f64 = data.iter().map(|value| value * value).sum();
        let spectral: f64 = spectrum.iter().map(|value| value.norm_sqr()).sum::<f64>() / n as f64;
        assert_relative_eq!(spatial, spectral, epsilon = 1e-10);
    }

    #[test]
    fn r2c_roundtrip_recovers_input() {
        let (nx, ny, nz) = (8, 8, 8);
        let plan = FftPlan3D::new(nx, ny, nz);
        let field = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            (i as f64 * 0.5 + j as f64 * 0.3 + k as f64 * 0.7).sin()
        });
        let mut spectrum = Array3::<Complex64>::zeros((nx, ny, plan.nz_c()));
        plan.forward_r2c_into(&field, &mut spectrum);
        let mut recovered = Array3::<f64>::zeros((nx, ny, nz));
        plan.inverse_c2r_inplace(&mut spectrum, &mut recovered);
        let max_diff = field
            .iter()
            .zip(recovered.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 1e-12, "r2c roundtrip max diff = {max_diff:.3e}");
    }

    #[test]
    fn rejects_invalid_derivative_axis() {
        let plan = FftPlan3D::new(4, 4, 4);
        let field = Array3::<f64>::zeros((4, 4, 4));
        let error = plan.spectral_derivative(&field, 3).unwrap_err();
        assert!(matches!(error, ApolloError::Validation { .. }));
    }

    proptest! {
        #[test]
        fn 1d_roundtrip_holds_for_random_lengths_and_signals(
            n in 2usize..48,
            seed in 0u64..500,
        ) {
            let plan = crate::application::plan::FftPlan1D::new(n);
            let signal = Array1::from_shape_fn(n, |i| {
                let x = i as f64 + seed as f64 * 0.01;
                (x * 0.31).sin() + 0.5 * (x * 0.17).cos()
            });
            let recovered = plan.inverse(&plan.forward(&signal));
            let max_err = signal
                .iter()
                .zip(recovered.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0_f64, f64::max);
            prop_assert!(max_err < 1e-12);
        }

        #[test]
        fn 3d_roundtrip_holds_for_small_random_shapes(
            nx in 2usize..6,
            ny in 2usize..6,
            nz in 2usize..6,
            seed in 0u64..200,
        ) {
            let plan = FftPlan3D::new(nx, ny, nz);
            let field = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
                let x = i as f64 + 0.3 * j as f64 + 0.7 * k as f64 + seed as f64 * 0.01;
                (x * 0.21).sin() - 0.4 * (x * 0.13).cos()
            });
            let recovered = plan.inverse(&plan.forward(&field));
            let max_err = field
                .iter()
                .zip(recovered.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0_f64, f64::max);
            prop_assert!(max_err < 1e-11);
        }
    }
}
