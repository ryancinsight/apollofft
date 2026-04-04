//! 3D FFT plan.

use super::{
    RealFftData, AXIS_BUF, AXIS_BUF_2D, AXIS_BUF_2D_32, AXIS_BUF_32, AXIS_SCRATCH, AXIS_SCRATCH_32,
    HALF_SPECTRUM_BUF, RFFT_REAL_BUF, VOLUME_COMPLEX_BUF,
};
use crate::domain::error::{ApolloError, ApolloResult};
use crate::types::PrecisionProfile;
use half::f16;
use ndarray::{Array3, ArrayBase, ArrayViewMut3, Axis, DataMut, Ix3, Zip};
use num_complex::Complex32;
use num_complex::Complex64;
use rayon::prelude::*;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

const PARALLEL_VOLUME_THRESHOLD: usize = 32 * 32 * 32;
const PARALLEL_BATCHES_PER_THREAD: usize = 4;
const HERMITIAN_TOLERANCE: f64 = 1e-12;

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
    precision: PrecisionProfile,
    fft_x: Arc<dyn Fft<f64>>,
    fft_y: Arc<dyn Fft<f64>>,
    fft_z: Arc<dyn Fft<f64>>,
    ifft_x: Arc<dyn Fft<f64>>,
    ifft_y: Arc<dyn Fft<f64>>,
    ifft_z: Arc<dyn Fft<f64>>,
    rfft_z: Arc<dyn RealToComplex<f64>>,
    irfft_z: Arc<dyn ComplexToReal<f64>>,
    fft_x_f32: Arc<dyn Fft<f32>>,
    fft_y_f32: Arc<dyn Fft<f32>>,
    fft_z_f32: Arc<dyn Fft<f32>>,
    ifft_x_f32: Arc<dyn Fft<f32>>,
    ifft_y_f32: Arc<dyn Fft<f32>>,
    ifft_z_f32: Arc<dyn Fft<f32>>,
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
        Self::with_precision(nx, ny, nz, PrecisionProfile::HIGH_ACCURACY_F64)
    }

    /// Create a new 3D plan with an explicit precision profile.
    #[must_use]
    pub fn with_precision(nx: usize, ny: usize, nz: usize, precision: PrecisionProfile) -> Self {
        let mut planner = FftPlanner::new();
        let mut planner_f32 = FftPlanner::<f32>::new();
        let mut real_planner = RealFftPlanner::<f64>::new();
        let nz_c = nz / 2 + 1;
        Self {
            nx,
            ny,
            nz,
            nz_c,
            precision,
            fft_x: planner.plan_fft_forward(nx),
            fft_y: planner.plan_fft_forward(ny),
            fft_z: planner.plan_fft_forward(nz),
            ifft_x: planner.plan_fft_inverse(nx),
            ifft_y: planner.plan_fft_inverse(ny),
            ifft_z: planner.plan_fft_inverse(nz),
            rfft_z: real_planner.plan_fft_forward(nz),
            irfft_z: real_planner.plan_fft_inverse(nz),
            fft_x_f32: planner_f32.plan_fft_forward(nx),
            fft_y_f32: planner_f32.plan_fft_forward(ny),
            fft_z_f32: planner_f32.plan_fft_forward(nz),
            ifft_x_f32: planner_f32.plan_fft_inverse(nx),
            ifft_y_f32: planner_f32.plan_fft_inverse(ny),
            ifft_z_f32: planner_f32.plan_fft_inverse(nz),
        }
    }

    /// Return the precision profile used by this plan.
    #[must_use]
    pub fn precision_profile(&self) -> PrecisionProfile {
        self.precision
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

    #[inline]
    fn use_parallel_volume(&self) -> bool {
        self.nx * self.ny * self.nz >= PARALLEL_VOLUME_THRESHOLD
    }

    #[inline]
    fn use_parallel_axis(&self, batch_count: usize) -> bool {
        self.use_parallel_volume()
            && batch_count >= rayon::current_num_threads() * PARALLEL_BATCHES_PER_THREAD
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

    /// Forward transform of a real field using generic storage dispatch.
    #[must_use]
    pub fn forward_typed<T: RealFftData>(&self, input: &Array3<T>) -> Array3<T::Spectrum> {
        T::forward_3d(self, input)
    }

    /// Inverse transform of a complex spectrum using generic storage dispatch.
    #[must_use]
    pub fn inverse_typed<T: RealFftData>(&self, input: &Array3<T::Spectrum>) -> Array3<T> {
        T::inverse_3d(self, input)
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
        self.inverse_complex_to_real_with_workspace(input, &mut output);
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
        if input.is_standard_layout() && output.is_standard_layout() {
            let mut used_half_spectrum = false;
            HALF_SPECTRUM_BUF.with(|cell| {
                let mut half = cell.borrow_mut();
                let half_len = self.nx * self.ny * self.nz_c;
                if half.len() < half_len {
                    half.resize(half_len, Complex64::default());
                }
                if self.try_extract_half_spectrum(input, &mut half[..half_len]) {
                    self.inverse_half_spectrum_to_real_into_slice(&mut half[..half_len], output);
                    used_half_spectrum = true;
                }
            });
            if used_half_spectrum {
                return;
            }
        }
        scratch.assign(input);
        self.inverse_complex_inplace(scratch);
        let norm = 1.0 / (self.nx * self.ny * self.nz) as f64;
        if self.use_parallel_volume() {
            Zip::from(output)
                .and(scratch.view())
                .par_for_each(|out, value| {
                    *out = value.re * norm;
                });
        } else {
            Zip::from(output)
                .and(scratch.view())
                .for_each(|out, value| {
                    *out = value.re * norm;
                });
        }
    }

    fn inverse_complex_to_real_with_workspace(
        &self,
        input: &Array3<Complex64>,
        output: &mut Array3<f64>,
    ) {
        self.check_full_complex_shape(input.dim(), "inverse input");
        self.check_real_shape(output.dim(), "inverse output");
        if input.is_standard_layout() && output.is_standard_layout() {
            let mut used_half_spectrum = false;
            HALF_SPECTRUM_BUF.with(|cell| {
                let mut half = cell.borrow_mut();
                let half_len = self.nx * self.ny * self.nz_c;
                if half.len() < half_len {
                    half.resize(half_len, Complex64::default());
                }
                if self.try_extract_half_spectrum(input, &mut half[..half_len]) {
                    self.inverse_half_spectrum_to_real_into_slice(&mut half[..half_len], output);
                    used_half_spectrum = true;
                }
            });
            if used_half_spectrum {
                return;
            }

            let input_slice = input
                .as_slice_memory_order()
                .expect("inverse input must be contiguous");
            let output_slice = output
                .as_slice_memory_order_mut()
                .expect("inverse output must be contiguous");
            let total = self.nx * self.ny * self.nz;
            VOLUME_COMPLEX_BUF.with(|cell| {
                let mut scratch = cell.borrow_mut();
                if scratch.len() < total {
                    scratch.resize(total, Complex64::default());
                }
                scratch[..total].copy_from_slice(input_slice);
                let mut scratch_view =
                    ArrayViewMut3::from_shape((self.nx, self.ny, self.nz), &mut scratch[..total])
                        .expect("scratch view shape must match plan dimensions");
                self.inverse_complex_inplace_impl(&mut scratch_view);
                let norm = 1.0 / total as f64;
                for (dst, value) in output_slice.iter_mut().zip(scratch[..total].iter()) {
                    *dst = value.re * norm;
                }
            });
            return;
        }

        let mut scratch = Array3::<Complex64>::zeros((self.nx, self.ny, self.nz));
        self.inverse_complex_to_real_into(input, output, &mut scratch);
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
        if !self.use_parallel_volume() {
            self.forward_r2c_into_batched(input, output);
            return;
        }

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
                        AXIS_SCRATCH.with(|scratch_cell| {
                            let mut scratch = scratch_cell.borrow_mut();
                            let len = fft_y.get_inplace_scratch_len();
                            if scratch.len() < len {
                                scratch.resize(len, Complex64::default());
                            }
                            fft_y.process_with_scratch(&mut buffer[..ny], &mut scratch[..len]);
                        });
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
                        AXIS_SCRATCH.with(|scratch_cell| {
                            let mut scratch = scratch_cell.borrow_mut();
                            let len = fft_x.get_inplace_scratch_len();
                            if scratch.len() < len {
                                scratch.resize(len, Complex64::default());
                            }
                            fft_x.process_with_scratch(&mut buffer[..nx], &mut scratch[..len]);
                        });
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
        if !self.use_parallel_volume() {
            let slice = data
                .as_slice_memory_order_mut()
                .expect("inverse_c2r_inplace: data must be contiguous");
            self.inverse_half_spectrum_to_real_into_slice(slice, output);
            return;
        }

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
                        AXIS_SCRATCH.with(|scratch_cell| {
                            let mut scratch = scratch_cell.borrow_mut();
                            let len = ifft_x.get_inplace_scratch_len();
                            if scratch.len() < len {
                                scratch.resize(len, Complex64::default());
                            }
                            ifft_x.process_with_scratch(&mut buffer[..nx], &mut scratch[..len]);
                        });
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
                        AXIS_SCRATCH.with(|scratch_cell| {
                            let mut scratch = scratch_cell.borrow_mut();
                            let len = ifft_y.get_inplace_scratch_len();
                            if scratch.len() < len {
                                scratch.resize(len, Complex64::default());
                            }
                            ifft_y.process_with_scratch(&mut buffer[..ny], &mut scratch[..len]);
                        });
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
        if !self.use_parallel_volume() {
            self.forward_complex_inplace_batched(data);
            return;
        }
        let fft_z = Arc::clone(&self.fft_z);
        let fft_y = Arc::clone(&self.fft_y);
        let fft_x = Arc::clone(&self.fft_x);
        let ny = self.ny;
        let nx = self.nx;
        let nz = self.nz;

        if self.use_parallel_axis(self.nx) {
            data.outer_iter_mut()
                .into_par_iter()
                .for_each(|mut x_slice| {
                    for mut row in x_slice.outer_iter_mut() {
                        AXIS_SCRATCH.with(|scratch_cell| {
                            let mut scratch = scratch_cell.borrow_mut();
                            let len = fft_z.get_inplace_scratch_len();
                            if scratch.len() < len {
                                scratch.resize(len, Complex64::default());
                            }
                            fft_z.process_with_scratch(
                                row.as_slice_mut().expect("z row must be contiguous"),
                                &mut scratch[..len],
                            );
                        });
                    }
                });
        } else {
            data.outer_iter_mut().into_iter().for_each(|mut x_slice| {
                for mut row in x_slice.outer_iter_mut() {
                    AXIS_SCRATCH.with(|scratch_cell| {
                        let mut scratch = scratch_cell.borrow_mut();
                        let len = fft_z.get_inplace_scratch_len();
                        if scratch.len() < len {
                            scratch.resize(len, Complex64::default());
                        }
                        fft_z.process_with_scratch(
                            row.as_slice_mut().expect("z row must be contiguous"),
                            &mut scratch[..len],
                        );
                    });
                }
            });
        }

        if self.use_parallel_axis(self.nx) {
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
                            AXIS_SCRATCH.with(|scratch_cell| {
                                let mut scratch = scratch_cell.borrow_mut();
                                let len = fft_y.get_inplace_scratch_len();
                                if scratch.len() < len {
                                    scratch.resize(len, Complex64::default());
                                }
                                fft_y.process_with_scratch(&mut buffer[..ny], &mut scratch[..len]);
                            });
                            for j in 0..ny {
                                x_slice[[j, k]] = buffer[j];
                            }
                        });
                    }
                });
        } else {
            data.axis_iter_mut(Axis(0))
                .into_iter()
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
                            AXIS_SCRATCH.with(|scratch_cell| {
                                let mut scratch = scratch_cell.borrow_mut();
                                let len = fft_y.get_inplace_scratch_len();
                                if scratch.len() < len {
                                    scratch.resize(len, Complex64::default());
                                }
                                fft_y.process_with_scratch(&mut buffer[..ny], &mut scratch[..len]);
                            });
                            for j in 0..ny {
                                x_slice[[j, k]] = buffer[j];
                            }
                        });
                    }
                });
        }

        if self.use_parallel_axis(self.ny) {
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
                            AXIS_SCRATCH.with(|scratch_cell| {
                                let mut scratch = scratch_cell.borrow_mut();
                                let len = fft_x.get_inplace_scratch_len();
                                if scratch.len() < len {
                                    scratch.resize(len, Complex64::default());
                                }
                                fft_x.process_with_scratch(&mut buffer[..nx], &mut scratch[..len]);
                            });
                            for i in 0..nx {
                                y_slice[[i, k]] = buffer[i];
                            }
                        });
                    }
                });
        } else {
            data.axis_iter_mut(Axis(1))
                .into_iter()
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
                            AXIS_SCRATCH.with(|scratch_cell| {
                                let mut scratch = scratch_cell.borrow_mut();
                                let len = fft_x.get_inplace_scratch_len();
                                if scratch.len() < len {
                                    scratch.resize(len, Complex64::default());
                                }
                                fft_x.process_with_scratch(&mut buffer[..nx], &mut scratch[..len]);
                            });
                            for i in 0..nx {
                                y_slice[[i, k]] = buffer[i];
                            }
                        });
                    }
                });
        }
    }

    /// Inverse complex transform in-place without normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array3<Complex64>) {
        self.inverse_complex_inplace_impl(data);
    }

    fn inverse_complex_inplace_impl<S>(&self, data: &mut ArrayBase<S, Ix3>)
    where
        S: DataMut<Elem = Complex64>,
    {
        self.check_full_complex_shape(data.dim(), "inverse_complex_inplace");
        if !self.use_parallel_volume() {
            let slice = data
                .as_slice_memory_order_mut()
                .expect("inverse batched data must be contiguous");
            self.inverse_complex_slice_inplace_batched(slice);
            return;
        }
        let ifft_x = Arc::clone(&self.ifft_x);
        let ifft_y = Arc::clone(&self.ifft_y);
        let ifft_z = Arc::clone(&self.ifft_z);
        let ny = self.ny;
        let nx = self.nx;
        let nz = self.nz;

        if self.use_parallel_axis(self.ny) {
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
                            AXIS_SCRATCH.with(|scratch_cell| {
                                let mut scratch = scratch_cell.borrow_mut();
                                let len = ifft_x.get_inplace_scratch_len();
                                if scratch.len() < len {
                                    scratch.resize(len, Complex64::default());
                                }
                                ifft_x.process_with_scratch(&mut buffer[..nx], &mut scratch[..len]);
                            });
                            for i in 0..nx {
                                y_slice[[i, k]] = buffer[i];
                            }
                        });
                    }
                });
        } else {
            data.axis_iter_mut(Axis(1))
                .into_iter()
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
                            AXIS_SCRATCH.with(|scratch_cell| {
                                let mut scratch = scratch_cell.borrow_mut();
                                let len = ifft_x.get_inplace_scratch_len();
                                if scratch.len() < len {
                                    scratch.resize(len, Complex64::default());
                                }
                                ifft_x.process_with_scratch(&mut buffer[..nx], &mut scratch[..len]);
                            });
                            for i in 0..nx {
                                y_slice[[i, k]] = buffer[i];
                            }
                        });
                    }
                });
        }

        if self.use_parallel_axis(self.nx) {
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
                            AXIS_SCRATCH.with(|scratch_cell| {
                                let mut scratch = scratch_cell.borrow_mut();
                                let len = ifft_y.get_inplace_scratch_len();
                                if scratch.len() < len {
                                    scratch.resize(len, Complex64::default());
                                }
                                ifft_y.process_with_scratch(&mut buffer[..ny], &mut scratch[..len]);
                            });
                            for j in 0..ny {
                                x_slice[[j, k]] = buffer[j];
                            }
                        });
                    }
                });
        } else {
            data.axis_iter_mut(Axis(0))
                .into_iter()
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
                            AXIS_SCRATCH.with(|scratch_cell| {
                                let mut scratch = scratch_cell.borrow_mut();
                                let len = ifft_y.get_inplace_scratch_len();
                                if scratch.len() < len {
                                    scratch.resize(len, Complex64::default());
                                }
                                ifft_y.process_with_scratch(&mut buffer[..ny], &mut scratch[..len]);
                            });
                            for j in 0..ny {
                                x_slice[[j, k]] = buffer[j];
                            }
                        });
                    }
                });
        }

        if self.use_parallel_axis(self.nx) {
            data.outer_iter_mut()
                .into_par_iter()
                .for_each(|mut x_slice| {
                    for mut row in x_slice.outer_iter_mut() {
                        AXIS_SCRATCH.with(|scratch_cell| {
                            let mut scratch = scratch_cell.borrow_mut();
                            let len = ifft_z.get_inplace_scratch_len();
                            if scratch.len() < len {
                                scratch.resize(len, Complex64::default());
                            }
                            ifft_z.process_with_scratch(
                                row.as_slice_mut().expect("z row must be contiguous"),
                                &mut scratch[..len],
                            );
                        });
                    }
                });
        } else {
            data.outer_iter_mut().into_iter().for_each(|mut x_slice| {
                for mut row in x_slice.outer_iter_mut() {
                    AXIS_SCRATCH.with(|scratch_cell| {
                        let mut scratch = scratch_cell.borrow_mut();
                        let len = ifft_z.get_inplace_scratch_len();
                        if scratch.len() < len {
                            scratch.resize(len, Complex64::default());
                        }
                        ifft_z.process_with_scratch(
                            row.as_slice_mut().expect("z row must be contiguous"),
                            &mut scratch[..len],
                        );
                    });
                }
            });
        }
    }

    fn inverse_complex_slice_inplace_batched(&self, slice: &mut [Complex64]) {
        let ifft_x = Arc::clone(&self.ifft_x);
        let ifft_y = Arc::clone(&self.ifft_y);
        let ifft_z = Arc::clone(&self.ifft_z);
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let total = nx * ny * nz;

        AXIS_BUF_2D.with(|buf_cell| {
            let mut lanes = buf_cell.borrow_mut();
            if lanes.len() < total {
                lanes.resize(total, Complex64::default());
            }
            for j in 0..ny {
                for k in 0..nz {
                    let lane = (j * nz + k) * nx;
                    for i in 0..nx {
                        lanes[lane + i] = slice[(i * ny + j) * nz + k];
                    }
                }
            }
            AXIS_SCRATCH.with(|scratch_cell| {
                let mut scratch = scratch_cell.borrow_mut();
                let len = ifft_x.get_inplace_scratch_len();
                if scratch.len() < len {
                    scratch.resize(len, Complex64::default());
                }
                ifft_x.process_with_scratch(&mut lanes[..total], &mut scratch[..len]);
            });
            for j in 0..ny {
                for k in 0..nz {
                    let lane = (j * nz + k) * nx;
                    for i in 0..nx {
                        slice[(i * ny + j) * nz + k] = lanes[lane + i];
                    }
                }
            }
        });

        AXIS_BUF_2D.with(|buf_cell| {
            let mut lanes = buf_cell.borrow_mut();
            if lanes.len() < total {
                lanes.resize(total, Complex64::default());
            }
            for x in 0..nx {
                for k in 0..nz {
                    let lane = (x * nz + k) * ny;
                    for j in 0..ny {
                        lanes[lane + j] = slice[(x * ny + j) * nz + k];
                    }
                }
            }
            AXIS_SCRATCH.with(|scratch_cell| {
                let mut scratch = scratch_cell.borrow_mut();
                let len = ifft_y.get_inplace_scratch_len();
                if scratch.len() < len {
                    scratch.resize(len, Complex64::default());
                }
                ifft_y.process_with_scratch(&mut lanes[..total], &mut scratch[..len]);
            });
            for x in 0..nx {
                for k in 0..nz {
                    let lane = (x * nz + k) * ny;
                    for j in 0..ny {
                        slice[(x * ny + j) * nz + k] = lanes[lane + j];
                    }
                }
            }
        });

        AXIS_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let len = ifft_z.get_inplace_scratch_len();
            if scratch.len() < len {
                scratch.resize(len, Complex64::default());
            }
            ifft_z.process_with_scratch(slice, &mut scratch[..len]);
        });
    }

    /// Forward transform of a real 3D field stored as `f32`.
    #[must_use]
    pub(crate) fn forward_f32(&self, input: &Array3<f32>) -> Array3<Complex32> {
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

    /// Forward transform of a real 3D field stored as `f16`.
    #[must_use]
    pub(crate) fn forward_f16(&self, input: &Array3<f16>) -> Array3<Complex32> {
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

    /// Inverse transform of a full `f32`-storage complex spectrum.
    #[must_use]
    pub(crate) fn inverse_f32(&self, input: &Array3<Complex32>) -> Array3<f32> {
        match self.precision {
            PrecisionProfile::LOW_PRECISION_F32 => {
                let mut data = input.clone();
                self.inverse_complex_inplace_f32(&mut data);
                let norm = 1.0 / (self.nx * self.ny * self.nz) as f32;
                data.mapv(|value| value.re * norm)
            }
            _ => {
                let promoted = input.mapv(|value| Complex64::new(value.re as f64, value.im as f64));
                self.inverse_complex_to_real(&promoted)
                    .mapv(|value| value as f32)
            }
        }
    }

    /// Inverse transform of a full `f32`-storage complex spectrum to `f16` storage.
    #[must_use]
    pub(crate) fn inverse_f16(&self, input: &Array3<Complex32>) -> Array3<f16> {
        match self.precision {
            PrecisionProfile::MIXED_PRECISION_F16_F32 => {
                let mut data = input.clone();
                self.inverse_complex_inplace_f32(&mut data);
                let norm = 1.0 / (self.nx * self.ny * self.nz) as f32;
                data.mapv(|value| f16::from_f32(value.re * norm))
            }
            _ => {
                let promoted = input.mapv(|value| Complex64::new(value.re as f64, value.im as f64));
                self.inverse_complex_to_real(&promoted)
                    .mapv(|value| f16::from_f32(value as f32))
            }
        }
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
        if input.is_standard_layout() && output.is_standard_layout() {
            self.forward_real_to_complex_into_full_hermitian(input, output);
            return;
        }
        if self.use_parallel_volume() {
            Zip::from(&mut *output)
                .and(input)
                .par_for_each(|dst, &src| {
                    *dst = Complex64::new(src, 0.0);
                });
        } else {
            Zip::from(&mut *output).and(input).for_each(|dst, &src| {
                *dst = Complex64::new(src, 0.0);
            });
        }
        self.forward_complex_inplace(output);
    }

    fn forward_real_to_complex_into_full_hermitian(
        &self,
        input: &Array3<f64>,
        output: &mut Array3<Complex64>,
    ) {
        let input_slice = input
            .as_slice_memory_order()
            .expect("forward input must be contiguous");
        let output_slice = output
            .as_slice_memory_order_mut()
            .expect("forward output must be contiguous");
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nz_c = self.nz_c;
        let half_total = nx * ny * nz_c;

        let rfft_z = Arc::clone(&self.rfft_z);
        for x in 0..nx {
            for j in 0..ny {
                let real_row_off = (x * ny + j) * nz;
                let complex_row_off = real_row_off;
                RFFT_REAL_BUF.with(|cell| {
                    let mut scratch = cell.borrow_mut();
                    if scratch.len() < nz {
                        scratch.resize(nz, 0.0);
                    }
                    scratch[..nz].copy_from_slice(&input_slice[real_row_off..real_row_off + nz]);
                    rfft_z
                        .process(
                            &mut scratch[..nz],
                            &mut output_slice[complex_row_off..complex_row_off + nz_c],
                        )
                        .expect("forward_real_to_complex_into_full_hermitian: rfft_z failed");
                });
            }
        }

        AXIS_BUF_2D.with(|buf_cell| {
            let mut lanes = buf_cell.borrow_mut();
            if lanes.len() < half_total {
                lanes.resize(half_total, Complex64::default());
            }
            for x in 0..nx {
                for k in 0..nz_c {
                    let lane = (x * nz_c + k) * ny;
                    for j in 0..ny {
                        lanes[lane + j] = output_slice[(x * ny + j) * nz + k];
                    }
                }
            }
            AXIS_SCRATCH.with(|scratch_cell| {
                let mut scratch = scratch_cell.borrow_mut();
                let len = self.fft_y.get_inplace_scratch_len();
                if scratch.len() < len {
                    scratch.resize(len, Complex64::default());
                }
                self.fft_y
                    .process_with_scratch(&mut lanes[..half_total], &mut scratch[..len]);
            });
            for x in 0..nx {
                for k in 0..nz_c {
                    let lane = (x * nz_c + k) * ny;
                    for j in 0..ny {
                        output_slice[(x * ny + j) * nz + k] = lanes[lane + j];
                    }
                }
            }
        });

        AXIS_BUF_2D.with(|buf_cell| {
            let mut lanes = buf_cell.borrow_mut();
            if lanes.len() < half_total {
                lanes.resize(half_total, Complex64::default());
            }
            for j in 0..ny {
                for k in 0..nz_c {
                    let lane = (j * nz_c + k) * nx;
                    for x in 0..nx {
                        lanes[lane + x] = output_slice[(x * ny + j) * nz + k];
                    }
                }
            }
            AXIS_SCRATCH.with(|scratch_cell| {
                let mut scratch = scratch_cell.borrow_mut();
                let len = self.fft_x.get_inplace_scratch_len();
                if scratch.len() < len {
                    scratch.resize(len, Complex64::default());
                }
                self.fft_x
                    .process_with_scratch(&mut lanes[..half_total], &mut scratch[..len]);
            });
            for j in 0..ny {
                for k in 0..nz_c {
                    let lane = (j * nz_c + k) * nx;
                    for x in 0..nx {
                        output_slice[(x * ny + j) * nz + k] = lanes[lane + x];
                    }
                }
            }
        });

        for x in 0..nx {
            let mirror_x = (nx - x) % nx;
            for j in 0..ny {
                let mirror_y = (ny - j) % ny;
                for k in nz_c..nz {
                    let src_k = nz - k;
                    output_slice[(x * ny + j) * nz + k] =
                        output_slice[(mirror_x * ny + mirror_y) * nz + src_k].conj();
                }
            }
        }
    }

    fn forward_r2c_into_batched(&self, input: &Array3<f64>, output: &mut Array3<Complex64>) {
        let input_slice = input
            .as_slice_memory_order()
            .expect("forward_r2c_into_batched: input must be contiguous");
        let output_slice = output
            .as_slice_memory_order_mut()
            .expect("forward_r2c_into_batched: output must be contiguous");
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nz_c = self.nz_c;
        let total = nx * ny * nz_c;

        for x in 0..nx {
            for y in 0..ny {
                let real_row_off = (x * ny + y) * nz;
                let complex_row_off = (x * ny + y) * nz_c;
                RFFT_REAL_BUF.with(|cell| {
                    let mut scratch = cell.borrow_mut();
                    if scratch.len() < nz {
                        scratch.resize(nz, 0.0);
                    }
                    scratch[..nz].copy_from_slice(&input_slice[real_row_off..real_row_off + nz]);
                    self.rfft_z
                        .process(
                            &mut scratch[..nz],
                            &mut output_slice[complex_row_off..complex_row_off + nz_c],
                        )
                        .expect("forward_r2c_into_batched: rfft_z failed");
                });
            }
        }

        AXIS_BUF_2D.with(|buf_cell| {
            let mut lanes = buf_cell.borrow_mut();
            if lanes.len() < total {
                lanes.resize(total, Complex64::default());
            }
            for x in 0..nx {
                for k in 0..nz_c {
                    let lane = (x * nz_c + k) * ny;
                    for y in 0..ny {
                        lanes[lane + y] = output_slice[(x * ny + y) * nz_c + k];
                    }
                }
            }
            AXIS_SCRATCH.with(|scratch_cell| {
                let mut scratch = scratch_cell.borrow_mut();
                let len = self.fft_y.get_inplace_scratch_len();
                if scratch.len() < len {
                    scratch.resize(len, Complex64::default());
                }
                self.fft_y
                    .process_with_scratch(&mut lanes[..total], &mut scratch[..len]);
            });
            for x in 0..nx {
                for k in 0..nz_c {
                    let lane = (x * nz_c + k) * ny;
                    for y in 0..ny {
                        output_slice[(x * ny + y) * nz_c + k] = lanes[lane + y];
                    }
                }
            }
        });

        AXIS_BUF_2D.with(|buf_cell| {
            let mut lanes = buf_cell.borrow_mut();
            if lanes.len() < total {
                lanes.resize(total, Complex64::default());
            }
            for y in 0..ny {
                for k in 0..nz_c {
                    let lane = (y * nz_c + k) * nx;
                    for x in 0..nx {
                        lanes[lane + x] = output_slice[(x * ny + y) * nz_c + k];
                    }
                }
            }
            AXIS_SCRATCH.with(|scratch_cell| {
                let mut scratch = scratch_cell.borrow_mut();
                let len = self.fft_x.get_inplace_scratch_len();
                if scratch.len() < len {
                    scratch.resize(len, Complex64::default());
                }
                self.fft_x
                    .process_with_scratch(&mut lanes[..total], &mut scratch[..len]);
            });
            for y in 0..ny {
                for k in 0..nz_c {
                    let lane = (y * nz_c + k) * nx;
                    for x in 0..nx {
                        output_slice[(x * ny + y) * nz_c + k] = lanes[lane + x];
                    }
                }
            }
        });
    }

    fn try_extract_half_spectrum(&self, input: &Array3<Complex64>, half: &mut [Complex64]) -> bool {
        let input_slice = input
            .as_slice_memory_order()
            .expect("inverse input must be contiguous");
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nz_c = self.nz_c;
        let half_len = nx * ny * nz_c;
        if half.len() < half_len {
            return false;
        }

        for x in 0..nx {
            for y in 0..ny {
                let src_row = (x * ny + y) * nz;
                let dst_row = (x * ny + y) * nz_c;
                half[dst_row..dst_row + nz_c]
                    .copy_from_slice(&input_slice[src_row..src_row + nz_c]);
            }
        }

        for x in 0..nx {
            let mirror_x = (nx - x) % nx;
            for y in 0..ny {
                let mirror_y = (ny - y) % ny;
                let base = (x * ny + y) * nz;
                let mirror_base = (mirror_x * ny + mirror_y) * nz;
                if input_slice[base].im.abs() > HERMITIAN_TOLERANCE {
                    return false;
                }
                if nz % 2 == 0 && input_slice[base + nz_c - 1].im.abs() > HERMITIAN_TOLERANCE {
                    return false;
                }
                for k in nz_c..nz {
                    let expected = input_slice[mirror_base + (nz - k)].conj();
                    if (input_slice[base + k] - expected).norm() > HERMITIAN_TOLERANCE {
                        return false;
                    }
                }
            }
        }

        true
    }

    fn inverse_half_spectrum_to_real_into_slice(
        &self,
        spectrum: &mut [Complex64],
        output: &mut Array3<f64>,
    ) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nz_c = self.nz_c;
        let total = nx * ny * nz_c;

        AXIS_BUF_2D.with(|buf_cell| {
            let mut lanes = buf_cell.borrow_mut();
            if lanes.len() < total {
                lanes.resize(total, Complex64::default());
            }
            for y in 0..ny {
                for k in 0..nz_c {
                    let lane = (y * nz_c + k) * nx;
                    for x in 0..nx {
                        lanes[lane + x] = spectrum[(x * ny + y) * nz_c + k];
                    }
                }
            }
            AXIS_SCRATCH.with(|scratch_cell| {
                let mut scratch = scratch_cell.borrow_mut();
                let len = self.ifft_x.get_inplace_scratch_len();
                if scratch.len() < len {
                    scratch.resize(len, Complex64::default());
                }
                self.ifft_x
                    .process_with_scratch(&mut lanes[..total], &mut scratch[..len]);
            });
            for y in 0..ny {
                for k in 0..nz_c {
                    let lane = (y * nz_c + k) * nx;
                    for x in 0..nx {
                        spectrum[(x * ny + y) * nz_c + k] = lanes[lane + x];
                    }
                }
            }
        });

        AXIS_BUF_2D.with(|buf_cell| {
            let mut lanes = buf_cell.borrow_mut();
            if lanes.len() < total {
                lanes.resize(total, Complex64::default());
            }
            for x in 0..nx {
                for k in 0..nz_c {
                    let lane = (x * nz_c + k) * ny;
                    for y in 0..ny {
                        lanes[lane + y] = spectrum[(x * ny + y) * nz_c + k];
                    }
                }
            }
            AXIS_SCRATCH.with(|scratch_cell| {
                let mut scratch = scratch_cell.borrow_mut();
                let len = self.ifft_y.get_inplace_scratch_len();
                if scratch.len() < len {
                    scratch.resize(len, Complex64::default());
                }
                self.ifft_y
                    .process_with_scratch(&mut lanes[..total], &mut scratch[..len]);
            });
            for x in 0..nx {
                for k in 0..nz_c {
                    let lane = (x * nz_c + k) * ny;
                    for y in 0..ny {
                        spectrum[(x * ny + y) * nz_c + k] = lanes[lane + y];
                    }
                }
            }
        });

        let output_slice = output
            .as_slice_memory_order_mut()
            .expect("inverse half output must be contiguous");
        let norm = 1.0 / (nx * ny * nz) as f64;
        for x in 0..nx {
            for y in 0..ny {
                let spec_row = (x * ny + y) * nz_c;
                let out_row = (x * ny + y) * nz;
                {
                    let row = &mut spectrum[spec_row..spec_row + nz_c];
                    row[0].im = 0.0;
                    if nz % 2 == 0 {
                        row[nz_c - 1].im = 0.0;
                    }
                }
                RFFT_REAL_BUF.with(|cell| {
                    let mut scratch = cell.borrow_mut();
                    if scratch.len() < nz {
                        scratch.resize(nz, 0.0);
                    }
                    self.irfft_z
                        .process(&mut spectrum[spec_row..spec_row + nz_c], &mut scratch[..nz])
                        .expect("inverse_half_spectrum_to_real_into_slice: irfft_z failed");
                    for (dst, src) in output_slice[out_row..out_row + nz]
                        .iter_mut()
                        .zip(scratch[..nz].iter())
                    {
                        *dst = *src * norm;
                    }
                });
            }
        }
    }

    fn forward_complex_inplace_f32(&self, data: &mut Array3<Complex32>) {
        self.check_full_complex_shape(data.dim(), "forward_complex_inplace_f32");
        if !self.use_parallel_volume() {
            self.forward_complex_inplace_batched_f32(data);
            return;
        }
        let fft_z = Arc::clone(&self.fft_z_f32);
        let fft_y = Arc::clone(&self.fft_y_f32);
        let fft_x = Arc::clone(&self.fft_x_f32);
        let ny = self.ny;
        let nx = self.nx;
        let nz = self.nz;

        if self.use_parallel_axis(self.nx) {
            data.outer_iter_mut()
                .into_par_iter()
                .for_each(|mut x_slice| {
                    for mut row in x_slice.outer_iter_mut() {
                        AXIS_SCRATCH_32.with(|scratch_cell| {
                            let mut scratch = scratch_cell.borrow_mut();
                            let len = fft_z.get_inplace_scratch_len();
                            if scratch.len() < len {
                                scratch.resize(len, Complex32::default());
                            }
                            fft_z.process_with_scratch(
                                row.as_slice_mut().expect("z row must be contiguous"),
                                &mut scratch[..len],
                            );
                        });
                    }
                });
        } else {
            data.outer_iter_mut().into_iter().for_each(|mut x_slice| {
                for mut row in x_slice.outer_iter_mut() {
                    AXIS_SCRATCH_32.with(|scratch_cell| {
                        let mut scratch = scratch_cell.borrow_mut();
                        let len = fft_z.get_inplace_scratch_len();
                        if scratch.len() < len {
                            scratch.resize(len, Complex32::default());
                        }
                        fft_z.process_with_scratch(
                            row.as_slice_mut().expect("z row must be contiguous"),
                            &mut scratch[..len],
                        );
                    });
                }
            });
        }

        let x_pass = |mut x_slice: ndarray::ArrayViewMut2<'_, Complex32>| {
            for k in 0..nz {
                AXIS_BUF_32.with(|cell| {
                    let mut buffer = cell.borrow_mut();
                    if buffer.len() < ny {
                        buffer.resize(ny, Complex32::default());
                    }
                    for j in 0..ny {
                        buffer[j] = x_slice[[j, k]];
                    }
                    AXIS_SCRATCH_32.with(|scratch_cell| {
                        let mut scratch = scratch_cell.borrow_mut();
                        let len = fft_y.get_inplace_scratch_len();
                        if scratch.len() < len {
                            scratch.resize(len, Complex32::default());
                        }
                        fft_y.process_with_scratch(&mut buffer[..ny], &mut scratch[..len]);
                    });
                    for j in 0..ny {
                        x_slice[[j, k]] = buffer[j];
                    }
                });
            }
        };
        if self.use_parallel_axis(self.nx) {
            data.axis_iter_mut(Axis(0)).into_par_iter().for_each(x_pass);
        } else {
            data.axis_iter_mut(Axis(0)).into_iter().for_each(x_pass);
        }

        let y_pass = |mut y_slice: ndarray::ArrayViewMut2<'_, Complex32>| {
            for k in 0..nz {
                AXIS_BUF_32.with(|cell| {
                    let mut buffer = cell.borrow_mut();
                    if buffer.len() < nx {
                        buffer.resize(nx, Complex32::default());
                    }
                    for i in 0..nx {
                        buffer[i] = y_slice[[i, k]];
                    }
                    AXIS_SCRATCH_32.with(|scratch_cell| {
                        let mut scratch = scratch_cell.borrow_mut();
                        let len = fft_x.get_inplace_scratch_len();
                        if scratch.len() < len {
                            scratch.resize(len, Complex32::default());
                        }
                        fft_x.process_with_scratch(&mut buffer[..nx], &mut scratch[..len]);
                    });
                    for i in 0..nx {
                        y_slice[[i, k]] = buffer[i];
                    }
                });
            }
        };
        if self.use_parallel_axis(self.ny) {
            data.axis_iter_mut(Axis(1)).into_par_iter().for_each(y_pass);
        } else {
            data.axis_iter_mut(Axis(1)).into_iter().for_each(y_pass);
        }
    }

    fn inverse_complex_inplace_f32(&self, data: &mut Array3<Complex32>) {
        self.check_full_complex_shape(data.dim(), "inverse_complex_inplace_f32");
        if !self.use_parallel_volume() {
            self.inverse_complex_inplace_batched_f32(data);
            return;
        }
        let ifft_x = Arc::clone(&self.ifft_x_f32);
        let ifft_y = Arc::clone(&self.ifft_y_f32);
        let ifft_z = Arc::clone(&self.ifft_z_f32);
        let ny = self.ny;
        let nx = self.nx;
        let nz = self.nz;

        let y_pass = |mut y_slice: ndarray::ArrayViewMut2<'_, Complex32>| {
            for k in 0..nz {
                AXIS_BUF_32.with(|cell| {
                    let mut buffer = cell.borrow_mut();
                    if buffer.len() < nx {
                        buffer.resize(nx, Complex32::default());
                    }
                    for i in 0..nx {
                        buffer[i] = y_slice[[i, k]];
                    }
                    AXIS_SCRATCH_32.with(|scratch_cell| {
                        let mut scratch = scratch_cell.borrow_mut();
                        let len = ifft_x.get_inplace_scratch_len();
                        if scratch.len() < len {
                            scratch.resize(len, Complex32::default());
                        }
                        ifft_x.process_with_scratch(&mut buffer[..nx], &mut scratch[..len]);
                    });
                    for i in 0..nx {
                        y_slice[[i, k]] = buffer[i];
                    }
                });
            }
        };
        if self.use_parallel_axis(self.ny) {
            data.axis_iter_mut(Axis(1)).into_par_iter().for_each(y_pass);
        } else {
            data.axis_iter_mut(Axis(1)).into_iter().for_each(y_pass);
        }

        let x_pass = |mut x_slice: ndarray::ArrayViewMut2<'_, Complex32>| {
            for k in 0..nz {
                AXIS_BUF_32.with(|cell| {
                    let mut buffer = cell.borrow_mut();
                    if buffer.len() < ny {
                        buffer.resize(ny, Complex32::default());
                    }
                    for j in 0..ny {
                        buffer[j] = x_slice[[j, k]];
                    }
                    AXIS_SCRATCH_32.with(|scratch_cell| {
                        let mut scratch = scratch_cell.borrow_mut();
                        let len = ifft_y.get_inplace_scratch_len();
                        if scratch.len() < len {
                            scratch.resize(len, Complex32::default());
                        }
                        ifft_y.process_with_scratch(&mut buffer[..ny], &mut scratch[..len]);
                    });
                    for j in 0..ny {
                        x_slice[[j, k]] = buffer[j];
                    }
                });
            }
        };
        if self.use_parallel_axis(self.nx) {
            data.axis_iter_mut(Axis(0)).into_par_iter().for_each(x_pass);
        } else {
            data.axis_iter_mut(Axis(0)).into_iter().for_each(x_pass);
        }

        if self.use_parallel_axis(self.nx) {
            data.outer_iter_mut()
                .into_par_iter()
                .for_each(|mut x_slice| {
                    for mut row in x_slice.outer_iter_mut() {
                        AXIS_SCRATCH_32.with(|scratch_cell| {
                            let mut scratch = scratch_cell.borrow_mut();
                            let len = ifft_z.get_inplace_scratch_len();
                            if scratch.len() < len {
                                scratch.resize(len, Complex32::default());
                            }
                            ifft_z.process_with_scratch(
                                row.as_slice_mut().expect("z row must be contiguous"),
                                &mut scratch[..len],
                            );
                        });
                    }
                });
        } else {
            data.outer_iter_mut().into_iter().for_each(|mut x_slice| {
                for mut row in x_slice.outer_iter_mut() {
                    AXIS_SCRATCH_32.with(|scratch_cell| {
                        let mut scratch = scratch_cell.borrow_mut();
                        let len = ifft_z.get_inplace_scratch_len();
                        if scratch.len() < len {
                            scratch.resize(len, Complex32::default());
                        }
                        ifft_z.process_with_scratch(
                            row.as_slice_mut().expect("z row must be contiguous"),
                            &mut scratch[..len],
                        );
                    });
                }
            });
        }
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

    fn forward_complex_inplace_batched(&self, data: &mut Array3<Complex64>) {
        let fft_z = Arc::clone(&self.fft_z);
        let fft_y = Arc::clone(&self.fft_y);
        let fft_x = Arc::clone(&self.fft_x);
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nyz = ny * nz;
        let total = nx * nyz;
        let slice = data
            .as_slice_memory_order_mut()
            .expect("forward batched data must be contiguous");

        AXIS_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let len = fft_z.get_inplace_scratch_len();
            if scratch.len() < len {
                scratch.resize(len, Complex64::default());
            }
            fft_z.process_with_scratch(slice, &mut scratch[..len]);
        });

        AXIS_BUF_2D.with(|buf_cell| {
            let mut lanes = buf_cell.borrow_mut();
            if lanes.len() < total {
                lanes.resize(total, Complex64::default());
            }
            for x in 0..nx {
                for k in 0..nz {
                    let lane = (x * nz + k) * ny;
                    for j in 0..ny {
                        lanes[lane + j] = slice[(x * ny + j) * nz + k];
                    }
                }
            }
            AXIS_SCRATCH.with(|scratch_cell| {
                let mut scratch = scratch_cell.borrow_mut();
                let len = fft_y.get_inplace_scratch_len();
                if scratch.len() < len {
                    scratch.resize(len, Complex64::default());
                }
                fft_y.process_with_scratch(&mut lanes[..total], &mut scratch[..len]);
            });
            for x in 0..nx {
                for k in 0..nz {
                    let lane = (x * nz + k) * ny;
                    for j in 0..ny {
                        slice[(x * ny + j) * nz + k] = lanes[lane + j];
                    }
                }
            }
        });

        AXIS_BUF_2D.with(|buf_cell| {
            let mut lanes = buf_cell.borrow_mut();
            if lanes.len() < total {
                lanes.resize(total, Complex64::default());
            }
            for j in 0..ny {
                for k in 0..nz {
                    let lane = (j * nz + k) * nx;
                    for i in 0..nx {
                        lanes[lane + i] = slice[(i * ny + j) * nz + k];
                    }
                }
            }
            AXIS_SCRATCH.with(|scratch_cell| {
                let mut scratch = scratch_cell.borrow_mut();
                let len = fft_x.get_inplace_scratch_len();
                if scratch.len() < len {
                    scratch.resize(len, Complex64::default());
                }
                fft_x.process_with_scratch(&mut lanes[..total], &mut scratch[..len]);
            });
            for j in 0..ny {
                for k in 0..nz {
                    let lane = (j * nz + k) * nx;
                    for i in 0..nx {
                        slice[(i * ny + j) * nz + k] = lanes[lane + i];
                    }
                }
            }
        });
    }

    fn forward_complex_inplace_batched_f32(&self, data: &mut Array3<Complex32>) {
        let fft_z = Arc::clone(&self.fft_z_f32);
        let fft_y = Arc::clone(&self.fft_y_f32);
        let fft_x = Arc::clone(&self.fft_x_f32);
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nyz = ny * nz;
        let total = nx * nyz;
        let slice = data
            .as_slice_memory_order_mut()
            .expect("forward batched f32 data must be contiguous");

        AXIS_SCRATCH_32.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let len = fft_z.get_inplace_scratch_len();
            if scratch.len() < len {
                scratch.resize(len, Complex32::default());
            }
            fft_z.process_with_scratch(slice, &mut scratch[..len]);
        });

        AXIS_BUF_2D_32.with(|buf_cell| {
            let mut lanes = buf_cell.borrow_mut();
            if lanes.len() < total {
                lanes.resize(total, Complex32::default());
            }
            for x in 0..nx {
                for k in 0..nz {
                    let lane = (x * nz + k) * ny;
                    for j in 0..ny {
                        lanes[lane + j] = slice[(x * ny + j) * nz + k];
                    }
                }
            }
            AXIS_SCRATCH_32.with(|scratch_cell| {
                let mut scratch = scratch_cell.borrow_mut();
                let len = fft_y.get_inplace_scratch_len();
                if scratch.len() < len {
                    scratch.resize(len, Complex32::default());
                }
                fft_y.process_with_scratch(&mut lanes[..total], &mut scratch[..len]);
            });
            for x in 0..nx {
                for k in 0..nz {
                    let lane = (x * nz + k) * ny;
                    for j in 0..ny {
                        slice[(x * ny + j) * nz + k] = lanes[lane + j];
                    }
                }
            }
        });

        AXIS_BUF_2D_32.with(|buf_cell| {
            let mut lanes = buf_cell.borrow_mut();
            if lanes.len() < total {
                lanes.resize(total, Complex32::default());
            }
            for j in 0..ny {
                for k in 0..nz {
                    let lane = (j * nz + k) * nx;
                    for i in 0..nx {
                        lanes[lane + i] = slice[(i * ny + j) * nz + k];
                    }
                }
            }
            AXIS_SCRATCH_32.with(|scratch_cell| {
                let mut scratch = scratch_cell.borrow_mut();
                let len = fft_x.get_inplace_scratch_len();
                if scratch.len() < len {
                    scratch.resize(len, Complex32::default());
                }
                fft_x.process_with_scratch(&mut lanes[..total], &mut scratch[..len]);
            });
            for j in 0..ny {
                for k in 0..nz {
                    let lane = (j * nz + k) * nx;
                    for i in 0..nx {
                        slice[(i * ny + j) * nz + k] = lanes[lane + i];
                    }
                }
            }
        });
    }

    fn inverse_complex_inplace_batched_f32(&self, data: &mut Array3<Complex32>) {
        let ifft_x = Arc::clone(&self.ifft_x_f32);
        let ifft_y = Arc::clone(&self.ifft_y_f32);
        let ifft_z = Arc::clone(&self.ifft_z_f32);
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nyz = ny * nz;
        let total = nx * nyz;
        let slice = data
            .as_slice_memory_order_mut()
            .expect("inverse batched f32 data must be contiguous");

        AXIS_BUF_2D_32.with(|buf_cell| {
            let mut lanes = buf_cell.borrow_mut();
            if lanes.len() < total {
                lanes.resize(total, Complex32::default());
            }
            for j in 0..ny {
                for k in 0..nz {
                    let lane = (j * nz + k) * nx;
                    for i in 0..nx {
                        lanes[lane + i] = slice[(i * ny + j) * nz + k];
                    }
                }
            }
            AXIS_SCRATCH_32.with(|scratch_cell| {
                let mut scratch = scratch_cell.borrow_mut();
                let len = ifft_x.get_inplace_scratch_len();
                if scratch.len() < len {
                    scratch.resize(len, Complex32::default());
                }
                ifft_x.process_with_scratch(&mut lanes[..total], &mut scratch[..len]);
            });
            for j in 0..ny {
                for k in 0..nz {
                    let lane = (j * nz + k) * nx;
                    for i in 0..nx {
                        slice[(i * ny + j) * nz + k] = lanes[lane + i];
                    }
                }
            }
        });

        AXIS_BUF_2D_32.with(|buf_cell| {
            let mut lanes = buf_cell.borrow_mut();
            if lanes.len() < total {
                lanes.resize(total, Complex32::default());
            }
            for x in 0..nx {
                for k in 0..nz {
                    let lane = (x * nz + k) * ny;
                    for j in 0..ny {
                        lanes[lane + j] = slice[(x * ny + j) * nz + k];
                    }
                }
            }
            AXIS_SCRATCH_32.with(|scratch_cell| {
                let mut scratch = scratch_cell.borrow_mut();
                let len = ifft_y.get_inplace_scratch_len();
                if scratch.len() < len {
                    scratch.resize(len, Complex32::default());
                }
                ifft_y.process_with_scratch(&mut lanes[..total], &mut scratch[..len]);
            });
            for x in 0..nx {
                for k in 0..nz {
                    let lane = (x * nz + k) * ny;
                    for j in 0..ny {
                        slice[(x * ny + j) * nz + k] = lanes[lane + j];
                    }
                }
            }
        });

        AXIS_SCRATCH_32.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let len = ifft_z.get_inplace_scratch_len();
            if scratch.len() < len {
                scratch.resize(len, Complex32::default());
            }
            ifft_z.process_with_scratch(slice, &mut scratch[..len]);
        });
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

    #[test]
    fn low_precision_roundtrip_stays_within_expected_envelope() {
        let plan = FftPlan3D::with_precision(6, 5, 4, PrecisionProfile::LOW_PRECISION_F32);
        let field = Array3::from_shape_fn((6, 5, 4), |(i, j, k)| {
            ((i as f32) * 0.31 + (j as f32) * 0.27 - (k as f32) * 0.19).sin()
        });
        let recovered: Array3<f32> = plan.inverse_typed(&plan.forward_typed(&field));
        let max_err = field
            .iter()
            .zip(recovered.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs() as f64)
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-5, "low precision max err = {max_err:.3e}");
    }

    #[test]
    fn mixed_precision_improves_on_low_precision_reference() {
        let field_f32 = Array3::from_shape_fn((6, 5, 4), |(i, j, k)| {
            ((i as f32) * 0.31 + (j as f32) * 0.27 - (k as f32) * 0.19).sin()
        });
        let field = field_f32.mapv(f16::from_f32);
        let low_plan = FftPlan3D::with_precision(6, 5, 4, PrecisionProfile::LOW_PRECISION_F32);
        let mixed_plan =
            FftPlan3D::with_precision(6, 5, 4, PrecisionProfile::MIXED_PRECISION_F16_F32);
        let low_recovered: Array3<f32> =
            low_plan.inverse_typed(&low_plan.forward_typed(&field_f32));
        let mixed_recovered: Array3<f16> =
            mixed_plan.inverse_typed(&mixed_plan.forward_typed(&field));
        let low_err = field_f32
            .iter()
            .zip(low_recovered.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs() as f64)
            .fold(0.0_f64, f64::max);
        let mixed_err = field
            .iter()
            .zip(mixed_recovered.iter())
            .map(|(lhs, rhs)| (lhs.to_f32() - rhs.to_f32()).abs() as f64)
            .fold(0.0_f64, f64::max);
        assert!(low_err < 1e-5, "low {low_err:.3e} should stay bounded");
        assert!(
            mixed_err < 5e-3,
            "mixed {mixed_err:.3e} should stay bounded"
        );
    }

    proptest! {
        #[test]
        fn one_dimensional_roundtrip_holds_for_random_lengths_and_signals(
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
        fn three_dimensional_roundtrip_holds_for_small_random_shapes(
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
