//! 3D FFT plan.
//!
//! Apollo-owned 3D FFT implementation based on separable FFT passes.
//!
//! The plan keeps the public API stable while removing production dependence on
//! external FFT engines. Forward and inverse transforms are executed by
//! applying the auto-selected 1D FFT kernel along each axis in sequence. The
//! inverse path uses FFTW-compatible normalization by dividing by the total
//! volume.
//!
//! # Mathematical contract
//!
//! For a complex field `x ∈ ℂ^{n_x × n_y × n_z}`, the forward transform is the
//! separable 3D DFT
//!
//! `X_{k_x,k_y,k_z} = Σ_x Σ_y Σ_z x_{x,y,z} · exp(-2πi (k_x x / n_x + k_y y / n_y + k_z z / n_z))`
//!
//! and the inverse transform is
//!
//! `x_{x,y,z} = (1 / (n_x n_y n_z)) Σ_kx Σ_ky Σ_kz X_{k_x,k_y,k_z}
//! · exp(2πi (k_x x / n_x + k_y y / n_y + k_z z / n_z))`.
//!
//! Because the transform is separable, the implementation applies the 1D FFT
//! kernel independently along each axis. This preserves linearity and the
//! expected roundtrip identity in exact arithmetic.
//!
//! # Complexity
//!
//! Let `C(n)` be the selected 1D FFT cost. The plan costs
//! `O(n_y n_z C(n_x) + n_x n_z C(n_y) + n_x n_y C(n_z))`, with
//! `C(n) = O(n log n)` for both radix-2 and Bluestein plan paths. Contiguous
//! innermost-axis passes mutate depth chunks in place, while non-contiguous
//! passes gather lanes into scratch buffers before scattering them back.
//!
//! # Failure modes
//!
//! - zero dimensions are rejected by `Shape3D::new`
//! - caller-supplied buffers must match the plan dimensions
//! - non-contiguous ndarray buffers panic when a contiguous slice is required

use crate::application::execution::kernel::{
    fft_forward_32, fft_forward_64, fft_inverse_32, fft_inverse_64,
};
use crate::application::execution::kernel::radix2::{
    build_forward_twiddle_table_32, build_forward_twiddle_table_64,
    build_inverse_twiddle_table_32, build_inverse_twiddle_table_64,
    forward_inplace_32_with_twiddles, forward_inplace_64_with_twiddles,
    inverse_inplace_32_with_twiddles, inverse_inplace_64_with_twiddles,
};
use crate::application::execution::plan::fft::real_storage::RealFftData;
use crate::domain::metadata::precision::PrecisionProfile;
use crate::domain::metadata::shape::Shape3D;
use half::f16;
use ndarray::{Array3, Axis, Zip};
use num_complex::Complex32;

/// Use rayon parallel iteration when total elements exceed this threshold.
/// Below the threshold, sequential iteration avoids rayon task-spawn overhead
/// that dominates for small volumes (e.g. 8³ = 512 elements).
const RAYON_THRESHOLD: usize = 32768;

/// Tile size for cache-blocked gather/scatter in axis-1 and axis-0 passes.
///
/// For each i-slice in axis-1, the gather is a [ny][nz] → [nz][ny] transpose.
/// A 32×32 tile of Complex64 = 16 KB, fitting in L1 (32–48 KB). The same
/// value works for axis-0 ([j,k]-plane transposes). Corresponds to the same
/// TRANSPOSE_TILE used in dimension_2d.rs.
const GATHER_TILE: usize = 32;
use num_complex::Complex64;
use rayon::prelude::*;

/// Reusable 3D FFT plan.
///
/// # Precomputed twiddle tables and scratch buffers
///
/// For power-of-two axis lengths, the plan precomputes contiguous per-stage
/// twiddle tables at construction time (total N-1 entries per axis). All
/// butterfly passes use these tables, eliminating the per-lane twiddle-table
/// allocation that naïve dispatch via `fft_forward_64` would otherwise perform.
///
/// Axis-1 (y) and axis-0 (x) passes require gathering non-contiguous lanes
/// into a temporary buffer before the FFT and scattering results back. This
/// buffer is preallocated at plan construction time (`scratch_y_64`,
/// `scratch_x_64`) and reused via a Mutex, avoiding a fresh `Vec<Complex64>`
/// allocation of size `nx * ny * nz` on every transform call.
///
/// For non-power-of-two axis lengths, the plan falls back to the
/// auto-selecting `fft_forward_64` / `fft_inverse_64` dispatch which handles
/// Bluestein with per-call scratch allocation.
pub struct FftPlan3D {
    nx: usize,
    ny: usize,
    nz: usize,
    nz_c: usize,
    precision: PrecisionProfile,
    // --- precomputed twiddle tables (Some iff axis length is power-of-two > 1) ---
    twiddle_z_fwd_64: Option<Vec<Complex64>>,
    twiddle_z_inv_64: Option<Vec<Complex64>>,
    twiddle_y_fwd_64: Option<Vec<Complex64>>,
    twiddle_y_inv_64: Option<Vec<Complex64>>,
    twiddle_x_fwd_64: Option<Vec<Complex64>>,
    twiddle_x_inv_64: Option<Vec<Complex64>>,
    twiddle_z_fwd_32: Option<Vec<Complex32>>,
    twiddle_z_inv_32: Option<Vec<Complex32>>,
    twiddle_y_fwd_32: Option<Vec<Complex32>>,
    twiddle_y_inv_32: Option<Vec<Complex32>>,
    twiddle_x_fwd_32: Option<Vec<Complex32>>,
    twiddle_x_inv_32: Option<Vec<Complex32>>,
    // --- preallocated scratch for y and x gather-FFT-scatter passes ---
    scratch_y_64: std::sync::Mutex<Vec<Complex64>>,
    scratch_x_64: std::sync::Mutex<Vec<Complex64>>,
    scratch_y_32: std::sync::Mutex<Vec<Complex32>>,
    scratch_x_32: std::sync::Mutex<Vec<Complex32>>,
}

impl std::fmt::Debug for FftPlan3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan3D")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .field("nz_c", &self.nz_c)
            .field("precision", &self.precision)
            .finish()
    }
}

impl FftPlan3D {
    /// Create a new 3D plan.
    #[must_use]
    pub fn new(shape: Shape3D) -> Self {
        Self::with_precision(shape, PrecisionProfile::HIGH_ACCURACY_F64)
    }

    /// Create a new 3D plan with an explicit precision profile.
    #[must_use]
    pub fn with_precision(shape: Shape3D, precision: PrecisionProfile) -> Self {
        let (nx, ny, nz) = (shape.nx, shape.ny, shape.nz);
        let vol = nx * ny * nz;
        let make64 = |n: usize, forward: bool| -> Option<Vec<Complex64>> {
            if n > 1 && n.is_power_of_two() {
                Some(if forward {
                    build_forward_twiddle_table_64(n)
                } else {
                    build_inverse_twiddle_table_64(n)
                })
            } else {
                None
            }
        };
        let make32 = |n: usize, forward: bool| -> Option<Vec<Complex32>> {
            if n > 1 && n.is_power_of_two() {
                Some(if forward {
                    build_forward_twiddle_table_32(n)
                } else {
                    build_inverse_twiddle_table_32(n)
                })
            } else {
                None
            }
        };
        Self {
            nx,
            ny,
            nz,
            nz_c: nz / 2 + 1,
            precision,
            twiddle_z_fwd_64: make64(nz, true),
            twiddle_z_inv_64: make64(nz, false),
            twiddle_y_fwd_64: make64(ny, true),
            twiddle_y_inv_64: make64(ny, false),
            twiddle_x_fwd_64: make64(nx, true),
            twiddle_x_inv_64: make64(nx, false),
            twiddle_z_fwd_32: make32(nz, true),
            twiddle_z_inv_32: make32(nz, false),
            twiddle_y_fwd_32: make32(ny, true),
            twiddle_y_inv_32: make32(ny, false),
            twiddle_x_fwd_32: make32(nx, true),
            twiddle_x_inv_32: make32(nx, false),
            scratch_y_64: std::sync::Mutex::new(vec![Complex64::default(); vol]),
            scratch_x_64: std::sync::Mutex::new(vec![Complex64::default(); vol]),
            scratch_y_32: std::sync::Mutex::new(vec![Complex32::default(); vol]),
            scratch_x_32: std::sync::Mutex::new(vec![Complex32::default(); vol]),
        }
    }

    /// Return the precision profile used by this plan.
    #[must_use]
    pub fn precision_profile(&self) -> PrecisionProfile {
        self.precision
    }

    /// Return the bookkeeping value `nz / 2 + 1`.
    ///
    /// This value is provided for API compatibility with callers that track
    /// half-spectrum layout metadata. No R2C half-spectrum transform is
    /// implemented: every forward and inverse transform always operates on the
    /// full `(nx, ny, nz)` complex array internally. This accessor does not
    /// describe an actual reduction in the Z dimension.
    #[must_use]
    pub fn nz_c(&self) -> usize {
        self.nz_c
    }

    /// Alias for `nz_c()`.
    ///
    /// Returns `nz / 2 + 1` for bookkeeping only. The full `nz`-length Z
    /// spectrum is always computed; no half-spectrum reduction is applied.
    #[must_use]
    pub fn nz_complex(&self) -> usize {
        self.nz_c()
    }

    /// Return the full real-domain shape owned by this plan.
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// Return the validated shape owned by this plan.
    #[must_use]
    pub fn shape(&self) -> Shape3D {
        Shape3D {
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
        }
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

    /// Forward transform of a real field into caller-owned typed spectrum storage.
    pub fn forward_typed_into<T: RealFftData>(
        &self,
        input: &Array3<T>,
        output: &mut Array3<T::Spectrum>,
    ) {
        T::forward_3d_into(self, input, output);
    }

    /// Inverse transform into caller-owned typed real storage and scratch spectrum.
    pub fn inverse_typed_into<T: RealFftData>(
        &self,
        input: &Array3<T::Spectrum>,
        output: &mut Array3<T>,
        scratch: &mut Array3<T::Spectrum>,
    ) {
        T::inverse_3d_into(self, input, output, scratch);
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
        scratch.assign(input);
        self.inverse_complex_inplace(scratch);
        Zip::from(output).and(scratch).for_each(|out, value| {
            *out = value.re;
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

    /// Forward transform of a complex field in-place.
    pub fn forward_complex_inplace(&self, data: &mut Array3<Complex64>) {
        self.check_full_complex_shape(data.dim(), "forward input");
        self.forward_complex_axis_pass(data);
    }

    /// Inverse transform of a complex field in-place with FFTW-compatible normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array3<Complex64>) {
        self.check_full_complex_shape(data.dim(), "inverse input");
        self.inverse_complex_axis_pass(data);
    }

    /// Forward transform of a real field stored as `f32`.
    #[must_use]
    pub(crate) fn forward_f32(&self, input: &Array3<f32>) -> Array3<Complex32> {
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            let mut output = Array3::<Complex32>::zeros((self.nx, self.ny, self.nz));
            self.forward_real_to_complex_f32_into(input, &mut output);
            output
        } else {
            let promoted = input.mapv(f64::from);
            self.forward_real_to_complex(&promoted)
                .mapv(|value| Complex32::new(value.re as f32, value.im as f32))
        }
    }

    /// Inverse transform of an `f32`-storage complex spectrum.
    #[must_use]
    pub(crate) fn inverse_f32(&self, input: &Array3<Complex32>) -> Array3<f32> {
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            let mut output = Array3::<Complex32>::zeros((self.nx, self.ny, self.nz));
            output.assign(input);
            self.inverse_complex_inplace_f32(&mut output);
            output.mapv(|value| value.re)
        } else {
            let promoted =
                input.mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im)));
            self.inverse_complex_to_real(&promoted)
                .mapv(|value| value as f32)
        }
    }

    /// Forward transform of a real `f32` field into caller-owned spectrum storage.
    pub(crate) fn forward_f32_into(&self, input: &Array3<f32>, output: &mut Array3<Complex32>) {
        self.check_real_shape(input.dim(), "forward input");
        self.check_full_complex_shape(output.dim(), "forward output");
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            self.forward_real_to_complex_f32_into(input, output);
        } else {
            output.assign(
                &self
                    .forward_real_to_complex(&input.mapv(f64::from))
                    .mapv(|value| Complex32::new(value.re as f32, value.im as f32)),
            );
        }
    }

    /// Inverse transform of an `f32` spectrum into caller-owned real storage.
    pub(crate) fn inverse_f32_into(
        &self,
        input: &Array3<Complex32>,
        output: &mut Array3<f32>,
        scratch: &mut Array3<Complex32>,
    ) {
        self.check_full_complex_shape(input.dim(), "inverse input");
        self.check_real_shape(output.dim(), "inverse output");
        self.check_full_complex_shape(scratch.dim(), "inverse scratch");
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            scratch.assign(input);
            self.inverse_complex_inplace_f32(scratch);
            Zip::from(output).and(scratch).for_each(|out, value| {
                *out = value.re;
            });
        } else {
            output.assign(
                &self
                    .inverse_complex_to_real(
                        &input
                            .mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im))),
                    )
                    .mapv(|value| value as f32),
            );
        }
    }

    /// Forward transform of a real field stored as `f16`.
    #[must_use]
    pub(crate) fn forward_f16(&self, input: &Array3<f16>) -> Array3<Complex32> {
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
    pub(crate) fn inverse_f16(&self, input: &Array3<Complex32>) -> Array3<f16> {
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

    /// Forward transform of a real `f16` field into caller-owned `f32` spectrum storage.
    pub(crate) fn forward_f16_into(&self, input: &Array3<f16>, output: &mut Array3<Complex32>) {
        self.check_real_shape(input.dim(), "forward input");
        self.check_full_complex_shape(output.dim(), "forward output");
        if self.precision == PrecisionProfile::MIXED_PRECISION_F16_F32 {
            Zip::from(&mut *output).and(input).for_each(|out, &value| {
                *out = Complex32::new(value.to_f32(), 0.0);
            });
            self.forward_complex_inplace_f32(output);
        } else {
            output.assign(
                &self
                    .forward_real_to_complex(&input.mapv(|value| f64::from(value.to_f32())))
                    .mapv(|value| Complex32::new(value.re as f32, value.im as f32)),
            );
        }
    }

    /// Inverse transform of a `Complex32` spectrum into caller-owned `f16` storage.
    pub(crate) fn inverse_f16_into(
        &self,
        input: &Array3<Complex32>,
        output: &mut Array3<f16>,
        scratch: &mut Array3<Complex32>,
    ) {
        self.check_full_complex_shape(input.dim(), "inverse input");
        self.check_real_shape(output.dim(), "inverse output");
        self.check_full_complex_shape(scratch.dim(), "inverse scratch");
        if self.precision == PrecisionProfile::MIXED_PRECISION_F16_F32 {
            scratch.assign(input);
            self.inverse_complex_inplace_f32(scratch);
            Zip::from(output).and(scratch).for_each(|out, value| {
                *out = f16::from_f32(value.re);
            });
        } else {
            output.assign(
                &self
                    .inverse_complex_to_real(
                        &input
                            .mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im))),
                    )
                    .mapv(|value| f16::from_f32(value as f32)),
            );
        }
    }

    fn forward_complex_axis_pass(&self, data: &mut Array3<Complex64>) {
        self.axis_pass_forward(data, Axis(2));
        self.axis_pass_forward(data, Axis(1));
        self.axis_pass_forward(data, Axis(0));
    }

    fn inverse_complex_axis_pass(&self, data: &mut Array3<Complex64>) {
        self.axis_pass_inverse(data, Axis(0));
        self.axis_pass_inverse(data, Axis(1));
        self.axis_pass_inverse(data, Axis(2));
    }

    fn forward_real_to_complex_into_full(
        &self,
        input: &Array3<f64>,
        output: &mut Array3<Complex64>,
    ) {
        self.check_real_shape(input.dim(), "forward input");
        self.check_full_complex_shape(output.dim(), "forward output");
        Zip::from(output.view_mut())
            .and(input.view())
            .for_each(|out, &value| *out = Complex64::new(value, 0.0));
        self.forward_complex_inplace(output);
    }

    fn forward_real_to_complex_f32_into(
        &self,
        input: &Array3<f32>,
        output: &mut Array3<Complex32>,
    ) {
        self.check_real_shape(input.dim(), "forward input");
        self.check_full_complex_shape(output.dim(), "forward output");
        Zip::from(output.view_mut())
            .and(input.view())
            .for_each(|out, &value| *out = Complex32::new(value, 0.0));
        self.forward_complex_inplace_f32(output);
    }

    fn forward_complex_inplace_f32(&self, data: &mut Array3<Complex32>) {
        self.check_full_complex_shape(data.dim(), "forward input");
        self.axis_pass_forward_f32(data, Axis(2));
        self.axis_pass_forward_f32(data, Axis(1));
        self.axis_pass_forward_f32(data, Axis(0));
    }

    fn inverse_complex_inplace_f32(&self, data: &mut Array3<Complex32>) {
        self.check_full_complex_shape(data.dim(), "inverse input");
        self.axis_pass_inverse_f32(data, Axis(0));
        self.axis_pass_inverse_f32(data, Axis(1));
        self.axis_pass_inverse_f32(data, Axis(2));
    }

    fn axis_pass_forward(&self, data: &mut Array3<Complex64>, axis: Axis) {
        self.axis_pass_complex(data, axis, true);
    }

    fn axis_pass_inverse(&self, data: &mut Array3<Complex64>, axis: Axis) {
        self.axis_pass_complex(data, axis, false);
    }

    fn axis_pass_forward_f32(&self, data: &mut Array3<Complex32>, axis: Axis) {
        self.axis_pass_complex_f32(data, axis, true);
    }

    fn axis_pass_inverse_f32(&self, data: &mut Array3<Complex32>, axis: Axis) {
        self.axis_pass_complex_f32(data, axis, false);
    }

    fn axis_pass_complex(&self, data: &mut Array3<Complex64>, axis: Axis, forward: bool) {
        if data.len_of(axis) <= 1 {
            return;
        }
        if axis.index() == 2 {
            self.axis2_pass_complex(data, forward);
            return;
        }
        if axis.index() == 1 {
            self.axis1_pass_complex(data, forward);
            return;
        }
        if axis.index() == 0 {
            self.axis0_pass_complex(data, forward);
        }
    }

    fn axis1_pass_complex(&self, data: &mut Array3<Complex64>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D complex data must be contiguous");
        let mut scratch = self.scratch_y_64.lock().expect("scratch_y_64 mutex poisoned");
        // Cache-blocked gather: data[i,j,k] (row-major) → scratch[i,k,j].
        // j-outer / k-inner: reads data_slice[i][j][k] sequentially in k (stride 1),
        // writes scratch[i][k][j] with stride ny. Strided stores buffer in the store
        // queue; strided loads stall the pipeline.
        for i in 0..self.nx {
            for j_t in (0..self.ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(self.ny);
                for k_t in (0..self.nz).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(self.nz);
                    for j in j_t..j_end {
                        let src = (i * self.ny + j) * self.nz;
                        for k in k_t..k_end {
                            scratch[(i * self.nz + k) * self.ny + j] = data_slice[src + k];
                        }
                    }
                }
            }
        }
        let lane_fn_64 = |lane: &mut [Complex64]| match (forward, &self.twiddle_y_fwd_64, &self.twiddle_y_inv_64) {
            (true,  Some(tw), _) => forward_inplace_64_with_twiddles(lane, tw.as_slice()),
            (false, _, Some(tw)) => inverse_inplace_64_with_twiddles(lane, tw.as_slice()),
            _ => if forward { fft_forward_64(lane) } else { fft_inverse_64(lane) },
        };
        if scratch.len() > RAYON_THRESHOLD {
            scratch.par_chunks_mut(self.ny).for_each(lane_fn_64);
        } else {
            scratch.chunks_mut(self.ny).for_each(lane_fn_64);
        }
        // Cache-blocked scatter: scratch[i,k,j] → data[i,j,k].
        // j-outer / k-inner: writes data_slice[i][j][k] sequentially in k (stride 1),
        // reads scratch[i][k][j] with stride ny.
        for i in 0..self.nx {
            for j_t in (0..self.ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(self.ny);
                for k_t in (0..self.nz).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(self.nz);
                    for j in j_t..j_end {
                        let dst = (i * self.ny + j) * self.nz;
                        for k in k_t..k_end {
                            data_slice[dst + k] = scratch[(i * self.nz + k) * self.ny + j];
                        }
                    }
                }
            }
        }
    }

    fn axis0_pass_complex(&self, data: &mut Array3<Complex64>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D complex data must be contiguous");
        let mut scratch = self.scratch_x_64.lock().expect("scratch_x_64 mutex poisoned");
        // Cache-blocked gather: data[i,j,k] → scratch[j,k,i].
        // i-outer loop: for each i reads data_slice[i][j][k] sequentially in k (stride 1),
        // writes scratch[j][k][i] with stride nx. Strided stores buffer in the store
        // queue; strided loads (stride ny*nz) stall the pipeline catastrophically.
        for i in 0..self.nx {
            let src_base = i * self.ny * self.nz;
            for j_t in (0..self.ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(self.ny);
                for k_t in (0..self.nz).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(self.nz);
                    for j in j_t..j_end {
                        let src = src_base + j * self.nz;
                        for k in k_t..k_end {
                            scratch[(j * self.nz + k) * self.nx + i] = data_slice[src + k];
                        }
                    }
                }
            }
        }
        let lane_fn_64 = |lane: &mut [Complex64]| match (forward, &self.twiddle_x_fwd_64, &self.twiddle_x_inv_64) {
            (true,  Some(tw), _) => forward_inplace_64_with_twiddles(lane, tw.as_slice()),
            (false, _, Some(tw)) => inverse_inplace_64_with_twiddles(lane, tw.as_slice()),
            _ => if forward { fft_forward_64(lane) } else { fft_inverse_64(lane) },
        };
        if scratch.len() > RAYON_THRESHOLD {
            scratch.par_chunks_mut(self.nx).for_each(lane_fn_64);
        } else {
            scratch.chunks_mut(self.nx).for_each(lane_fn_64);
        }
        // Cache-blocked scatter: scratch[j,k,i] → data[i,j,k].
        // i-outer loop: writes data_slice[i][j][k] sequentially in k (stride 1),
        // reads scratch[j][k][i] with stride nx.
        for i in 0..self.nx {
            let dst_base = i * self.ny * self.nz;
            for j_t in (0..self.ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(self.ny);
                for k_t in (0..self.nz).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(self.nz);
                    for j in j_t..j_end {
                        let dst = dst_base + j * self.nz;
                        for k in k_t..k_end {
                            data_slice[dst + k] = scratch[(j * self.nz + k) * self.nx + i];
                        }
                    }
                }
            }
        }
    }

    fn axis2_pass_complex(&self, data: &mut Array3<Complex64>, forward: bool) {
        if self.nz <= 1 {
            return;
        }
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D complex data must be contiguous");
        let lane_fn_64 = |lane: &mut [Complex64]| match (forward, &self.twiddle_z_fwd_64, &self.twiddle_z_inv_64) {
            (true,  Some(tw), _) => forward_inplace_64_with_twiddles(lane, tw.as_slice()),
            (false, _, Some(tw)) => inverse_inplace_64_with_twiddles(lane, tw.as_slice()),
            _ => if forward { fft_forward_64(lane) } else { fft_inverse_64(lane) },
        };
        if data_slice.len() > RAYON_THRESHOLD {
            data_slice.par_chunks_mut(self.nz).for_each(lane_fn_64);
        } else {
            data_slice.chunks_mut(self.nz).for_each(lane_fn_64);
        }
    }

    fn axis_pass_complex_f32(&self, data: &mut Array3<Complex32>, axis: Axis, forward: bool) {
        if data.len_of(axis) <= 1 {
            return;
        }
        if axis.index() == 2 {
            self.axis2_pass_complex_f32(data, forward);
            return;
        }
        if axis.index() == 1 {
            self.axis1_pass_complex_f32(data, forward);
            return;
        }
        if axis.index() == 0 {
            self.axis0_pass_complex_f32(data, forward);
        }
    }

    fn axis1_pass_complex_f32(&self, data: &mut Array3<Complex32>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D f32 complex data must be contiguous");
        let mut scratch = self.scratch_y_32.lock().expect("scratch_y_32 mutex poisoned");
        for i in 0..self.nx {
            for j_t in (0..self.ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(self.ny);
                for k_t in (0..self.nz).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(self.nz);
                    for j in j_t..j_end {
                        let src = (i * self.ny + j) * self.nz;
                        for k in k_t..k_end {
                            scratch[(i * self.nz + k) * self.ny + j] = data_slice[src + k];
                        }
                    }
                }
            }
        }
        let lane_fn_32 = |lane: &mut [Complex32]| match (forward, &self.twiddle_y_fwd_32, &self.twiddle_y_inv_32) {
            (true,  Some(tw), _) => forward_inplace_32_with_twiddles(lane, tw.as_slice()),
            (false, _, Some(tw)) => inverse_inplace_32_with_twiddles(lane, tw.as_slice()),
            _ => if forward { fft_forward_32(lane) } else { fft_inverse_32(lane) },
        };
        if scratch.len() > RAYON_THRESHOLD {
            scratch.par_chunks_mut(self.ny).for_each(lane_fn_32);
        } else {
            scratch.chunks_mut(self.ny).for_each(lane_fn_32);
        }
        for i in 0..self.nx {
            for j_t in (0..self.ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(self.ny);
                for k_t in (0..self.nz).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(self.nz);
                    for j in j_t..j_end {
                        let dst = (i * self.ny + j) * self.nz;
                        for k in k_t..k_end {
                            data_slice[dst + k] = scratch[(i * self.nz + k) * self.ny + j];
                        }
                    }
                }
            }
        }
    }

    fn axis0_pass_complex_f32(&self, data: &mut Array3<Complex32>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D f32 complex data must be contiguous");
        let mut scratch = self.scratch_x_32.lock().expect("scratch_x_32 mutex poisoned");
        for i in 0..self.nx {
            let src_base = i * self.ny * self.nz;
            for j_t in (0..self.ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(self.ny);
                for k_t in (0..self.nz).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(self.nz);
                    for j in j_t..j_end {
                        let src = src_base + j * self.nz;
                        for k in k_t..k_end {
                            scratch[(j * self.nz + k) * self.nx + i] = data_slice[src + k];
                        }
                    }
                }
            }
        }
        let lane_fn_32 = |lane: &mut [Complex32]| match (forward, &self.twiddle_x_fwd_32, &self.twiddle_x_inv_32) {
            (true,  Some(tw), _) => forward_inplace_32_with_twiddles(lane, tw.as_slice()),
            (false, _, Some(tw)) => inverse_inplace_32_with_twiddles(lane, tw.as_slice()),
            _ => if forward { fft_forward_32(lane) } else { fft_inverse_32(lane) },
        };
        if scratch.len() > RAYON_THRESHOLD {
            scratch.par_chunks_mut(self.nx).for_each(lane_fn_32);
        } else {
            scratch.chunks_mut(self.nx).for_each(lane_fn_32);
        }
        for i in 0..self.nx {
            let dst_base = i * self.ny * self.nz;
            for j_t in (0..self.ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(self.ny);
                for k_t in (0..self.nz).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(self.nz);
                    for j in j_t..j_end {
                        let dst = dst_base + j * self.nz;
                        for k in k_t..k_end {
                            data_slice[dst + k] = scratch[(j * self.nz + k) * self.nx + i];
                        }
                    }
                }
            }
        }
    }

    fn axis2_pass_complex_f32(&self, data: &mut Array3<Complex32>, forward: bool) {
        if self.nz <= 1 {
            return;
        }
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D f32 complex data must be contiguous");
        let lane_fn_32 = |lane: &mut [Complex32]| match (forward, &self.twiddle_z_fwd_32, &self.twiddle_z_inv_32) {
            (true,  Some(tw), _) => forward_inplace_32_with_twiddles(lane, tw.as_slice()),
            (false, _, Some(tw)) => inverse_inplace_32_with_twiddles(lane, tw.as_slice()),
            _ => if forward { fft_forward_32(lane) } else { fft_inverse_32(lane) },
        };
        if data_slice.len() > RAYON_THRESHOLD {
            data_slice.par_chunks_mut(self.nz).for_each(lane_fn_32);
        } else {
            data_slice.chunks_mut(self.nz).for_each(lane_fn_32);
        }
    }

    fn check_real_shape(&self, dim: (usize, usize, usize), label: &str) {
        assert_eq!(dim, (self.nx, self.ny, self.nz), "{label} shape mismatch");
    }

    fn check_full_complex_shape(&self, dim: (usize, usize, usize), label: &str) {
        assert_eq!(dim, (self.nx, self.ny, self.nz), "{label} shape mismatch");
    }

    fn inverse_complex_to_real_with_workspace(
        &self,
        input: &Array3<Complex64>,
        output: &mut Array3<f64>,
    ) {
        self.check_full_complex_shape(input.dim(), "inverse input");
        self.check_real_shape(output.dim(), "inverse output");
        let transformed = self.inverse_complex(input);
        Zip::from(output).and(&transformed).for_each(|out, value| {
            *out = value.re;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signal(nx: usize, ny: usize, nz: usize) -> Array3<f64> {
        Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            (i as f64 * 0.31 + j as f64 * 0.17 + k as f64 * 0.41).sin()
                + 0.5 * (i as f64 * 0.07 + j as f64 * 0.23 + k as f64 * 0.13).cos()
        })
    }

    /// Roundtrip identity: inverse(forward(x)) == x for asymmetric non-power-of-two sizes.
    #[test]
    fn roundtrip_recovers_asymmetric_inputs() {
        for (nx, ny, nz) in [(7usize, 13usize, 5usize), (16, 8, 9)] {
            let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
            let plan = FftPlan3D::new(shape);
            let input = make_signal(nx, ny, nz);
            let recovered = plan.inverse(&plan.forward(&input));
            for (a, b) in input.iter().zip(recovered.iter()) {
                let err = (a - b).abs();
                assert!(err < 1e-10, "roundtrip n=({nx},{ny},{nz}) err={err:.2e}");
            }
        }
    }

    /// Linearity: forward(a*s1 + b*s2) == a*forward(s1) + b*forward(s2), eps 1e-9.
    #[test]
    fn forward_is_linear() {
        let (nx, ny, nz) = (5usize, 7usize, 3usize);
        let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
        let plan = FftPlan3D::new(shape);
        let a = 2.3f64;
        let b = -1.7f64;
        let s1 = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            (i as f64 * 0.3 + j as f64 * 0.2 + k as f64 * 0.5).sin()
        });
        let s2 = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            (i as f64 * 0.7 + j as f64 * 0.4 + k as f64 * 0.1).cos()
        });
        let combined = &s1 * a + &s2 * b;
        let lhs = plan.forward(&combined);
        let rhs = plan.forward(&s1).mapv(|v| v * a) + plan.forward(&s2).mapv(|v| v * b);
        for (l, r) in lhs.iter().zip(rhs.iter()) {
            let err = (l - r).norm();
            assert!(err < 1e-9, "linearity err={err:.2e}");
        }
    }

    /// Parseval: sum|x|^2 == sum|X|^2 / (nx*ny*nz), eps 1e-6.
    #[test]
    fn parseval_identity_holds() {
        let (nx, ny, nz) = (8usize, 6usize, 5usize);
        let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
        let plan = FftPlan3D::new(shape);
        let input = make_signal(nx, ny, nz);
        let spectrum = plan.forward(&input);
        let time_energy: f64 = input.iter().map(|x| x * x).sum();
        let spectral_energy: f64 =
            spectrum.iter().map(|x| x.norm_sqr()).sum::<f64>() / (nx * ny * nz) as f64;
        let err = (time_energy - spectral_energy).abs();
        assert!(err < 1e-6, "Parseval err={err:.2e}");
    }

    /// Complex in-place forward then inverse recovers original, eps 1e-10.
    #[test]
    fn complex_inplace_roundtrip() {
        let (nx, ny, nz) = (8usize, 4usize, 6usize);
        let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
        let plan = FftPlan3D::new(shape);
        let input = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            Complex64::new(
                (i as f64 * 0.2).sin(),
                (j as f64 * 0.3 + k as f64 * 0.1).cos(),
            )
        });
        let mut data = input.clone();
        plan.forward_complex_inplace(&mut data);
        plan.inverse_complex_inplace(&mut data);
        for (a, b) in input.iter().zip(data.iter()) {
            let err = (a - b).norm();
            assert!(err < 1e-10, "complex roundtrip err={err:.2e}");
        }
    }

    /// inverse_complex_to_real_into matches the allocating inverse_complex_to_real.
    #[test]
    fn caller_owned_inverse_matches_allocating() {
        let (nx, ny, nz) = (6usize, 5usize, 4usize);
        let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
        let plan = FftPlan3D::new(shape);
        let input = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            Complex64::new(
                (i as f64 * 0.5 + j as f64 * 0.3).sin(),
                (k as f64 * 0.7).cos(),
            )
        });
        let alloc_result = plan.inverse_complex_to_real(&input);
        let mut out = Array3::<f64>::zeros((nx, ny, nz));
        let mut scratch = Array3::<Complex64>::zeros((nx, ny, nz));
        plan.inverse_complex_to_real_into(&input, &mut out, &mut scratch);
        for (a, b) in alloc_result.iter().zip(out.iter()) {
            let err = (a - b).abs();
            assert!(err < 1e-14, "caller-owned vs alloc mismatch: {err:.2e}");
        }
    }

    /// forward_real_to_complex_into panics on wrong output shape.
    #[test]
    #[should_panic(expected = "forward output shape mismatch")]
    fn forward_rejects_wrong_shape() {
        let shape = Shape3D::new(4, 4, 4).expect("valid shape");
        let plan = FftPlan3D::new(shape);
        let input = Array3::<f64>::zeros((4, 4, 4));
        let mut wrong_output = Array3::<Complex64>::zeros((4, 4, 3));
        plan.forward_real_to_complex_into(&input, &mut wrong_output);
    }

    /// LOW_PRECISION_F32 typed roundtrip stays within f32 tolerance.
    #[test]
    fn typed_low_precision_roundtrip() {
        let (nx, ny, nz) = (8usize, 8usize, 8usize);
        let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
        let plan = FftPlan3D::with_precision(shape, PrecisionProfile::LOW_PRECISION_F32);
        let input = Array3::<f32>::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            (i as f32 * 0.31 + j as f32 * 0.17 + k as f32 * 0.41).sin()
        });
        let spectrum = plan.forward_typed(&input);
        let recovered: Array3<f32> = plan.inverse_typed(&spectrum);
        for (a, b) in input.iter().zip(recovered.iter()) {
            let err = (a - b).abs();
            assert!(err < 1e-4, "low-precision roundtrip err={err:.2e}");
        }
    }

    /// MIXED_PRECISION_F16_F32 typed roundtrip stays within f16 tolerance.
    #[test]
    fn typed_mixed_precision_roundtrip() {
        let (nx, ny, nz) = (8usize, 8usize, 8usize);
        let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
        let plan = FftPlan3D::with_precision(shape, PrecisionProfile::MIXED_PRECISION_F16_F32);
        let input = Array3::<f16>::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            f16::from_f32((i as f32 * 0.31 + j as f32 * 0.17 + k as f32 * 0.41).sin())
        });
        let spectrum = plan.forward_typed(&input);
        let recovered: Array3<f16> = plan.inverse_typed(&spectrum);
        for (a, b) in input.iter().zip(recovered.iter()) {
            let err = (a.to_f32() - b.to_f32()).abs();
            assert!(err < 5e-2, "mixed-precision roundtrip err={err:.2e}");
        }
    }
}
