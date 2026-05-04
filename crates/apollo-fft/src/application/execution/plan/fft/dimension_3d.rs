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

use crate::application::execution::kernel::radix2::{
    build_forward_twiddle_table_32, build_forward_twiddle_table_64, build_inverse_twiddle_table_32,
    build_inverse_twiddle_table_64, forward_inplace_32_with_twiddles,
    forward_inplace_64_with_twiddles, inverse_inplace_32_with_twiddles,
    inverse_inplace_64_with_twiddles,
};
use crate::application::execution::kernel::{
    fft_forward_32, fft_forward_64, fft_inverse_32, fft_inverse_64,
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
    // --- r2c/c2r: half-length z-axis twiddles (m = nz/2) ---
    //
    // The r2c forward z-axis pass applies a length-m complex DFT to packed real
    // pairs, followed by Cooley-Tukey extraction. These tables are the twiddle
    // factors for the length-m sub-FFT (precomputed when m is a power of two).
    twiddle_zh_fwd_64: Option<Vec<Complex64>>,
    twiddle_zh_inv_64: Option<Vec<Complex64>>,
    // f32 r2c/c2r fields reserved for future Complex32 r2c implementation.
    #[allow(dead_code)]
    twiddle_zh_fwd_32: Option<Vec<Complex32>>,
    #[allow(dead_code)]
    twiddle_zh_inv_32: Option<Vec<Complex32>>,
    /// Extraction twiddles W_k = exp(−2πi·k/nz) for k = 0..nz_c−1.
    /// Used in the Cooley-Tukey r2c split step and its inverse.
    r2c_twiddles_64: Vec<Complex64>,
    #[allow(dead_code)]
    r2c_twiddles_32: Vec<Complex32>,
    // --- preallocated scratch for r2c y and x passes (half-spectrum volume) ---
    scratch_r2c_y_64: std::sync::Mutex<Vec<Complex64>>,
    scratch_r2c_x_64: std::sync::Mutex<Vec<Complex64>>,
    #[allow(dead_code)]
    scratch_r2c_y_32: std::sync::Mutex<Vec<Complex32>>,
    #[allow(dead_code)]
    scratch_r2c_x_32: std::sync::Mutex<Vec<Complex32>>,
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
        // Half-length z-axis sub-FFT for r2c/c2r (m = nz/2).
        let m = nz / 2;
        let nz_c_val = m + 1; // = nz/2+1
        let r2c_vol = nx * ny * nz_c_val;
        // Extraction twiddles W_k = exp(−2πi·k/nz) for k = 0..nz_c_val−1.
        let r2c_twiddles_64: Vec<Complex64> = (0..nz_c_val)
            .map(|k| {
                let a = -std::f64::consts::TAU * k as f64 / nz as f64;
                Complex64::new(a.cos(), a.sin())
            })
            .collect();
        let r2c_twiddles_32: Vec<Complex32> = (0..nz_c_val)
            .map(|k| {
                let a = -std::f32::consts::TAU * k as f32 / nz as f32;
                Complex32::new(a.cos(), a.sin())
            })
            .collect();
        Self {
            nx,
            ny,
            nz,
            nz_c: nz_c_val,
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
            twiddle_zh_fwd_64: make64(m, true),
            twiddle_zh_inv_64: make64(m, false),
            twiddle_zh_fwd_32: make32(m, true),
            twiddle_zh_inv_32: make32(m, false),
            r2c_twiddles_64,
            r2c_twiddles_32,
            scratch_r2c_y_64: std::sync::Mutex::new(vec![Complex64::default(); r2c_vol]),
            scratch_r2c_x_64: std::sync::Mutex::new(vec![Complex64::default(); r2c_vol]),
            scratch_r2c_y_32: std::sync::Mutex::new(vec![Complex32::default(); r2c_vol]),
            scratch_r2c_x_32: std::sync::Mutex::new(vec![Complex32::default(); r2c_vol]),
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
        let mut scratch = self
            .scratch_y_64
            .lock()
            .expect("scratch_y_64 mutex poisoned");
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
        let lane_fn_64 = |lane: &mut [Complex64]| match (
            forward,
            &self.twiddle_y_fwd_64,
            &self.twiddle_y_inv_64,
        ) {
            (true, Some(tw), _) => forward_inplace_64_with_twiddles(lane, tw.as_slice()),
            (false, _, Some(tw)) => inverse_inplace_64_with_twiddles(lane, tw.as_slice()),
            _ => {
                if forward {
                    fft_forward_64(lane)
                } else {
                    fft_inverse_64(lane)
                }
            }
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
        let mut scratch = self
            .scratch_x_64
            .lock()
            .expect("scratch_x_64 mutex poisoned");
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
        let lane_fn_64 = |lane: &mut [Complex64]| match (
            forward,
            &self.twiddle_x_fwd_64,
            &self.twiddle_x_inv_64,
        ) {
            (true, Some(tw), _) => forward_inplace_64_with_twiddles(lane, tw.as_slice()),
            (false, _, Some(tw)) => inverse_inplace_64_with_twiddles(lane, tw.as_slice()),
            _ => {
                if forward {
                    fft_forward_64(lane)
                } else {
                    fft_inverse_64(lane)
                }
            }
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
        let lane_fn_64 = |lane: &mut [Complex64]| match (
            forward,
            &self.twiddle_z_fwd_64,
            &self.twiddle_z_inv_64,
        ) {
            (true, Some(tw), _) => forward_inplace_64_with_twiddles(lane, tw.as_slice()),
            (false, _, Some(tw)) => inverse_inplace_64_with_twiddles(lane, tw.as_slice()),
            _ => {
                if forward {
                    fft_forward_64(lane)
                } else {
                    fft_inverse_64(lane)
                }
            }
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
        let mut scratch = self
            .scratch_y_32
            .lock()
            .expect("scratch_y_32 mutex poisoned");
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
        let lane_fn_32 = |lane: &mut [Complex32]| match (
            forward,
            &self.twiddle_y_fwd_32,
            &self.twiddle_y_inv_32,
        ) {
            (true, Some(tw), _) => forward_inplace_32_with_twiddles(lane, tw.as_slice()),
            (false, _, Some(tw)) => inverse_inplace_32_with_twiddles(lane, tw.as_slice()),
            _ => {
                if forward {
                    fft_forward_32(lane)
                } else {
                    fft_inverse_32(lane)
                }
            }
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
        let mut scratch = self
            .scratch_x_32
            .lock()
            .expect("scratch_x_32 mutex poisoned");
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
        let lane_fn_32 = |lane: &mut [Complex32]| match (
            forward,
            &self.twiddle_x_fwd_32,
            &self.twiddle_x_inv_32,
        ) {
            (true, Some(tw), _) => forward_inplace_32_with_twiddles(lane, tw.as_slice()),
            (false, _, Some(tw)) => inverse_inplace_32_with_twiddles(lane, tw.as_slice()),
            _ => {
                if forward {
                    fft_forward_32(lane)
                } else {
                    fft_inverse_32(lane)
                }
            }
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
        let lane_fn_32 = |lane: &mut [Complex32]| match (
            forward,
            &self.twiddle_z_fwd_32,
            &self.twiddle_z_inv_32,
        ) {
            (true, Some(tw), _) => forward_inplace_32_with_twiddles(lane, tw.as_slice()),
            (false, _, Some(tw)) => inverse_inplace_32_with_twiddles(lane, tw.as_slice()),
            _ => {
                if forward {
                    fft_forward_32(lane)
                } else {
                    fft_inverse_32(lane)
                }
            }
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

    fn check_half_complex_shape(&self, dim: (usize, usize, usize), label: &str) {
        assert_eq!(
            dim,
            (self.nx, self.ny, self.nz_c),
            "{label} half-spectrum shape mismatch"
        );
    }

    // ── Real-to-Complex (R2C) and Complex-to-Real (C2R) transforms ──────────

    /// Forward real-to-complex 3D transform.
    ///
    /// # Mathematical Contract
    ///
    /// For real `x ∈ ℝ^{nx × ny × nz}`, computes the unique half-spectrum
    /// `X ∈ ℂ^{nx × ny × (nz/2+1)}`. The omitted conjugate-symmetric modes
    /// satisfy `X[kx,ky,kz] = X*[(-kx)%nx, (-ky)%ny, nz-kz]` for `kz > nz/2`.
    ///
    /// ## Algorithm — Separable R2C via Cooley-Tukey Split (Sorensen et al. 1987)
    ///
    /// **Z-axis (real → half-complex)**: For each (i,j) row:
    /// 1. Pack real pairs: `h[k] = x[2k] + j·x[2k+1]` for `k = 0..m-1`, `m = nz/2`.
    /// 2. Apply length-m complex FFT: `H = DFT_m(h)`.
    /// 3. Extract half-spectrum via the split identity (Theorem below):
    ///    `X[k] = (H_k + H_mk*)/2 − j·W_k·(H_k − H_mk*)/2`
    ///    where `H_k = H[k mod m]`, `H_mk = conj(H[(m−k) mod m])`,
    ///    and `W_k = exp(−2πi·k/nz)` (precomputed in `r2c_twiddles_64`).
    ///
    /// **Y-axis and X-axis**: Standard complex FFT passes on the `(nx,ny,nz_c)` data.
    ///
    /// ## Theorem: Cooley-Tukey R2C Split
    ///
    /// For real `x[n]`, the N-point DFT splits as `X[k] = E[k] + W_N^k · O[k]`
    /// where `E[k]` and `O[k]` are M = N/2 point DFTs of even/odd samples.
    /// Forming `h[k] = x[2k] + j·x[2k+1]` gives `H[k] = E[k] + j·O[k]`.
    /// Hermitian symmetry of `E` and `O` (both DFTs of real sequences) gives
    /// `E[M−k] = E[k]*` and `O[M−k] = O[k]*`, hence `H[(M−k)%M] = E[k]* + j·O[k]*`.
    /// Therefore `E[k] = (H[k] + H[(M−k)%M]*)/2` and
    /// `O[k] = (H[k] − H[(M−k)%M]*)/(2j)`, yielding the split formula above. □
    ///
    /// ## Normalization
    ///
    /// Forward: no normalization (unnormalized DFT). Inverse `inverse_c2r_into`
    /// normalizes by `1/(nx·ny·nz)`, matching FFTW convention.
    ///
    /// ## Correctness Invariant
    ///
    /// `inverse_c2r_into(forward_r2c_into(x), out, scratch)` recovers `x` with
    /// absolute error `< 1e-10` for f64 on 64³ grids.
    #[must_use]
    pub fn forward_r2c(&self, input: &Array3<f64>) -> Array3<Complex64> {
        let mut out = Array3::<Complex64>::zeros((self.nx, self.ny, self.nz_c));
        self.forward_r2c_into(input, &mut out);
        out
    }

    /// Forward real-to-complex 3D transform into caller-owned half-spectrum buffer.
    pub fn forward_r2c_into(&self, input: &Array3<f64>, output: &mut Array3<Complex64>) {
        self.check_real_shape(input.dim(), "r2c forward input");
        self.check_half_complex_shape(output.dim(), "r2c forward output");

        let nz = self.nz;
        let nz_c = self.nz_c;
        let m = nz / 2;

        // Edge case: nz == 1 — single real sample per row; no z-transform.
        if nz == 1 {
            ndarray::Zip::from(output).and(input).for_each(|out, &v| {
                *out = Complex64::new(v, 0.0);
            });
            return;
        }

        // ── Step 1: Z-axis R2C split-radix pass ─────────────────────────────
        //
        // Processes each (i,j) row independently via rayon par_chunks zip.
        // Each row: pack m complex values, FFT of length m, Cooley-Tukey extraction.
        {
            let in_sl = input
                .as_slice_memory_order()
                .expect("r2c input must be contiguous");
            let out_sl = output
                .as_slice_memory_order_mut()
                .expect("r2c output must be contiguous");
            let n_rows = self.nx * self.ny;
            let large = n_rows > RAYON_THRESHOLD / nz_c.max(1);
            if large {
                in_sl
                    .par_chunks(nz)
                    .zip(out_sl.par_chunks_mut(nz_c))
                    .for_each(|(in_row, out_row)| {
                        self.r2c_z_forward_row_64(in_row, out_row, m);
                    });
            } else {
                in_sl
                    .chunks(nz)
                    .zip(out_sl.chunks_mut(nz_c))
                    .for_each(|(in_row, out_row)| {
                        self.r2c_z_forward_row_64(in_row, out_row, m);
                    });
            }
        }

        // ── Step 2: Y-axis complex FFT on (nx, ny, nz_c) data ───────────────
        {
            let out_sl = output
                .as_slice_memory_order_mut()
                .expect("r2c output must be contiguous");
            self.r2c_axis1_pass_64(out_sl, true);
        }

        // ── Step 3: X-axis complex FFT on (nx, ny, nz_c) data ───────────────
        {
            let out_sl = output
                .as_slice_memory_order_mut()
                .expect("r2c output must be contiguous");
            self.r2c_axis0_pass_64(out_sl, true);
        }
    }

    /// Inverse complex-to-real 3D transform.
    ///
    /// # Mathematical Contract
    ///
    /// For `X ∈ ℂ^{nx × ny × (nz/2+1)}` (the r2c half-spectrum), recovers
    /// `x ∈ ℝ^{nx × ny × nz}` via the conjugate-symmetric IDFT.
    ///
    /// ## Algorithm — Inverse Cooley-Tukey Split
    ///
    /// **X-axis and Y-axis**: Standard complex IFFT passes on `(nx,ny,nz_c)`.
    ///
    /// **Z-axis (half-complex → real)**: For each (i,j) row:
    /// 1. Recover H[k] from X[0..m] using the inverse split formula:
    ///    `H[k] = (X[k] + conj(X[(m−k)%m]) + j·W_k* · (X[k] − conj(X[(m−k)%m]))) / 2`
    ///    where `W_k* = conj(exp(−2πi·k/nz)) = exp(+2πi·k/nz)`.
    /// 2. Apply normalized length-m IFFT: `h = IFFT_m(H)` (divides by m).
    /// 3. Extract: `x[2k] = Re(h[k])/2`, `x[2k+1] = Im(h[k])/2` for k=0..m-1.
    ///    The `/2` corrects for the 1/m normalization of the sub-IFFT (we want
    ///    total z-axis normalization `1/nz = 1/(2m)`).
    ///
    /// Combined with the x- and y-axis IFFT normalizations by `1/nx` and `1/ny`,
    /// the total normalization is `1/(nx·ny·nz)`. □
    #[must_use]
    pub fn inverse_c2r(&self, input: &Array3<Complex64>) -> Array3<f64> {
        let mut out = Array3::<f64>::zeros((self.nx, self.ny, self.nz));
        let mut scratch = input.clone();
        self.inverse_c2r_into_with_scratch(&mut scratch, &mut out);
        out
    }

    /// Inverse complex-to-real 3D transform into caller-owned real buffer.
    ///
    /// `scratch` must have shape `(nx, ny, nz_c)` and is overwritten.
    pub fn inverse_c2r_into(
        &self,
        input: &Array3<Complex64>,
        output: &mut Array3<f64>,
        scratch: &mut Array3<Complex64>,
    ) {
        self.check_half_complex_shape(input.dim(), "c2r inverse input");
        self.check_real_shape(output.dim(), "c2r inverse output");
        self.check_half_complex_shape(scratch.dim(), "c2r inverse scratch");
        scratch.assign(input);
        self.inverse_c2r_into_with_scratch(scratch, output);
    }

    /// Inner c2r using a pre-filled mutable scratch buffer.
    fn inverse_c2r_into_with_scratch(
        &self,
        scratch: &mut Array3<Complex64>,
        output: &mut Array3<f64>,
    ) {
        let nz = self.nz;
        let nz_c = self.nz_c;
        let m = nz / 2;

        // Edge case: nz == 1 — single complex sample per row; extract real part.
        if nz == 1 {
            let norm = 1.0 / (self.nx * self.ny) as f64;
            ndarray::Zip::from(output)
                .and(&*scratch)
                .for_each(|out, &v| {
                    *out = v.re * norm;
                });
            return;
        }

        // ── Step 1: X-axis complex IFFT on (nx, ny, nz_c) data ──────────────
        {
            let sc_sl = scratch
                .as_slice_memory_order_mut()
                .expect("c2r scratch must be contiguous");
            self.r2c_axis0_pass_64(sc_sl, false);
        }

        // ── Step 2: Y-axis complex IFFT on (nx, ny, nz_c) data ──────────────
        {
            let sc_sl = scratch
                .as_slice_memory_order_mut()
                .expect("c2r scratch must be contiguous");
            self.r2c_axis1_pass_64(sc_sl, false);
        }

        // ── Step 3: Z-axis C2R split-radix pass ─────────────────────────────
        {
            let sc_sl = scratch
                .as_slice_memory_order()
                .expect("c2r scratch must be contiguous");
            let out_sl = output
                .as_slice_memory_order_mut()
                .expect("c2r output must be contiguous");
            let n_rows = self.nx * self.ny;
            let large = n_rows > RAYON_THRESHOLD / nz_c.max(1);
            if large {
                sc_sl
                    .par_chunks(nz_c)
                    .zip(out_sl.par_chunks_mut(nz))
                    .for_each(|(in_row, out_row)| {
                        self.r2c_z_inverse_row_64(in_row, out_row, m);
                    });
            } else {
                sc_sl
                    .chunks(nz_c)
                    .zip(out_sl.chunks_mut(nz))
                    .for_each(|(in_row, out_row)| {
                        self.r2c_z_inverse_row_64(in_row, out_row, m);
                    });
            }
        }
    }

    /// Z-axis r2c forward for a single row: pack + sub-FFT + Cooley-Tukey extraction.
    ///
    /// `in_row` has `nz` f64 values; `out_row` receives `nz_c = m+1` Complex64 values.
    fn r2c_z_forward_row_64(&self, in_row: &[f64], out_row: &mut [Complex64], m: usize) {
        let nz = self.nz;

        // Stage 1: pack pairs → complex of length m.
        let mut h: Vec<Complex64> = (0..m)
            .map(|k| Complex64::new(in_row[2 * k], in_row[2 * k + 1]))
            .collect();

        // Stage 2: length-m complex FFT in-place.
        match &self.twiddle_zh_fwd_64 {
            Some(tw) => forward_inplace_64_with_twiddles(&mut h, tw.as_slice()),
            None => fft_forward_64(&mut h),
        }

        // Stage 3: Cooley-Tukey extraction — X[k] for k = 0..m (inclusive).
        // X[k] = (H_k + H_mk*) / 2 − j·W_k·(H_k − H_mk*) / 2
        // where H_k = h[k % m] and H_mk = conj(h[(m−k) % m]).
        // Process pairs simultaneously to avoid read-after-write aliasing.
        let j = Complex64::new(0.0, 1.0);

        // k = 0: H_0 = h[0], H_m0 = conj(h[0]).
        {
            let h0 = h[0];
            let hm0_conj = h0.conj(); // conj(h[(m-0)%m]) = conj(h[0])
            let wk = self.r2c_twiddles_64[0]; // W_0 = 1
            let sum = h0 + hm0_conj;
            let diff = h0 - hm0_conj;
            out_row[0] = sum * 0.5 - j * wk * diff * 0.5;
        }

        // k = m: H_km = h[0] (same slot), H_mk = conj(h[0]).
        {
            let h0 = h[0];
            let hm_conj = h0.conj();
            let wk = self.r2c_twiddles_64[m]; // W_m = exp(−πi) = −1
            let sum = h0 + hm_conj;
            let diff = h0 - hm_conj;
            out_row[m] = sum * 0.5 - j * wk * diff * 0.5;
        }

        // k = 1..m/2 and symmetric counterpart m-k (processed as pairs).
        let k_max = m / 2 + if m % 2 == 1 { 1 } else { 0 };
        for k in 1..k_max {
            let mk = m - k;
            let hk = h[k];
            let hmk = h[mk]; // h[(m-k)]
            let wk = self.r2c_twiddles_64[k];
            let wmk = self.r2c_twiddles_64[mk];

            // X[k]
            let hmk_conj = hmk.conj();
            let sum_k = hk + hmk_conj;
            let diff_k = hk - hmk_conj;
            out_row[k] = sum_k * 0.5 - j * wk * diff_k * 0.5;

            if k != mk {
                // X[m-k] — distinct slot, process as pair.
                let hk_conj = hk.conj();
                let sum_mk = hmk + hk_conj;
                let diff_mk = hmk - hk_conj;
                out_row[mk] = sum_mk * 0.5 - j * wmk * diff_mk * 0.5;
            }
        }

        // When m is even, k = m/2 is its own symmetric partner.
        if m % 2 == 0 && m >= 2 {
            let k = m / 2;
            let hk = h[k];
            let wk = self.r2c_twiddles_64[k];
            let hk_conj = hk.conj();
            let sum_k = hk + hk_conj;
            let diff_k = hk - hk_conj;
            out_row[k] = sum_k * 0.5 - j * wk * diff_k * 0.5;
        }

        let _ = nz; // suppress unused warning when nz used only via m
    }

    /// Z-axis c2r inverse for a single row: inverse split + sub-IFFT + unpack.
    ///
    /// `in_row` has `nz_c = m+1` Complex64 values (the half-spectrum after x/y IFFTs).
    /// `out_row` receives `nz` f64 values.
    ///
    /// Normalization: each output value is divided by `nz` so that combined with
    /// the x-axis (÷nx) and y-axis (÷ny) IFFT normalizations, the total is `÷(nx·ny·nz)`.
    fn r2c_z_inverse_row_64(&self, in_row: &[Complex64], out_row: &mut [f64], m: usize) {
        let nz = self.nz;

        // Stage 1: Recover H[k] for k = 0..m-1 from X[0..m] using the inverse split.
        //
        // Theorem (inverse split): from X[k] = (H_k + H_mk*)/2 − j·W_k·(H_k − H_mk*)/2
        // and X[m−k] = (H_mk + H_k*)/2 + j·conj(W_k)·(H_mk − H_k*)/2, solving gives:
        //   H[k] = (X[k] + conj(X[m−k]) + j·conj(W_k)·(X[k] − conj(X[m−k]))) / 2
        // for k = 0..m−1, where conj(W_k) = exp(+2πi·k/nz) and X[m−k] is at index
        // m−k in the nz_c = m+1 half-spectrum (k=0 uses X[m], the Nyquist bin). □

        let j = Complex64::new(0.0, 1.0);

        let mut h: Vec<Complex64> = vec![Complex64::default(); m];

        // k = 0: conj(X[m-0]) = conj(X[m]).
        //
        // X[m] is the Nyquist bin stored at index m in the nz_c = m+1 half-spectrum.
        // For real input X[m] is real, so conj(X[m]) = X[m].
        // conj(W_0) = conj(1) = 1.
        //
        // Derivation: X[0] = Re{H[0]} + Im{H[0]}, X[m] = Re{H[0]} − Im{H[0]}.
        // H[0] = (X[0] + X[m])/2 + j·(X[0] − X[m])/2, which matches the general
        // formula with xmk_conj = conj(X[m]).
        {
            let xk = in_row[0];
            let xmk_conj = in_row[m].conj(); // X[m] is at index m in the half-spectrum
            let w_conj = self.r2c_twiddles_64[0].conj(); // = 1
            h[0] = (xk + xmk_conj + j * w_conj * (xk - xmk_conj)) * 0.5;
        }

        // k = 1..m-1: symmetric pairs.
        let k_max = m / 2 + if m % 2 == 1 { 1 } else { 0 };
        for k in 1..k_max {
            let mk = m - k;
            let xk = in_row[k];
            let xmk_conj = in_row[mk].conj();
            let w_conj = self.r2c_twiddles_64[k].conj(); // exp(+2πi·k/nz)

            h[k] = (xk + xmk_conj + j * w_conj * (xk - xmk_conj)) * 0.5;

            if k != mk {
                let xmk = in_row[mk];
                let xk_conj = in_row[k].conj();
                let wm_conj = self.r2c_twiddles_64[mk].conj();
                h[mk] = (xmk + xk_conj + j * wm_conj * (xmk - xk_conj)) * 0.5;
            }
        }

        // k = m/2 singleton (even m).
        if m % 2 == 0 && m >= 2 {
            let k = m / 2;
            let xk = in_row[k];
            let xmk_conj = in_row[k].conj(); // self-conjugate slot
            let w_conj = self.r2c_twiddles_64[k].conj();
            h[k] = (xk + xmk_conj + j * w_conj * (xk - xmk_conj)) * 0.5;
        }

        // Stage 2: normalized IFFT of length m in-place.
        match &self.twiddle_zh_inv_64 {
            Some(tw) => inverse_inplace_64_with_twiddles(&mut h, tw.as_slice()),
            None => fft_inverse_64(&mut h),
        }

        // Stage 3: unpack h[k] → out_row[2k], out_row[2k+1].
        //
        // No additional scaling factor. The normalized IFFT_m satisfies
        //   IFFT_m(FFT_m(h)) = h
        // so stage 2 recovers h exactly. The forward z r2c packs
        // h[n] = x[2n] + j·x[2n+1] and applies an unnormalized FFT_m;
        // stage 1 recovers H_true, stage 2 recovers h_true; unpacking gives
        // x back directly. No residual normalization factor arises here: the
        // z-axis forward/inverse pair is its own identity (H_true recovered via
        // the inverse split, IFFT_m cancels FFT_m). The x- and y-axis normalized
        // IFFTs in the outer c2r caller cancel their respective DFTs identically.
        for k in 0..m {
            out_row[2 * k] = h[k].re;
            out_row[2 * k + 1] = h[k].im;
        }

        let _ = nz; // suppress unused warning
    }

    /// Y-axis (axis-1) complex FFT/IFFT pass on `(nx, ny, nz_c)` half-spectrum data.
    ///
    /// Operates on a flat slice of length `nx * ny * nz_c` laid out in C order
    /// `[i][j][k]`, applying length-ny transforms along axis 1.
    fn r2c_axis1_pass_64(&self, data: &mut [Complex64], forward: bool) {
        if self.ny <= 1 {
            return;
        }
        let nz_c = self.nz_c;
        let nx = self.nx;
        let ny = self.ny;
        let total = nx * ny * nz_c;
        let mut scratch = self
            .scratch_r2c_y_64
            .lock()
            .expect("scratch_r2c_y_64 mutex poisoned");

        // Cache-blocked gather: data[i,j,k] → scratch[i,k,j] (transpose j↔k).
        for i in 0..nx {
            for j_t in (0..ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(ny);
                for k_t in (0..nz_c).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(nz_c);
                    for j in j_t..j_end {
                        let src = (i * ny + j) * nz_c;
                        for k in k_t..k_end {
                            scratch[(i * nz_c + k) * ny + j] = data[src + k];
                        }
                    }
                }
            }
        }

        let lane_fn = |lane: &mut [Complex64]| match (
            forward,
            &self.twiddle_y_fwd_64,
            &self.twiddle_y_inv_64,
        ) {
            (true, Some(tw), _) => forward_inplace_64_with_twiddles(lane, tw.as_slice()),
            (false, _, Some(tw)) => inverse_inplace_64_with_twiddles(lane, tw.as_slice()),
            _ => {
                if forward {
                    fft_forward_64(lane);
                } else {
                    fft_inverse_64(lane);
                }
            }
        };

        if total > RAYON_THRESHOLD {
            scratch[..total].par_chunks_mut(ny).for_each(lane_fn);
        } else {
            scratch[..total].chunks_mut(ny).for_each(lane_fn);
        }

        // Cache-blocked scatter: scratch[i,k,j] → data[i,j,k].
        for i in 0..nx {
            for j_t in (0..ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(ny);
                for k_t in (0..nz_c).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(nz_c);
                    for j in j_t..j_end {
                        let dst = (i * ny + j) * nz_c;
                        for k in k_t..k_end {
                            data[dst + k] = scratch[(i * nz_c + k) * ny + j];
                        }
                    }
                }
            }
        }
    }

    /// X-axis (axis-0) complex FFT/IFFT pass on `(nx, ny, nz_c)` half-spectrum data.
    fn r2c_axis0_pass_64(&self, data: &mut [Complex64], forward: bool) {
        if self.nx <= 1 {
            return;
        }
        let nz_c = self.nz_c;
        let nx = self.nx;
        let ny = self.ny;
        let total = nx * ny * nz_c;
        let mut scratch = self
            .scratch_r2c_x_64
            .lock()
            .expect("scratch_r2c_x_64 mutex poisoned");

        // Cache-blocked gather: data[i,j,k] → scratch[j,k,i].
        for i in 0..nx {
            let src_base = i * ny * nz_c;
            for j_t in (0..ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(ny);
                for k_t in (0..nz_c).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(nz_c);
                    for j in j_t..j_end {
                        let src = src_base + j * nz_c;
                        for k in k_t..k_end {
                            scratch[(j * nz_c + k) * nx + i] = data[src + k];
                        }
                    }
                }
            }
        }

        let lane_fn = |lane: &mut [Complex64]| match (
            forward,
            &self.twiddle_x_fwd_64,
            &self.twiddle_x_inv_64,
        ) {
            (true, Some(tw), _) => forward_inplace_64_with_twiddles(lane, tw.as_slice()),
            (false, _, Some(tw)) => inverse_inplace_64_with_twiddles(lane, tw.as_slice()),
            _ => {
                if forward {
                    fft_forward_64(lane);
                } else {
                    fft_inverse_64(lane);
                }
            }
        };

        if total > RAYON_THRESHOLD {
            scratch[..total].par_chunks_mut(nx).for_each(lane_fn);
        } else {
            scratch[..total].chunks_mut(nx).for_each(lane_fn);
        }

        // Cache-blocked scatter: scratch[j,k,i] → data[i,j,k].
        for i in 0..nx {
            let dst_base = i * ny * nz_c;
            for j_t in (0..ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(ny);
                for k_t in (0..nz_c).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(nz_c);
                    for j in j_t..j_end {
                        let dst = dst_base + j * nz_c;
                        for k in k_t..k_end {
                            data[dst + k] = scratch[(j * nz_c + k) * nx + i];
                        }
                    }
                }
            }
        }
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

    /// R2C forward then C2R inverse recovers the original real signal, eps 1e-10.
    ///
    /// Validates the Cooley-Tukey split formula and its inverse for power-of-two sizes.
    #[test]
    fn r2c_roundtrip_power_of_two() {
        for (nx, ny, nz) in [(4usize, 4usize, 4usize), (8, 8, 8), (16, 4, 8)] {
            let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
            let plan = FftPlan3D::new(shape);
            let input = make_signal(nx, ny, nz);
            let spectrum = plan.forward_r2c(&input);
            assert_eq!(spectrum.dim(), (nx, ny, nz / 2 + 1));
            let recovered = plan.inverse_c2r(&spectrum);
            assert_eq!(recovered.dim(), (nx, ny, nz));
            for (a, b) in input.iter().zip(recovered.iter()) {
                let err = (a - b).abs();
                assert!(err < 1e-10, "r2c roundtrip ({nx},{ny},{nz}) err={err:.2e}");
            }
        }
    }

    /// R2C for non-power-of-two nz uses the Bluestein fallback.
    #[test]
    fn r2c_roundtrip_non_power_of_two_nz() {
        // nz = 6 (not power of two), so sub-FFT length m=3 falls back to Bluestein.
        let (nx, ny, nz) = (4usize, 4usize, 6usize);
        let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
        let plan = FftPlan3D::new(shape);
        let input = make_signal(nx, ny, nz);
        let spectrum = plan.forward_r2c(&input);
        let recovered = plan.inverse_c2r(&spectrum);
        for (a, b) in input.iter().zip(recovered.iter()) {
            let err = (a - b).abs();
            assert!(err < 1e-9, "r2c non-pow2 roundtrip err={err:.2e}");
        }
    }

    /// R2C half-spectrum matches the first nz_c rows of the full-complex forward FFT.
    ///
    /// Correctness invariant: `forward_r2c(x)[i,j,k] == forward(x)[i,j,k]` for k = 0..nz_c-1.
    #[test]
    fn r2c_spectrum_matches_full_forward() {
        let (nx, ny, nz) = (8usize, 6usize, 8usize);
        let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
        let plan = FftPlan3D::new(shape);
        let input = make_signal(nx, ny, nz);

        let full_spectrum = plan.forward(&input);
        let half_spectrum = plan.forward_r2c(&input);

        let nz_c = nz / 2 + 1;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz_c {
                    let full = full_spectrum[[i, j, k]];
                    let half = half_spectrum[[i, j, k]];
                    let err = (full - half).norm();
                    assert!(
                        err < 1e-10,
                        "r2c vs full: [{i},{j},{k}] full={full} half={half} err={err:.2e}"
                    );
                }
            }
        }
    }

    /// Parseval identity for r2c: sum|x|² == sum|X_half|² * 2 / (nx*ny*nz), eps 1e-6.
    ///
    /// For real x, the full spectrum satisfies sum|X|² = nx*ny*nz * sum|x|².
    /// The half-spectrum has nz_c = nz/2+1 slabs; the interior slabs (k=1..nz/2-1)
    /// are duplicated (Hermitian symmetry), so their energy counts double:
    ///   sum|x|² = [|X[*,*,0]|² + |X[*,*,nz/2]|² + 2*sum_{k=1}^{nz/2-1} |X[*,*,k]|²] / (nx*ny*nz)
    #[test]
    fn r2c_parseval_holds() {
        let (nx, ny, nz) = (8usize, 6usize, 8usize);
        let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
        let plan = FftPlan3D::new(shape);
        let input = make_signal(nx, ny, nz);

        let half_spectrum = plan.forward_r2c(&input);
        let nz_c = nz / 2 + 1;

        let time_energy: f64 = input.iter().map(|&x| x * x).sum();

        // Compute weighted spectral energy accounting for Hermitian symmetry.
        let mut spectral_energy = 0.0_f64;
        for i in 0..nx {
            for j in 0..ny {
                // k = 0 and k = nz/2 (if nz even): boundary terms, weight 1.
                spectral_energy += half_spectrum[[i, j, 0]].norm_sqr();
                if nz % 2 == 0 {
                    spectral_energy += half_spectrum[[i, j, nz_c - 1]].norm_sqr();
                }
                // k = 1..nz_c-2: interior, each mode appears twice (Hermitian pair).
                let k_end = if nz % 2 == 0 { nz_c - 1 } else { nz_c };
                for k in 1..k_end {
                    spectral_energy += 2.0 * half_spectrum[[i, j, k]].norm_sqr();
                }
            }
        }
        spectral_energy /= (nx * ny * nz) as f64;

        let err = (time_energy - spectral_energy).abs();
        assert!(
            err < 1e-6,
            "r2c Parseval err={err:.2e}: time_energy={time_energy:.6e} spectral={spectral_energy:.6e}"
        );
    }

    /// C2R inverse using caller-owned scratch matches the allocating inverse_c2r.
    #[test]
    fn c2r_caller_owned_matches_allocating() {
        let (nx, ny, nz) = (6usize, 4usize, 8usize);
        let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
        let plan = FftPlan3D::new(shape);
        let input = make_signal(nx, ny, nz);
        let spectrum = plan.forward_r2c(&input);

        let alloc = plan.inverse_c2r(&spectrum);
        let mut out = Array3::<f64>::zeros((nx, ny, nz));
        let mut scratch = Array3::<Complex64>::zeros((nx, ny, nz / 2 + 1));
        plan.inverse_c2r_into(&spectrum, &mut out, &mut scratch);

        for (a, b) in alloc.iter().zip(out.iter()) {
            let err = (a - b).abs();
            assert!(err < 1e-14, "c2r caller-owned mismatch: {err:.2e}");
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
