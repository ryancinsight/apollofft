#![warn(missing_docs)]
// ── Pedantic suppressions ────────────────────────────────────────────────────
// FFT math inherently uses index-to-float casts for normalisation factors and
// twiddle-factor computation. Grid sizes are bounded by available memory
// (< 2^52), so precision loss and truncation are hypothetical, not real.
// Naming conventions in signal processing (n_x / n_y, coeff_re / coeff_im)
// are standardised in the literature; renaming them reduces clarity.
// Complex FFT plans necessarily carry many boolean precision-mode flags;
// bitset refactors would add complexity without improving safety.
// These suppressions mirror those already configured in the apollo sub-workspace
// Cargo.toml (`similar_names = "allow"`, `too_many_lines = "allow"`, etc.).
#![allow(
    clippy::cast_possible_truncation, // grid sizes < 2^24 for f32, < 2^52 for f64
    clippy::cast_precision_loss,      // usize→f32/f64 normalisation, bounded by memory
    clippy::cast_sign_loss,           // non-negative index arithmetic
    clippy::cast_possible_wrap,       // modular butterfly arithmetic
    clippy::similar_names,            // n_x/n_y/n_z, coeff_re/coeff_im — math convention
    clippy::too_many_lines,           // FFT plan builders are inherently long
    clippy::missing_panics_doc,       // cache helpers panic only on logic error / OOM
    clippy::missing_errors_doc,       // error paths documented inline in struct fields
    clippy::missing_fields_in_debug,  // manual Debug omits large internal buffers by design
    clippy::struct_excessive_bools,   // PrecisionProfile flags are orthogonal bit fields
    clippy::needless_pass_by_value,          // Copy-sized plan/shape types passed by value intentionally
    clippy::missing_const_for_thread_local,  // all thread_local! initializers already use const { }
)]
//! Apollo core crate.
//!
//! This crate owns the reusable CPU FFT implementation, shared shape and error
//! contracts, backend abstractions, and cache-backed convenience helpers.

/// Application-layer execution and orchestration.
pub mod application;
pub mod domain;
/// Infrastructure adapters.
pub mod infrastructure;

/// Compatibility module for backend contracts.
pub mod backend {
    pub use crate::domain::contracts::backend::*;
}

/// Compatibility module for error contracts.
pub mod error {
    pub use crate::domain::contracts::error::*;
}

/// Compatibility module for FFT plan types.
pub mod plan {
    pub use crate::application::execution::plan::fft::{
        dimension_1d::FftPlan1D, dimension_2d::FftPlan2D, dimension_3d::FftPlan3D,
        real_storage::RealFftData,
    };
}

/// Compatibility module for shared metadata types.
pub mod types {
    pub use crate::domain::metadata::precision::{
        BackendKind, ComputePrecision, Normalization, PrecisionMode, PrecisionProfile,
        StoragePrecision,
    };
    pub use crate::domain::metadata::shape::{HalfSpectrum3D, Shape1D, Shape2D, Shape3D};
}
pub use application::orchestration::cache::plans::{
    get_fft_for_grid, Fft1dCache, Fft1dCacheKey, Fft2dCache, Fft2dCacheKey, Fft3dCache,
    Fft3dCacheKey, FFT_CACHE, FFT_CACHE_1D, FFT_CACHE_2D, FFT_CACHE_3D,
};
pub use backend::FftBackend;
pub use error::{ApolloError, ApolloResult};
pub use half::f16;
pub use infrastructure::transport::cpu::CpuBackend;

pub use num_complex::Complex32;
pub use num_complex::Complex64;
pub use plan::{FftPlan1D, FftPlan2D, FftPlan3D, RealFftData};

pub use types::{
    BackendKind, ComputePrecision, HalfSpectrum3D, Normalization, PrecisionMode, PrecisionProfile,
    Shape1D, Shape2D, Shape3D, StoragePrecision,
};

use ndarray::{Array1, Array2, Array3, Zip};

/// Compatibility alias for legacy `kwavers` code paths.
pub type ProcessorFft3d = FftPlan3D;

/// Forward 1D FFT of a real signal.
#[must_use]
pub fn fft_1d_array(field: &Array1<f64>) -> Array1<Complex64> {
    FFT_CACHE_1D
        .get_or_create(Shape1D::new(field.len()).expect("fft_1d_array requires non-zero length"))
        .forward(field)
}

/// Forward 1D FFT of a real array using generic storage dispatch.
#[must_use]
pub fn fft_1d_array_typed<T: RealFftData>(
    field: &Array1<T>,
    profile: PrecisionProfile,
) -> Array1<T::Spectrum> {
    FFT_CACHE_1D
        .get_or_create_with_precision(
            Shape1D::new(field.len()).expect("fft_1d_array_typed requires non-zero length"),
            profile,
        )
        .forward_typed(field)
}

/// Forward 2D FFT of a real array.
#[must_use]
pub fn fft_2d_array(field: &Array2<f64>) -> Array2<Complex64> {
    let (nx, ny) = field.dim();
    FFT_CACHE_2D
        .get_or_create(Shape2D::new(nx, ny).expect("fft_2d_array requires non-zero dimensions"))
        .forward(field)
}

/// Forward 2D FFT of a real array using generic storage dispatch.
#[must_use]
pub fn fft_2d_array_typed<T: RealFftData>(
    field: &Array2<T>,
    profile: PrecisionProfile,
) -> Array2<T::Spectrum> {
    let (nx, ny) = field.dim();
    FFT_CACHE_2D
        .get_or_create_with_precision(
            Shape2D::new(nx, ny).expect("fft_2d_array_typed requires non-zero dimensions"),
            profile,
        )
        .forward_typed(field)
}

/// Forward 3D FFT of a real array.
#[must_use]
pub fn fft_3d_array(field: &Array3<f64>) -> Array3<Complex64> {
    let (nx, ny, nz) = field.dim();
    FFT_CACHE_3D
        .get_or_create(Shape3D::new(nx, ny, nz).expect("fft_3d_array requires non-zero dimensions"))
        .forward(field)
}

/// Forward 3D FFT of a real array using generic storage dispatch.
#[must_use]
pub fn fft_3d_array_typed<T: RealFftData>(
    field: &Array3<T>,
    profile: PrecisionProfile,
) -> Array3<T::Spectrum> {
    let (nx, ny, nz) = field.dim();
    FFT_CACHE_3D
        .get_or_create_with_precision(
            Shape3D::new(nx, ny, nz).expect("fft_3d_array_typed requires non-zero dimensions"),
            profile,
        )
        .forward_typed(field)
}

/// Inverse 1D FFT of a complex signal.
#[must_use]
pub fn ifft_1d_array(field_hat: &Array1<Complex64>) -> Array1<f64> {
    FFT_CACHE_1D
        .get_or_create(
            Shape1D::new(field_hat.len()).expect("ifft_1d_array requires non-zero length"),
        )
        .inverse(field_hat)
}

/// Inverse 1D FFT of a complex spectrum using generic storage dispatch.
#[must_use]
pub fn ifft_1d_array_typed<T: RealFftData>(
    field_hat: &Array1<T::Spectrum>,
    profile: PrecisionProfile,
) -> Array1<T> {
    FFT_CACHE_1D
        .get_or_create_with_precision(
            Shape1D::new(field_hat.len()).expect("ifft_1d_array_typed requires non-zero length"),
            profile,
        )
        .inverse_typed(field_hat)
}

/// Inverse 2D FFT of a complex array.
#[must_use]
pub fn ifft_2d_array(field_hat: &Array2<Complex64>) -> Array2<f64> {
    let (nx, ny) = field_hat.dim();
    FFT_CACHE_2D
        .get_or_create(Shape2D::new(nx, ny).expect("ifft_2d_array requires non-zero dimensions"))
        .inverse(field_hat)
}

/// Inverse 2D FFT of a complex spectrum using generic storage dispatch.
#[must_use]
pub fn ifft_2d_array_typed<T: RealFftData>(
    field_hat: &Array2<T::Spectrum>,
    profile: PrecisionProfile,
) -> Array2<T> {
    let (nx, ny) = field_hat.dim();
    FFT_CACHE_2D
        .get_or_create_with_precision(
            Shape2D::new(nx, ny).expect("ifft_2d_array_typed requires non-zero dimensions"),
            profile,
        )
        .inverse_typed(field_hat)
}

/// Inverse 3D FFT of a complex array.
#[must_use]
pub fn ifft_3d_array(field_hat: &Array3<Complex64>) -> Array3<f64> {
    let (nx, ny, nz) = field_hat.dim();
    FFT_CACHE_3D
        .get_or_create(
            Shape3D::new(nx, ny, nz).expect("ifft_3d_array requires non-zero dimensions"),
        )
        .inverse(field_hat)
}

/// Inverse 3D FFT of a complex spectrum using generic storage dispatch.
#[must_use]
pub fn ifft_3d_array_typed<T: RealFftData>(
    field_hat: &Array3<T::Spectrum>,
    profile: PrecisionProfile,
) -> Array3<T> {
    let (nx, ny, nz) = field_hat.dim();
    FFT_CACHE_3D
        .get_or_create_with_precision(
            Shape3D::new(nx, ny, nz).expect("ifft_3d_array_typed requires non-zero dimensions"),
            profile,
        )
        .inverse_typed(field_hat)
}

/// Forward complex 1D FFT in-place.
pub fn fft_1d_complex_inplace(data: &mut Array1<Complex64>) {
    FFT_CACHE_1D
        .get_or_create(
            Shape1D::new(data.len()).expect("fft_1d_complex_inplace requires non-zero length"),
        )
        .forward_complex_inplace(data);
}

/// Inverse complex 1D FFT in-place with FFTW-compatible normalization.
pub fn ifft_1d_complex_inplace(data: &mut Array1<Complex64>) {
    let n = data.len();
    FFT_CACHE_1D
        .get_or_create(Shape1D::new(n).expect("ifft_1d_complex_inplace requires non-zero length"))
        .inverse_complex_inplace(data);
}

/// Forward complex 1D FFT returning a new buffer.
#[must_use]
pub fn fft_1d_complex(field: &Array1<Complex64>) -> Array1<Complex64> {
    let mut output = field.to_owned();
    fft_1d_complex_inplace(&mut output);
    output
}

/// Inverse complex 1D FFT returning a new buffer.
#[must_use]
pub fn ifft_1d_complex(field_hat: &Array1<Complex64>) -> Array1<Complex64> {
    let mut output = field_hat.to_owned();
    ifft_1d_complex_inplace(&mut output);
    output
}

/// Forward complex 2D FFT in-place.
pub fn fft_2d_complex_inplace(data: &mut Array2<Complex64>) {
    let (nx, ny) = data.dim();
    FFT_CACHE_2D
        .get_or_create(
            Shape2D::new(nx, ny).expect("fft_2d_complex_inplace requires non-zero dimensions"),
        )
        .forward_complex_inplace(data);
}

/// Inverse complex 2D FFT in-place with FFTW-compatible normalization.
pub fn ifft_2d_complex_inplace(data: &mut Array2<Complex64>) {
    let (nx, ny) = data.dim();
    FFT_CACHE_2D
        .get_or_create(
            Shape2D::new(nx, ny).expect("ifft_2d_complex_inplace requires non-zero dimensions"),
        )
        .inverse_complex_inplace(data);
}

/// Forward complex 2D FFT returning a new buffer.
#[must_use]
pub fn fft_2d_complex(field: &Array2<Complex64>) -> Array2<Complex64> {
    let mut output = field.to_owned();
    fft_2d_complex_inplace(&mut output);
    output
}

/// Inverse complex 2D FFT returning a new buffer.
#[must_use]
pub fn ifft_2d_complex(field_hat: &Array2<Complex64>) -> Array2<Complex64> {
    let mut output = field_hat.to_owned();
    ifft_2d_complex_inplace(&mut output);
    output
}

/// Forward complex 3D FFT in-place.
pub fn fft_3d_complex_inplace(data: &mut Array3<Complex64>) {
    let (nx, ny, nz) = data.dim();
    FFT_CACHE_3D
        .get_or_create(
            Shape3D::new(nx, ny, nz).expect("fft_3d_complex_inplace requires non-zero dimensions"),
        )
        .forward_complex_inplace(data);
}

/// Inverse complex 3D FFT in-place with FFTW-compatible normalization.
pub fn ifft_3d_complex_inplace(data: &mut Array3<Complex64>) {
    let (nx, ny, nz) = data.dim();
    FFT_CACHE_3D
        .get_or_create(
            Shape3D::new(nx, ny, nz).expect("ifft_3d_complex_inplace requires non-zero dimensions"),
        )
        .inverse_complex_inplace(data);
}

/// Forward complex 3D FFT returning a new buffer.
#[must_use]
pub fn fft_3d_complex(field: &Array3<Complex64>) -> Array3<Complex64> {
    let mut output = field.to_owned();
    fft_3d_complex_inplace(&mut output);
    output
}

/// Inverse complex 3D FFT returning a new buffer.
#[must_use]
pub fn ifft_3d_complex(field_hat: &Array3<Complex64>) -> Array3<Complex64> {
    let mut output = field_hat.to_owned();
    ifft_3d_complex_inplace(&mut output);
    output
}

/// Forward 3D FFT of a real array into a caller-provided complex buffer.
pub fn fft_3d_array_into(field: &Array3<f64>, out: &mut Array3<Complex64>) {
    debug_assert_eq!(field.dim(), out.dim(), "fft_3d_array_into: shape mismatch");
    Zip::from(out.view_mut())
        .and(field)
        .par_for_each(|dst, &src| {
            *dst = Complex64::new(src, 0.0);
        });
    let (nx, ny, nz) = field.dim();
    FFT_CACHE_3D
        .get_or_create(
            Shape3D::new(nx, ny, nz).expect("fft_3d_array_into requires non-zero dimensions"),
        )
        .forward_complex_inplace(out);
}

/// Inverse 3D FFT of a complex array into a caller-provided real buffer.
pub fn ifft_3d_array_into(field_hat: &mut Array3<Complex64>, out: &mut Array3<f64>) {
    let (nx, ny, nz) = field_hat.dim();
    debug_assert_eq!(
        out.dim(),
        (nx, ny, nz),
        "ifft_3d_array_into: shape mismatch"
    );
    FFT_CACHE_3D
        .get_or_create(
            Shape3D::new(nx, ny, nz).expect("ifft_3d_array_into requires non-zero dimensions"),
        )
        .inverse_complex_inplace(field_hat);
    Zip::from(out.view_mut())
        .and(field_hat.view())
        .par_for_each(|dst, src| *dst = src.re);
}

/// Forward 3D FFT of a complex array into a caller-provided complex buffer.
pub fn fft_3d_complex_into(field: &Array3<Complex64>, out: &mut Array3<Complex64>) {
    out.assign(field);
    fft_3d_complex_inplace(out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn fft_3d_array_into_matches_allocating_path() {
        let (nx, ny, nz) = (8, 8, 8);
        let field =
            Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| ((i + j + k) as f64 * 0.3).sin());
        let expected = fft_3d_array(&field);
        let mut actual = Array3::<Complex64>::zeros((nx, ny, nz));
        fft_3d_array_into(&field, &mut actual);
        for (lhs, rhs) in expected.iter().zip(actual.iter()) {
            assert!((lhs - rhs).norm() < 1e-13);
        }
    }
}
