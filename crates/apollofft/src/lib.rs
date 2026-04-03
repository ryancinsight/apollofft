#![warn(missing_docs)]
//! Apollo FFT core crate.
//!
//! This crate owns the reusable CPU FFT implementation, shared shape and error
//! contracts, backend abstractions, and cache-backed convenience helpers.

pub mod application;
pub mod backend;
pub mod domain;
pub mod error;
pub mod infrastructure;
pub mod nufft;
pub mod plan;
pub mod types;

pub use application::cache::{
    get_fft_for_grid, Fft1dCache, Fft1dCacheKey, Fft2dCache, Fft2dCacheKey, Fft3dCache,
    Fft3dCacheKey, FFT_CACHE, FFT_CACHE_1D, FFT_CACHE_2D, FFT_CACHE_3D,
};
pub use backend::FftBackend;
pub use error::{ApolloError, ApolloResult};
pub use infrastructure::cpu_backend::CpuBackend;
pub use nufft::{
    nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d, nufft_type1_3d_fast, nufft_type2_1d,
    nufft_type2_1d_fast, NufftPlan1D, NufftPlan3D, DEFAULT_NUFFT_KERNEL_WIDTH,
    DEFAULT_NUFFT_OVERSAMPLING,
};
pub use num_complex::Complex64;
pub use plan::{FftPlan1D, FftPlan2D, FftPlan3D};
pub use types::{
    BackendKind, HalfSpectrum3D, Normalization, Shape1D, Shape2D, Shape3D, UniformDomain1D,
    UniformGrid3D,
};

use ndarray::{Array1, Array2, Array3, Zip};
use rayon::prelude::*;

/// Compatibility alias for legacy `kwavers` code paths.
pub type ProcessorFft3d = FftPlan3D;

/// Forward 1D FFT of a real signal.
#[must_use]
pub fn fft_1d_array(field: &Array1<f64>) -> Array1<Complex64> {
    FFT_CACHE_1D.get_or_create(field.len()).forward(field)
}

/// Forward 2D FFT of a real array.
#[must_use]
pub fn fft_2d_array(field: &Array2<f64>) -> Array2<Complex64> {
    let (nx, ny) = field.dim();
    FFT_CACHE_2D.get_or_create(nx, ny).forward(field)
}

/// Forward 3D FFT of a real array.
#[must_use]
pub fn fft_3d_array(field: &Array3<f64>) -> Array3<Complex64> {
    let (nx, ny, nz) = field.dim();
    FFT_CACHE_3D.get_or_create(nx, ny, nz).forward(field)
}

/// Inverse 1D FFT of a complex signal.
#[must_use]
pub fn ifft_1d_array(field_hat: &Array1<Complex64>) -> Array1<f64> {
    FFT_CACHE_1D
        .get_or_create(field_hat.len())
        .inverse(field_hat)
}

/// Inverse 2D FFT of a complex array.
#[must_use]
pub fn ifft_2d_array(field_hat: &Array2<Complex64>) -> Array2<f64> {
    let (nx, ny) = field_hat.dim();
    FFT_CACHE_2D.get_or_create(nx, ny).inverse(field_hat)
}

/// Inverse 3D FFT of a complex array.
#[must_use]
pub fn ifft_3d_array(field_hat: &Array3<Complex64>) -> Array3<f64> {
    let (nx, ny, nz) = field_hat.dim();
    FFT_CACHE_3D.get_or_create(nx, ny, nz).inverse(field_hat)
}

/// Forward complex 1D FFT in-place.
pub fn fft_1d_complex_inplace(data: &mut Array1<Complex64>) {
    FFT_CACHE_1D
        .get_or_create(data.len())
        .forward_complex_inplace(data);
}

/// Inverse complex 1D FFT in-place with FFTW-compatible normalization.
pub fn ifft_1d_complex_inplace(data: &mut Array1<Complex64>) {
    let n = data.len();
    FFT_CACHE_1D.get_or_create(n).inverse_complex_inplace(data);
    let norm = 1.0 / n as f64;
    data.par_iter_mut().for_each(|value| *value *= norm);
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
        .get_or_create(nx, ny)
        .forward_complex_inplace(data);
}

/// Inverse complex 2D FFT in-place with FFTW-compatible normalization.
pub fn ifft_2d_complex_inplace(data: &mut Array2<Complex64>) {
    let (nx, ny) = data.dim();
    FFT_CACHE_2D
        .get_or_create(nx, ny)
        .inverse_complex_inplace(data);
    let norm = 1.0 / (nx * ny) as f64;
    data.par_iter_mut().for_each(|value| *value *= norm);
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
        .get_or_create(nx, ny, nz)
        .forward_complex_inplace(data);
}

/// Inverse complex 3D FFT in-place with FFTW-compatible normalization.
pub fn ifft_3d_complex_inplace(data: &mut Array3<Complex64>) {
    let (nx, ny, nz) = data.dim();
    FFT_CACHE_3D
        .get_or_create(nx, ny, nz)
        .inverse_complex_inplace(data);
    let norm = 1.0 / (nx * ny * nz) as f64;
    data.par_iter_mut().for_each(|value| *value *= norm);
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
        .get_or_create(nx, ny, nz)
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
        .get_or_create(nx, ny, nz)
        .inverse_complex_inplace(field_hat);
    let norm = 1.0 / (nx * ny * nz) as f64;
    Zip::from(out.view_mut())
        .and(field_hat.view())
        .par_for_each(|dst, src| *dst = src.re * norm);
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
            assert!((lhs - rhs).norm() < 1e-14);
        }
    }
}
