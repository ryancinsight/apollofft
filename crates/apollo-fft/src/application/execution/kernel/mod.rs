//! Apollo FFT kernel module.
//!
//! Provides three kernel implementations:
//! - `direct`: O(N^2) reference DFT kept for testing and validation.
//! - `radix2`: O(N log N) iterative Cooley-Tukey radix-2 DIT FFT for power-of-two lengths.
//! - `bluestein`: O(N log N) chirp-Z FFT for arbitrary lengths using radix-2 internally.
//! - `radix4`: radix-4 strategy facade for power-of-four lengths.
//! - `radix8`: radix-8 strategy facade for power-of-eight lengths.
//! - `radix16`: radix-16 strategy facade for power-of-sixteen lengths.
//! - `radix32`: radix-32 strategy facade for power-of-thirty-two lengths.
//! - `radix64`: radix-64 strategy facade for power-of-sixty-four lengths.
//! - `mixed_radix`: mixed strategy router for power-of-two lengths.
//!
//! The functions below auto-select radix-2 for power-of-2 sizes and
//! Bluestein otherwise, providing a single authoritative entry point for all
//! higher-level plan code.

pub mod bluestein;
pub mod direct;
pub(crate) mod f16_bridge;
pub(crate) mod kernel_api;
pub mod mixed_radix;
pub mod radix16;
pub mod radix2;
pub mod radix2_f16;
pub mod radix32;
pub mod radix4;
pub mod radix64;
pub mod radix8;
pub(crate) mod radix_permute;
pub(crate) mod radix_shape;
pub(crate) mod radix_stage;
pub(crate) mod tuning;
pub(crate) mod twiddle_table;
pub mod winograd;

#[cfg(test)]
pub(crate) mod test_utils;

pub use direct::{
    dft_forward_32, dft_forward_64, dft_inverse_32, dft_inverse_64, forward_owned_64,
    inverse_owned_64, KernelScalar,
};
pub use radix2_f16::Cf16;

use num_complex::{Complex32, Complex64};

// ── Precision-generic auto-selecting API ─────────────────────────────────────

/// Precision-generic auto-selecting FFT operations.
///
/// Implementors delegate to the `mixed_radix` facade, which chooses the best
/// available radix kernel for the given length and type at compile time.
/// Implemented for `Complex64`, `Complex32`, and `Cf16`.
pub trait FftPrecision: Sized {
    /// In-place forward FFT, unnormalized.
    fn fft_forward(data: &mut [Self]);
    /// In-place inverse FFT, normalized by 1/N.
    fn fft_inverse(data: &mut [Self]);
    /// In-place inverse FFT, unnormalized (no 1/N division).
    ///
    /// Use this when normalization is deferred to a single outer call
    /// (e.g., separable multi-dimensional transforms).
    fn fft_inverse_unnorm(data: &mut [Self]);
}

/// Unified auto-selecting forward FFT entry point across all supported precisions.
#[inline]
pub fn fft_forward<C: FftPrecision>(data: &mut [C]) {
    C::fft_forward(data);
}

/// Unified auto-selecting inverse FFT entry point (normalized by 1/N).
#[inline]
pub fn fft_inverse<C: FftPrecision>(data: &mut [C]) {
    C::fft_inverse(data);
}

/// Unified auto-selecting inverse FFT entry point (unnormalized).
#[inline]
pub fn fft_inverse_unnorm<C: FftPrecision>(data: &mut [C]) {
    C::fft_inverse_unnorm(data);
}

// ── Free-function entry points (backward-compatible) ─────────────────────────
//
// Each function is a thin shim over `mixed_radix`, which already handles the
// full radix dispatch (power-of-8 → radix8, power-of-4 → radix4, PoT →
// radix2, else → Bluestein).  The dispatch logic lives in exactly one place.

/// Auto-selecting forward FFT (unnormalized, f64).
#[inline]
pub fn fft_forward_64(data: &mut [Complex64]) {
    mixed_radix::forward_inplace_64(data);
}

/// Auto-selecting inverse FFT, normalized by 1/N (f64).
#[inline]
pub fn fft_inverse_64(data: &mut [Complex64]) {
    mixed_radix::inverse_inplace_64(data);
}

/// Auto-selecting inverse FFT, unnormalized (f64).
///
/// Required for nD separable axis passes where normalization is applied once externally.
#[inline]
pub fn fft_inverse_unnorm_64(data: &mut [Complex64]) {
    mixed_radix::inverse_inplace_unnorm_64(data);
}

/// Auto-selecting forward FFT (unnormalized, f32).
#[inline]
pub fn fft_forward_32(data: &mut [Complex32]) {
    mixed_radix::forward_inplace_32(data);
}

/// Auto-selecting inverse FFT, normalized by 1/N (f32).
#[inline]
pub fn fft_inverse_32(data: &mut [Complex32]) {
    mixed_radix::inverse_inplace_32(data);
}

/// Auto-selecting inverse FFT, unnormalized (f32).
#[inline]
pub fn fft_inverse_unnorm_32(data: &mut [Complex32]) {
    mixed_radix::inverse_inplace_unnorm_32(data);
}

/// Auto-selecting forward FFT over `Cf16` (f16 storage, mixed-precision arithmetic).
///
/// Power-of-two lengths use the native f16 SIMD/scalar radix kernel.
/// Non-PoT lengths promote to f32, run Bluestein-f32, then demote.
#[inline]
pub fn fft_forward_f16(data: &mut [Cf16]) {
    mixed_radix::forward_inplace_f16(data);
}

/// Auto-selecting inverse FFT over `Cf16`, normalized by 1/N.
#[inline]
pub fn fft_inverse_f16(data: &mut [Cf16]) {
    mixed_radix::inverse_inplace_f16(data);
}

/// Auto-selecting inverse FFT over `Cf16`, unnormalized.
#[inline]
pub fn fft_inverse_unnorm_f16(data: &mut [Cf16]) {
    mixed_radix::inverse_inplace_unnorm_f16(data);
}

// ── FftPrecision implementations ─────────────────────────────────────────────

impl FftPrecision for Complex64 {
    #[inline]
    fn fft_forward(data: &mut [Self]) {
        fft_forward_64(data);
    }
    #[inline]
    fn fft_inverse(data: &mut [Self]) {
        fft_inverse_64(data);
    }
    #[inline]
    fn fft_inverse_unnorm(data: &mut [Self]) {
        fft_inverse_unnorm_64(data);
    }
}

impl FftPrecision for Complex32 {
    #[inline]
    fn fft_forward(data: &mut [Self]) {
        fft_forward_32(data);
    }
    #[inline]
    fn fft_inverse(data: &mut [Self]) {
        fft_inverse_32(data);
    }
    #[inline]
    fn fft_inverse_unnorm(data: &mut [Self]) {
        fft_inverse_unnorm_32(data);
    }
}

impl FftPrecision for Cf16 {
    #[inline]
    fn fft_forward(data: &mut [Self]) {
        fft_forward_f16(data);
    }
    #[inline]
    fn fft_inverse(data: &mut [Self]) {
        fft_inverse_f16(data);
    }
    #[inline]
    fn fft_inverse_unnorm(data: &mut [Self]) {
        fft_inverse_unnorm_f16(data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward_32, dft_forward_64};
    use crate::application::execution::kernel::test_utils::{max_abs_err_32, max_abs_err_64};

    fn sig64(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|k| {
                let t = k as f64;
                Complex64::new((0.27 * t).sin(), 0.35 * (0.11 * t).cos())
            })
            .collect()
    }

    fn sig32(n: usize) -> Vec<Complex32> {
        (0..n)
            .map(|k| {
                let t = k as f32;
                Complex32::new((0.27_f32 * t).sin(), 0.35_f32 * (0.11_f32 * t).cos())
            })
            .collect()
    }

    fn max_abs_err_f16(got: &[Cf16], expected: &[Cf16]) -> f32 {
        got.iter()
            .zip(expected.iter())
            .map(|(x, y)| {
                let (xr, xi) = x.to_f32_pair();
                let (yr, yi) = y.to_f32_pair();
                let dr = xr - yr;
                let di = xi - yi;
                (dr * dr + di * di).sqrt()
            })
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn unified_api_forward_64_matches_direct_and_typed() {
        let n = 45usize;
        let input = sig64(n);

        let mut generic = input.clone();
        fft_forward(&mut generic);

        let mut typed = input.clone();
        fft_forward_64(&mut typed);

        let direct = dft_forward_64(&input);
        assert!(max_abs_err_64(&generic, &typed) < 1e-12);
        assert!(max_abs_err_64(&generic, &direct) < 1e-10);
    }

    #[test]
    fn unified_api_forward_32_matches_direct_and_typed() {
        let n = 45usize;
        let input = sig32(n);

        let mut generic = input.clone();
        fft_forward(&mut generic);

        let mut typed = input.clone();
        fft_forward_32(&mut typed);

        let direct = dft_forward_32(&input);
        assert!(max_abs_err_32(&generic, &typed) < 1e-6);
        assert!(max_abs_err_32(&generic, &direct) < 5e-4);
    }

    #[test]
    fn unified_api_forward_f16_matches_typed() {
        let n = 45usize;
        let input: Vec<Cf16> = sig32(n)
            .into_iter()
            .map(|c| Cf16::from_f32_pair(c.re, c.im))
            .collect();

        let mut generic = input.clone();
        fft_forward(&mut generic);

        let mut typed = input;
        fft_forward_f16(&mut typed);

        assert!(max_abs_err_f16(&generic, &typed) < 2e-3);
    }
}
