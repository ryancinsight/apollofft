//! Apollo FFT kernel module.
//!
//! Provides three kernel implementations:
//! - `direct`: O(N^2) reference DFT kept for testing and validation.
//! - `radix2`: O(N log N) iterative Cooley-Tukey radix-2 DIT FFT for power-of-two lengths.
//! - `bluestein`: O(N log N) chirp-Z FFT for arbitrary lengths using radix-2 internally.
//!
//! The functions below auto-select radix-2 for power-of-2 sizes and
//! Bluestein otherwise, providing a single authoritative entry point for all
//! higher-level plan code.

pub mod bluestein;
pub mod direct;
pub mod radix2;

pub use direct::{
    dft_forward_32, dft_forward_64, dft_inverse_32, dft_inverse_64, forward_owned_64,
    inverse_owned_64, KernelScalar,
};

use num_complex::{Complex32, Complex64};

/// Auto-selecting forward FFT (unnormalized).
/// Uses radix-2 for power-of-2 sizes, Bluestein otherwise.
#[inline]
pub fn fft_forward_64(data: &mut [Complex64]) {
    if data.len().is_power_of_two() {
        radix2::forward_inplace_64(data);
    } else {
        bluestein::forward_inplace_64(data);
    }
}

/// Auto-selecting inverse FFT, normalized by 1/N.
#[inline]
pub fn fft_inverse_64(data: &mut [Complex64]) {
    if data.len().is_power_of_two() {
        radix2::inverse_inplace_64(data);
    } else {
        bluestein::inverse_inplace_64(data);
    }
}

/// Auto-selecting inverse FFT, unnormalized (no 1/N division).
/// Required for nD separable axis passes where normalization is applied once externally.
#[inline]
pub fn fft_inverse_unnorm_64(data: &mut [Complex64]) {
    if data.len().is_power_of_two() {
        radix2::inverse_inplace_unnorm_64(data);
    } else {
        bluestein::inverse_inplace_unnorm_64(data);
    }
}

/// Auto-selecting forward FFT f32 (unnormalized).
#[inline]
pub fn fft_forward_32(data: &mut [Complex32]) {
    if data.len().is_power_of_two() {
        radix2::forward_inplace_32(data);
    } else {
        bluestein::forward_inplace_32(data);
    }
}

/// Auto-selecting inverse FFT f32, normalized by 1/N.
#[inline]
pub fn fft_inverse_32(data: &mut [Complex32]) {
    if data.len().is_power_of_two() {
        radix2::inverse_inplace_32(data);
    } else {
        bluestein::inverse_inplace_32(data);
    }
}

/// Auto-selecting inverse FFT f32, unnormalized.
#[inline]
pub fn fft_inverse_unnorm_32(data: &mut [Complex32]) {
    if data.len().is_power_of_two() {
        radix2::inverse_inplace_unnorm_32(data);
    } else {
        bluestein::inverse_inplace_unnorm_32(data);
    }
}
