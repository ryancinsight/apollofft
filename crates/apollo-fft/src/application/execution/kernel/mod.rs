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
pub mod mixed_radix;
pub mod radix2;
pub mod radix16;
pub mod radix32;
pub mod radix4;
pub mod radix64;
pub mod radix8;

pub use direct::{
    dft_forward_32, dft_forward_64, dft_inverse_32, dft_inverse_64, forward_owned_64,
    inverse_owned_64, KernelScalar,
};

use num_complex::{Complex32, Complex64};

#[inline]
fn is_power_of_four(n: usize) -> bool {
    n.is_power_of_two() && (n.trailing_zeros() % 2 == 0)
}

#[inline]
fn is_power_of_eight(n: usize) -> bool {
    n.is_power_of_two() && (n.trailing_zeros() % 3 == 0)
}

/// Auto-selecting forward FFT (unnormalized).
/// Uses radix-2 for power-of-2 sizes, Bluestein otherwise.
#[inline]
pub fn fft_forward_64(data: &mut [Complex64]) {
    if is_power_of_eight(data.len()) {
        radix8::forward_inplace_64(data);
    } else if is_power_of_four(data.len()) {
        mixed_radix::forward_inplace_64(data);
    } else if data.len().is_power_of_two() {
        radix2::forward_inplace_64(data);
    } else {
        bluestein::forward_inplace_64(data);
    }
}

/// Auto-selecting inverse FFT, normalized by 1/N.
#[inline]
pub fn fft_inverse_64(data: &mut [Complex64]) {
    if is_power_of_eight(data.len()) {
        radix8::inverse_inplace_64(data);
    } else if is_power_of_four(data.len()) {
        mixed_radix::inverse_inplace_64(data);
    } else if data.len().is_power_of_two() {
        radix2::inverse_inplace_64(data);
    } else {
        bluestein::inverse_inplace_64(data);
    }
}

/// Auto-selecting inverse FFT, unnormalized (no 1/N division).
/// Required for nD separable axis passes where normalization is applied once externally.
#[inline]
pub fn fft_inverse_unnorm_64(data: &mut [Complex64]) {
    if is_power_of_eight(data.len()) {
        radix8::inverse_inplace_unnorm_64(data);
    } else if is_power_of_four(data.len()) {
        mixed_radix::inverse_inplace_unnorm_64(data);
    } else if data.len().is_power_of_two() {
        radix2::inverse_inplace_unnorm_64(data);
    } else {
        bluestein::inverse_inplace_unnorm_64(data);
    }
}

/// Auto-selecting forward FFT f32 (unnormalized).
#[inline]
pub fn fft_forward_32(data: &mut [Complex32]) {
    if is_power_of_eight(data.len()) {
        radix8::forward_inplace_32(data);
    } else if is_power_of_four(data.len()) {
        mixed_radix::forward_inplace_32(data);
    } else if data.len().is_power_of_two() {
        radix2::forward_inplace_32(data);
    } else {
        bluestein::forward_inplace_32(data);
    }
}

/// Auto-selecting inverse FFT f32, normalized by 1/N.
#[inline]
pub fn fft_inverse_32(data: &mut [Complex32]) {
    if is_power_of_eight(data.len()) {
        radix8::inverse_inplace_32(data);
    } else if is_power_of_four(data.len()) {
        mixed_radix::inverse_inplace_32(data);
    } else if data.len().is_power_of_two() {
        radix2::inverse_inplace_32(data);
    } else {
        bluestein::inverse_inplace_32(data);
    }
}

/// Auto-selecting inverse FFT f32, unnormalized.
#[inline]
pub fn fft_inverse_unnorm_32(data: &mut [Complex32]) {
    if is_power_of_eight(data.len()) {
        radix8::inverse_inplace_unnorm_32(data);
    } else if is_power_of_four(data.len()) {
        mixed_radix::inverse_inplace_unnorm_32(data);
    } else if data.len().is_power_of_two() {
        radix2::inverse_inplace_unnorm_32(data);
    } else {
        bluestein::inverse_inplace_unnorm_32(data);
    }
}
