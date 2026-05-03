//! Radix-8 strategy facade aligned with Apollo's twiddle-specialized in-place kernels.
//!
//! This module exposes radix-8 eligibility plus API-compatible forward/inverse
//! entry points for power-of-eight lengths. Execution delegates to the
//! optimized radix-2 kernels, preserving contiguous twiddle access and in-place
//! butterfly execution with zero extra data allocations per call.

use super::radix2;
use num_complex::{Complex32, Complex64};

#[inline]
fn is_power_of_eight(n: usize) -> bool {
    n.is_power_of_two() && (n.trailing_zeros() % 3 == 0)
}

/// Forward FFT (unnormalized) for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn forward_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    radix2::forward_inplace_64_with_twiddles(data, twiddles);
}

/// Inverse FFT (unnormalized) for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_unnorm_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    radix2::inverse_inplace_unnorm_64_with_twiddles(data, twiddles);
}

/// Inverse FFT normalized by 1/N for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    radix2::inverse_inplace_64_with_twiddles(data, twiddles);
}

/// Forward FFT (unnormalized) for power-of-eight lengths.
pub fn forward_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_forward_twiddle_table_64(data.len());
    forward_inplace_64_with_twiddles(data, &twiddles);
}

/// Inverse FFT (unnormalized) for power-of-eight lengths.
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_64(data.len());
    inverse_inplace_unnorm_64_with_twiddles(data, &twiddles);
}

/// Inverse FFT normalized by 1/N for power-of-eight lengths.
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_64(data.len());
    inverse_inplace_64_with_twiddles(data, &twiddles);
}

/// Forward FFT (unnormalized, f32) for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn forward_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    radix2::forward_inplace_32_with_twiddles(data, twiddles);
}

/// Inverse FFT (unnormalized, f32) for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_unnorm_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    radix2::inverse_inplace_unnorm_32_with_twiddles(data, twiddles);
}

/// Inverse FFT normalized by 1/N (f32) for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    radix2::inverse_inplace_32_with_twiddles(data, twiddles);
}

/// Forward FFT (unnormalized, f32) for power-of-eight lengths.
pub fn forward_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_forward_twiddle_table_32(data.len());
    forward_inplace_32_with_twiddles(data, &twiddles);
}

/// Inverse FFT (unnormalized, f32) for power-of-eight lengths.
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_32(data.len());
    inverse_inplace_unnorm_32_with_twiddles(data, &twiddles);
}

/// Inverse FFT normalized by 1/N (f32) for power-of-eight lengths.
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_32(data.len());
    inverse_inplace_32_with_twiddles(data, &twiddles);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward_64, dft_inverse_64};

    fn max_abs_err_64(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).norm())
            .fold(0.0, f64::max)
    }

    #[test]
    fn radix8_forward_n64_matches_direct() {
        let n = 64usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.13).sin(), (k as f64 * 0.09).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "radix8 forward mismatch err={err:.2e}");
    }

    #[test]
    fn radix8_inverse_unnorm_n64_matches_direct() {
        let n = 64usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.07).cos(), (k as f64 * 0.11).sin()))
            .collect();
        let mut got = input.clone();
        inverse_inplace_unnorm_64(&mut got);
        let expected = dft_inverse_64(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect::<Vec<_>>();
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "radix8 inverse mismatch err={err:.2e}");
    }
}
