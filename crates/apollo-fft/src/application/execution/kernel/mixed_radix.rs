//! Mixed-radix strategy facade.
//!
//! The previous recursive implementation allocated temporary vectors at every
//! recursion level. This module now keeps the mixed-radix strategy contract
//! while aligning with Apollo's in-place, twiddle-specialized architecture:
//! - power-of-eight lengths route to the radix8 facade,
//! - power-of-four lengths route to the radix4 facade,
//! - other power-of-two lengths route to radix2,
//! - non-power-of-two lengths route to Bluestein.
//!
//! Optional *_with_twiddles entry points allow callers with precomputed tables
//! to avoid per-call twiddle construction.

use super::{bluestein, radix2, radix4, radix8};
use num_complex::{Complex32, Complex64};

#[inline]
fn is_power_of_four(n: usize) -> bool {
    n.is_power_of_two() && (n.trailing_zeros() % 2 == 0)
}

#[inline]
fn is_power_of_eight(n: usize) -> bool {
    n.is_power_of_two() && (n.trailing_zeros() % 3 == 0)
}

/// In-place forward FFT (unnormalized, f64) with optional precomputed twiddles.
#[inline]
pub fn forward_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: Option<&[Complex64]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        if is_power_of_eight(data.len()) {
            if let Some(tw) = twiddles {
                radix8::forward_inplace_64_with_twiddles(data, tw);
            } else {
                radix8::forward_inplace_64(data);
            }
        } else if is_power_of_four(data.len()) {
            if let Some(tw) = twiddles {
                radix4::forward_inplace_64_with_twiddles(data, tw);
            } else {
                radix4::forward_inplace_64(data);
            }
        } else if let Some(tw) = twiddles {
            radix2::forward_inplace_64_with_twiddles(data, tw);
        } else {
            radix2::forward_inplace_64(data);
        }
    } else {
        bluestein::forward_inplace_64(data);
    }
}

/// In-place inverse FFT (unnormalized, f64) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_unnorm_64_with_twiddles(
    data: &mut [Complex64],
    twiddles: Option<&[Complex64]>,
) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        if is_power_of_eight(data.len()) {
            if let Some(tw) = twiddles {
                radix8::inverse_inplace_unnorm_64_with_twiddles(data, tw);
            } else {
                radix8::inverse_inplace_unnorm_64(data);
            }
        } else if is_power_of_four(data.len()) {
            if let Some(tw) = twiddles {
                radix4::inverse_inplace_unnorm_64_with_twiddles(data, tw);
            } else {
                radix4::inverse_inplace_unnorm_64(data);
            }
        } else if let Some(tw) = twiddles {
            radix2::inverse_inplace_unnorm_64_with_twiddles(data, tw);
        } else {
            radix2::inverse_inplace_unnorm_64(data);
        }
    } else {
        bluestein::inverse_inplace_unnorm_64(data);
    }
}

/// In-place inverse FFT normalized by 1/N (f64) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: Option<&[Complex64]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        if is_power_of_eight(data.len()) {
            if let Some(tw) = twiddles {
                radix8::inverse_inplace_64_with_twiddles(data, tw);
            } else {
                radix8::inverse_inplace_64(data);
            }
        } else if is_power_of_four(data.len()) {
            if let Some(tw) = twiddles {
                radix4::inverse_inplace_64_with_twiddles(data, tw);
            } else {
                radix4::inverse_inplace_64(data);
            }
        } else if let Some(tw) = twiddles {
            radix2::inverse_inplace_64_with_twiddles(data, tw);
        } else {
            radix2::inverse_inplace_64(data);
        }
    } else {
        bluestein::inverse_inplace_64(data);
    }
}

/// In-place forward FFT (unnormalized, f64).
pub fn forward_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let twiddles = radix2::build_forward_twiddle_table_64(data.len());
        forward_inplace_64_with_twiddles(data, Some(&twiddles));
    } else {
        forward_inplace_64_with_twiddles(data, None);
    }
}

/// In-place inverse FFT (unnormalized, f64).
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let twiddles = radix2::build_inverse_twiddle_table_64(data.len());
        inverse_inplace_unnorm_64_with_twiddles(data, Some(&twiddles));
    } else {
        inverse_inplace_unnorm_64_with_twiddles(data, None);
    }
}

/// In-place inverse FFT normalized by 1/N (f64).
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let twiddles = radix2::build_inverse_twiddle_table_64(data.len());
        inverse_inplace_64_with_twiddles(data, Some(&twiddles));
    } else {
        inverse_inplace_64_with_twiddles(data, None);
    }
}

/// In-place forward FFT (unnormalized, f32) with optional precomputed twiddles.
#[inline]
pub fn forward_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: Option<&[Complex32]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        if is_power_of_eight(data.len()) {
            if let Some(tw) = twiddles {
                radix8::forward_inplace_32_with_twiddles(data, tw);
            } else {
                radix8::forward_inplace_32(data);
            }
        } else if is_power_of_four(data.len()) {
            if let Some(tw) = twiddles {
                radix4::forward_inplace_32_with_twiddles(data, tw);
            } else {
                radix4::forward_inplace_32(data);
            }
        } else if let Some(tw) = twiddles {
            radix2::forward_inplace_32_with_twiddles(data, tw);
        } else {
            radix2::forward_inplace_32(data);
        }
    } else {
        bluestein::forward_inplace_32(data);
    }
}

/// In-place inverse FFT (unnormalized, f32) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_unnorm_32_with_twiddles(
    data: &mut [Complex32],
    twiddles: Option<&[Complex32]>,
) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        if is_power_of_eight(data.len()) {
            if let Some(tw) = twiddles {
                radix8::inverse_inplace_unnorm_32_with_twiddles(data, tw);
            } else {
                radix8::inverse_inplace_unnorm_32(data);
            }
        } else if is_power_of_four(data.len()) {
            if let Some(tw) = twiddles {
                radix4::inverse_inplace_unnorm_32_with_twiddles(data, tw);
            } else {
                radix4::inverse_inplace_unnorm_32(data);
            }
        } else if let Some(tw) = twiddles {
            radix2::inverse_inplace_unnorm_32_with_twiddles(data, tw);
        } else {
            radix2::inverse_inplace_unnorm_32(data);
        }
    } else {
        bluestein::inverse_inplace_unnorm_32(data);
    }
}

/// In-place inverse FFT normalized by 1/N (f32) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: Option<&[Complex32]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        if is_power_of_eight(data.len()) {
            if let Some(tw) = twiddles {
                radix8::inverse_inplace_32_with_twiddles(data, tw);
            } else {
                radix8::inverse_inplace_32(data);
            }
        } else if is_power_of_four(data.len()) {
            if let Some(tw) = twiddles {
                radix4::inverse_inplace_32_with_twiddles(data, tw);
            } else {
                radix4::inverse_inplace_32(data);
            }
        } else if let Some(tw) = twiddles {
            radix2::inverse_inplace_32_with_twiddles(data, tw);
        } else {
            radix2::inverse_inplace_32(data);
        }
    } else {
        bluestein::inverse_inplace_32(data);
    }
}

/// In-place forward FFT (unnormalized, f32).
pub fn forward_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let twiddles = radix2::build_forward_twiddle_table_32(data.len());
        forward_inplace_32_with_twiddles(data, Some(&twiddles));
    } else {
        forward_inplace_32_with_twiddles(data, None);
    }
}

/// In-place inverse FFT (unnormalized, f32).
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let twiddles = radix2::build_inverse_twiddle_table_32(data.len());
        inverse_inplace_unnorm_32_with_twiddles(data, Some(&twiddles));
    } else {
        inverse_inplace_unnorm_32_with_twiddles(data, None);
    }
}

/// In-place inverse FFT normalized by 1/N (f32).
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let twiddles = radix2::build_inverse_twiddle_table_32(data.len());
        inverse_inplace_32_with_twiddles(data, Some(&twiddles));
    } else {
        inverse_inplace_32_with_twiddles(data, None);
    }
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
    fn mixed_forward_n32_matches_direct() {
        let n = 32usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.17).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "mixed-radix forward mismatch err={err:.2e}");
    }

    #[test]
    fn mixed_inverse_unnorm_n32_matches_direct() {
        let n = 32usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.19).cos(), (k as f64 * 0.07).sin()))
            .collect();
        let mut got = input.clone();
        inverse_inplace_unnorm_64(&mut got);
        let expected = dft_inverse_64(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect::<Vec<_>>();
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "mixed-radix inverse mismatch err={err:.2e}");
    }
}
