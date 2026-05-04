//! True radix-16 Cooley-Tukey kernels using Winograd DFT-16 inner butterflies.
//!
//! ## Algorithm
//!
//! Each stage processes groups of 16 elements using the Winograd DFT-16
//! kernel, which recursively decomposes via 2×DFT-8 (each 2×DFT-4 with
//! ±√2/2 twiddles) to reduce the inner butterfly from 256 generic
//! multiplications to 8 real multiplications per group.
//!
//! ## References
//!
//! - Winograd, S. (1978). On computing the discrete Fourier transform.
//!   *Mathematics of Computation*, 32(141), 175–199.

use super::{radix2, winograd};
use num_complex::{Complex32, Complex64};

#[inline]
fn is_power_of_sixteen(n: usize) -> bool {
    n.is_power_of_two() && (n.trailing_zeros() % 4 == 0)
}

#[inline]
/// Forward FFT (unnormalized) for power-of-sixteen lengths using caller-provided twiddles.
pub fn forward_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    radix16_inplace_64(data, false, Some(twiddles));
}

#[inline]
/// Inverse FFT (unnormalized) for power-of-sixteen lengths using caller-provided twiddles.
pub fn inverse_inplace_unnorm_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    radix16_inplace_64(data, true, Some(twiddles));
}

#[inline]
/// Inverse FFT normalized by 1/N for power-of-sixteen lengths using caller-provided twiddles.
pub fn inverse_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    radix16_inplace_64(data, true, Some(twiddles));
    let inv_n = 1.0 / data.len() as f64;
    for value in data.iter_mut() {
        *value *= inv_n;
    }
}

/// Forward FFT (unnormalized) for power-of-sixteen lengths.
pub fn forward_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    let twiddles = radix2::build_forward_twiddle_table_64(data.len());
    radix16_inplace_64(data, false, Some(&twiddles));
}

/// Inverse FFT (unnormalized) for power-of-sixteen lengths.
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_64(data.len());
    radix16_inplace_64(data, true, Some(&twiddles));
}

/// Inverse FFT normalized by 1/N for power-of-sixteen lengths.
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_64(data.len());
    radix16_inplace_64(data, true, Some(&twiddles));
    let inv_n = 1.0 / data.len() as f64;
    for value in data.iter_mut() {
        *value *= inv_n;
    }
}

#[inline]
/// Forward FFT (unnormalized, f32) for power-of-sixteen lengths using caller-provided twiddles.
pub fn forward_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    radix16_inplace_32(data, false, Some(twiddles));
}

#[inline]
/// Inverse FFT (unnormalized, f32) for power-of-sixteen lengths using caller-provided twiddles.
pub fn inverse_inplace_unnorm_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    radix16_inplace_32(data, true, Some(twiddles));
}

#[inline]
/// Inverse FFT normalized by 1/N (f32) for power-of-sixteen lengths using caller-provided twiddles.
pub fn inverse_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    radix16_inplace_32(data, true, Some(twiddles));
    let inv_n = 1.0f32 / data.len() as f32;
    for value in data.iter_mut() {
        *value *= inv_n;
    }
}

/// Forward FFT (unnormalized, f32) for power-of-sixteen lengths.
pub fn forward_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    let twiddles = radix2::build_forward_twiddle_table_32(data.len());
    radix16_inplace_32(data, false, Some(&twiddles));
}

/// Inverse FFT (unnormalized, f32) for power-of-sixteen lengths.
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_32(data.len());
    radix16_inplace_32(data, true, Some(&twiddles));
}

/// Inverse FFT normalized by 1/N (f32) for power-of-sixteen lengths.
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_32(data.len());
    radix16_inplace_32(data, true, Some(&twiddles));
    let inv_n = 1.0f32 / data.len() as f32;
    for value in data.iter_mut() {
        *value *= inv_n;
    }
}

fn reverse_base_radix(mut value: usize, radix: usize, digits: u32) -> usize {
    let mut reversed = 0usize;
    for _ in 0..digits {
        reversed = reversed * radix + (value % radix);
        value /= radix;
    }
    reversed
}

fn digit_reverse_permute_64<const RADIX: usize>(data: &mut [Complex64]) {
    let digits = (data.len().trailing_zeros() as usize) / RADIX.trailing_zeros() as usize;
    for index in 0..data.len() {
        let reversed = reverse_base_radix(index, RADIX, digits as u32);
        if reversed > index {
            data.swap(index, reversed);
        }
    }
}

fn digit_reverse_permute_32<const RADIX: usize>(data: &mut [Complex32]) {
    let digits = (data.len().trailing_zeros() as usize) / RADIX.trailing_zeros() as usize;
    for index in 0..data.len() {
        let reversed = reverse_base_radix(index, RADIX, digits as u32);
        if reversed > index {
            data.swap(index, reversed);
        }
    }
}

fn radix16_inplace_64(data: &mut [Complex64], inverse: bool, twiddles: Option<&[Complex64]>) {
    const RADIX: usize = 16;
    debug_assert!(data.len().is_power_of_two());
    debug_assert!((data.len().trailing_zeros() as usize) % 4 == 0);
    if data.len() <= 1 {
        return;
    }
    digit_reverse_permute_64::<RADIX>(data);
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut m = 1usize;
    while m < data.len() {
        let len = m * RADIX;
        let half = len >> 1;
        let stage_twiddles = twiddles.map(|t| &t[(half - 1)..(half - 1 + half)]);
        for chunk in data.chunks_exact_mut(len) {
            for j in 0..m {
                let step = if let Some(st) = stage_twiddles {
                    st[j]
                } else {
                    let a = sign * std::f64::consts::TAU * j as f64 / len as f64;
                    Complex64::new(a.cos(), a.sin())
                };
                let mut buf = [Complex64::new(0.0, 0.0); RADIX];
                buf[0] = chunk[j];
                let mut tw = step;
                for p in 1..RADIX {
                    buf[p] = winograd::apply_twiddle_64(chunk[j + p * m], tw);
                    tw = winograd::apply_twiddle_64(tw, step);
                }
                winograd::dft16_64(&mut buf, inverse);
                for p in 0..RADIX {
                    chunk[j + p * m] = buf[p];
                }
            }
        }
        m = len;
    }
}

fn radix16_inplace_32(data: &mut [Complex32], inverse: bool, twiddles: Option<&[Complex32]>) {
    const RADIX: usize = 16;
    debug_assert!(data.len().is_power_of_two());
    debug_assert!((data.len().trailing_zeros() as usize) % 4 == 0);
    if data.len() <= 1 {
        return;
    }
    digit_reverse_permute_32::<RADIX>(data);
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut m = 1usize;
    while m < data.len() {
        let len = m * RADIX;
        let half = len >> 1;
        let stage_twiddles = twiddles.map(|t| &t[(half - 1)..(half - 1 + half)]);
        for chunk in data.chunks_exact_mut(len) {
            for j in 0..m {
                let step = if let Some(st) = stage_twiddles {
                    st[j]
                } else {
                    let a = sign * std::f64::consts::TAU * j as f64 / len as f64;
                    Complex32::new(a.cos() as f32, a.sin() as f32)
                };
                let mut buf = [Complex32::new(0.0, 0.0); RADIX];
                buf[0] = chunk[j];
                let mut tw = step;
                for p in 1..RADIX {
                    buf[p] = winograd::apply_twiddle_32(chunk[j + p * m], tw);
                    tw = winograd::apply_twiddle_32(tw, step);
                }
                winograd::dft16_32(&mut buf, inverse);
                for p in 0..RADIX {
                    chunk[j + p * m] = buf[p];
                }
            }
        }
        m = len;
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
    fn radix16_forward_n16_matches_direct() {
        let n = 16usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.23).sin(), (k as f64 * 0.07).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        assert!(max_abs_err_64(&got, &expected) < 1e-10);
    }

    #[test]
    fn radix16_inverse_unnorm_n16_matches_direct() {
        let n = 16usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.15).cos(), (k as f64 * 0.11).sin()))
            .collect();
        let mut got = input.clone();
        inverse_inplace_unnorm_64(&mut got);
        let expected = dft_inverse_64(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect::<Vec<_>>();
        assert!(max_abs_err_64(&got, &expected) < 1e-10);
    }
}
