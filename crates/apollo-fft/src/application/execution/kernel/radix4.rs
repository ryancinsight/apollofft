//! True radix-4 Cooley-Tukey kernels.
//!
//! This module implements in-place radix-4 DIT transforms for power-of-four
//! lengths. The radix-4 modules for 16/32/64 can then build higher-radix
//! behavior on top of this kernel family without routing through radix-2.

use super::radix2;
use num_complex::{Complex32, Complex64};

#[inline]
fn is_power_of_four(n: usize) -> bool {
    n.is_power_of_two() && (n.trailing_zeros() % 2 == 0)
}

/// Forward FFT (unnormalized) for power-of-four lengths using caller-provided twiddles.
#[inline]
pub fn forward_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(data.len()));
    radix4_inplace_64(data, twiddles, false);
}

/// Inverse FFT (unnormalized) for power-of-four lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_unnorm_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(data.len()));
    radix4_inplace_64(data, twiddles, true);
}

/// Inverse FFT normalized by 1/N for power-of-four lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(data.len()));
    radix4_inplace_64(data, twiddles, true);
    let inv_n = 1.0 / data.len() as f64;
    for value in data.iter_mut() {
        *value *= inv_n;
    }
}

/// Forward FFT (unnormalized) for power-of-four lengths.
pub fn forward_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(data.len()));
    let twiddles = radix2::build_forward_twiddle_table_64(data.len());
    radix4_inplace_64(data, &twiddles, false);
}

/// Inverse FFT (unnormalized) for power-of-four lengths.
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_64(data.len());
    radix4_inplace_64(data, &twiddles, true);
}

/// Inverse FFT normalized by 1/N for power-of-four lengths.
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_64(data.len());
    radix4_inplace_64(data, &twiddles, true);
    let inv_n = 1.0 / data.len() as f64;
    for value in data.iter_mut() {
        *value *= inv_n;
    }
}

/// Forward FFT (unnormalized, f32) for power-of-four lengths using caller-provided twiddles.
#[inline]
pub fn forward_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(data.len()));
    radix4_inplace_32(data, twiddles, false);
}

/// Inverse FFT (unnormalized, f32) for power-of-four lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_unnorm_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(data.len()));
    radix4_inplace_32(data, twiddles, true);
}

/// Inverse FFT normalized by 1/N (f32) for power-of-four lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(data.len()));
    radix4_inplace_32(data, twiddles, true);
    let inv_n = 1.0f32 / data.len() as f32;
    for value in data.iter_mut() {
        *value *= inv_n;
    }
}

/// Forward FFT (unnormalized, f32) for power-of-four lengths.
pub fn forward_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(data.len()));
    let twiddles = radix2::build_forward_twiddle_table_32(data.len());
    radix4_inplace_32(data, &twiddles, false);
}

/// Inverse FFT (unnormalized, f32) for power-of-four lengths.
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_32(data.len());
    radix4_inplace_32(data, &twiddles, true);
}

/// Inverse FFT normalized by 1/N (f32) for power-of-four lengths.
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_32(data.len());
    radix4_inplace_32(data, &twiddles, true);
    let inv_n = 1.0f32 / data.len() as f32;
    for value in data.iter_mut() {
        *value *= inv_n;
    }
}

#[inline]
fn cmul_64(a: Complex64, b: Complex64) -> Complex64 {
    Complex64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

#[inline]
fn cmul_32(a: Complex32, b: Complex32) -> Complex32 {
    Complex32::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

fn reverse_base4(mut value: usize, digits: u32) -> usize {
    let mut reversed = 0usize;
    for _ in 0..digits {
        reversed = (reversed << 2) | (value & 0b11);
        value >>= 2;
    }
    reversed
}

fn digit_reverse_permute_64(data: &mut [Complex64]) {
    let digits = data.len().trailing_zeros() / 2;
    for index in 0..data.len() {
        let reversed = reverse_base4(index, digits);
        if reversed > index {
            data.swap(index, reversed);
        }
    }
}

fn digit_reverse_permute_32(data: &mut [Complex32]) {
    let digits = data.len().trailing_zeros() / 2;
    for index in 0..data.len() {
        let reversed = reverse_base4(index, digits);
        if reversed > index {
            data.swap(index, reversed);
        }
    }
}

#[inline]
fn stage_twiddle_64(stage: &[Complex64], half: usize, exponent: usize) -> Complex64 {
    if exponent < half {
        stage[exponent]
    } else {
        -stage[exponent - half]
    }
}

#[inline]
fn stage_twiddle_32(stage: &[Complex32], half: usize, exponent: usize) -> Complex32 {
    if exponent < half {
        stage[exponent]
    } else {
        -stage[exponent - half]
    }
}

fn radix4_inplace_64(data: &mut [Complex64], twiddles: &[Complex64], inverse: bool) {
    debug_assert!(is_power_of_four(data.len()));
    if data.len() <= 1 {
        return;
    }

    digit_reverse_permute_64(data);

    let n = data.len();
    let mut len = 4usize;
    while len <= n {
        let quarter = len >> 2;
        let half = len >> 1;
        let stage = if len > 4 {
            Some(&twiddles[(half - 1)..(half - 1 + half)])
        } else {
            None
        };

        for chunk in data.chunks_exact_mut(len) {
            for j in 0..quarter {
                let i0 = j;
                let i1 = i0 + quarter;
                let i2 = i1 + quarter;
                let i3 = i2 + quarter;

                let a0 = chunk[i0];
                let mut a1 = chunk[i1];
                let mut a2 = chunk[i2];
                let mut a3 = chunk[i3];

                if let Some(stage_twiddles) = stage {
                    let w1 = stage_twiddle_64(stage_twiddles, half, j);
                    let w2 = stage_twiddle_64(stage_twiddles, half, 2 * j);
                    let w3 = stage_twiddle_64(stage_twiddles, half, 3 * j);
                    a1 = cmul_64(a1, w1);
                    a2 = cmul_64(a2, w2);
                    a3 = cmul_64(a3, w3);
                }

                let t0 = a0 + a2;
                let t1 = a0 - a2;
                let t2 = a1 + a3;
                let t3 = a1 - a3;

                let y0 = t0 + t2;
                let y2 = t0 - t2;

                let (y1, y3) = if inverse {
                    (
                        Complex64::new(t1.re - t3.im, t1.im + t3.re),
                        Complex64::new(t1.re + t3.im, t1.im - t3.re),
                    )
                } else {
                    (
                        Complex64::new(t1.re + t3.im, t1.im - t3.re),
                        Complex64::new(t1.re - t3.im, t1.im + t3.re),
                    )
                };

                chunk[i0] = y0;
                chunk[i1] = y1;
                chunk[i2] = y2;
                chunk[i3] = y3;
            }
        }

        len <<= 2;
    }
}

fn radix4_inplace_32(data: &mut [Complex32], twiddles: &[Complex32], inverse: bool) {
    debug_assert!(is_power_of_four(data.len()));
    if data.len() <= 1 {
        return;
    }

    digit_reverse_permute_32(data);

    let n = data.len();
    let mut len = 4usize;
    while len <= n {
        let quarter = len >> 2;
        let half = len >> 1;
        let stage = if len > 4 {
            Some(&twiddles[(half - 1)..(half - 1 + half)])
        } else {
            None
        };

        for chunk in data.chunks_exact_mut(len) {
            for j in 0..quarter {
                let i0 = j;
                let i1 = i0 + quarter;
                let i2 = i1 + quarter;
                let i3 = i2 + quarter;

                let a0 = chunk[i0];
                let mut a1 = chunk[i1];
                let mut a2 = chunk[i2];
                let mut a3 = chunk[i3];

                if let Some(stage_twiddles) = stage {
                    let w1 = stage_twiddle_32(stage_twiddles, half, j);
                    let w2 = stage_twiddle_32(stage_twiddles, half, 2 * j);
                    let w3 = stage_twiddle_32(stage_twiddles, half, 3 * j);
                    a1 = cmul_32(a1, w1);
                    a2 = cmul_32(a2, w2);
                    a3 = cmul_32(a3, w3);
                }

                let t0 = a0 + a2;
                let t1 = a0 - a2;
                let t2 = a1 + a3;
                let t3 = a1 - a3;

                let y0 = t0 + t2;
                let y2 = t0 - t2;

                let (y1, y3) = if inverse {
                    (
                        Complex32::new(t1.re - t3.im, t1.im + t3.re),
                        Complex32::new(t1.re + t3.im, t1.im - t3.re),
                    )
                } else {
                    (
                        Complex32::new(t1.re + t3.im, t1.im - t3.re),
                        Complex32::new(t1.re - t3.im, t1.im + t3.re),
                    )
                };

                chunk[i0] = y0;
                chunk[i1] = y1;
                chunk[i2] = y2;
                chunk[i3] = y3;
            }
        }

        len <<= 2;
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
    fn radix4_forward_n16_matches_direct() {
        let n = 16usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.3).sin(), (k as f64 * 0.11).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "radix4 forward mismatch err={err:.2e}");
    }

    #[test]
    fn radix4_inverse_unnorm_n16_matches_direct() {
        let n = 16usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.27).cos(), (k as f64 * 0.17).sin()))
            .collect();
        let mut got = input.clone();
        inverse_inplace_unnorm_64(&mut got);
        let expected = dft_inverse_64(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect::<Vec<_>>();
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "radix4 inverse mismatch err={err:.2e}");
    }
}
