//! True radix-16 Cooley-Tukey kernels.

use super::radix2;
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
    radix_r_inplace_64::<16>(data, false, Some(twiddles));
}

#[inline]
/// Inverse FFT (unnormalized) for power-of-sixteen lengths using caller-provided twiddles.
pub fn inverse_inplace_unnorm_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    radix_r_inplace_64::<16>(data, true, Some(twiddles));
}

#[inline]
/// Inverse FFT normalized by 1/N for power-of-sixteen lengths using caller-provided twiddles.
pub fn inverse_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    radix_r_inplace_64::<16>(data, true, Some(twiddles));
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
    radix_r_inplace_64::<16>(data, false, Some(&twiddles));
}

/// Inverse FFT (unnormalized) for power-of-sixteen lengths.
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_64(data.len());
    radix_r_inplace_64::<16>(data, true, Some(&twiddles));
}

/// Inverse FFT normalized by 1/N for power-of-sixteen lengths.
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_64(data.len());
    radix_r_inplace_64::<16>(data, true, Some(&twiddles));
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
    radix_r_inplace_32::<16>(data, false, Some(twiddles));
}

#[inline]
/// Inverse FFT (unnormalized, f32) for power-of-sixteen lengths using caller-provided twiddles.
pub fn inverse_inplace_unnorm_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    radix_r_inplace_32::<16>(data, true, Some(twiddles));
}

#[inline]
/// Inverse FFT normalized by 1/N (f32) for power-of-sixteen lengths using caller-provided twiddles.
pub fn inverse_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    radix_r_inplace_32::<16>(data, true, Some(twiddles));
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
    radix_r_inplace_32::<16>(data, false, Some(&twiddles));
}

/// Inverse FFT (unnormalized, f32) for power-of-sixteen lengths.
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_32(data.len());
    radix_r_inplace_32::<16>(data, true, Some(&twiddles));
}

/// Inverse FFT normalized by 1/N (f32) for power-of-sixteen lengths.
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_sixteen(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_32(data.len());
    radix_r_inplace_32::<16>(data, true, Some(&twiddles));
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

#[inline]
fn cmul_64(a: Complex64, b: Complex64) -> Complex64 {
    Complex64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

#[inline]
fn cmul_32(a: Complex32, b: Complex32) -> Complex32 {
    Complex32::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

fn dft_matrix_64<const RADIX: usize>(sign: f64) -> [[Complex64; RADIX]; RADIX] {
    core::array::from_fn(|q| {
        core::array::from_fn(|p| {
            let angle = sign * std::f64::consts::TAU * (p * q) as f64 / RADIX as f64;
            Complex64::new(angle.cos(), angle.sin())
        })
    })
}

fn dft_matrix_32<const RADIX: usize>(sign: f64) -> [[Complex32; RADIX]; RADIX] {
    core::array::from_fn(|q| {
        core::array::from_fn(|p| {
            let angle = sign * std::f64::consts::TAU * (p * q) as f64 / RADIX as f64;
            Complex32::new(angle.cos() as f32, angle.sin() as f32)
        })
    })
}

fn radix_r_inplace_64<const RADIX: usize>(
    data: &mut [Complex64],
    inverse: bool,
    twiddles: Option<&[Complex64]>,
) {
    debug_assert!(data.len().is_power_of_two());
    debug_assert!((data.len().trailing_zeros() as usize) % RADIX.trailing_zeros() as usize == 0);
    if data.len() <= 1 {
        return;
    }

    digit_reverse_permute_64::<RADIX>(data);
    let sign = if inverse { 1.0 } else { -1.0 };
    let dft = dft_matrix_64::<RADIX>(sign);
    let mut m = 1usize;
    let mut scratch = [Complex64::new(0.0, 0.0); RADIX];
    let mut output = [Complex64::new(0.0, 0.0); RADIX];

    while m < data.len() {
        let len = m * RADIX;
        let half = len >> 1;
        let stage_twiddles = twiddles.map(|table| &table[(half - 1)..(half - 1 + half)]);
        for chunk in data.chunks_exact_mut(len) {
            for j in 0..m {
                let step = if let Some(stage) = stage_twiddles {
                    stage[j]
                } else {
                    let angle = sign * std::f64::consts::TAU * j as f64 / len as f64;
                    Complex64::new(angle.cos(), angle.sin())
                };
                let mut tw = Complex64::new(1.0, 0.0);
                for p in 0..RADIX {
                    let value = cmul_64(chunk[j + p * m], tw);
                    scratch[p] = value;
                    tw = cmul_64(tw, step);
                }

                for (q, out) in output.iter_mut().enumerate() {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for (p, value) in scratch.iter().enumerate() {
                        sum += cmul_64(*value, dft[q][p]);
                    }
                    *out = sum;
                }

                for q in 0..RADIX {
                    chunk[j + q * m] = output[q];
                }
            }
        }
        m = len;
    }
}

fn radix_r_inplace_32<const RADIX: usize>(
    data: &mut [Complex32],
    inverse: bool,
    twiddles: Option<&[Complex32]>,
) {
    debug_assert!(data.len().is_power_of_two());
    debug_assert!((data.len().trailing_zeros() as usize) % RADIX.trailing_zeros() as usize == 0);
    if data.len() <= 1 {
        return;
    }

    digit_reverse_permute_32::<RADIX>(data);
    let sign = if inverse { 1.0 } else { -1.0 };
    let dft = dft_matrix_32::<RADIX>(sign);
    let mut m = 1usize;
    let mut scratch = [Complex32::new(0.0, 0.0); RADIX];
    let mut output = [Complex32::new(0.0, 0.0); RADIX];

    while m < data.len() {
        let len = m * RADIX;
        let half = len >> 1;
        let stage_twiddles = twiddles.map(|table| &table[(half - 1)..(half - 1 + half)]);
        for chunk in data.chunks_exact_mut(len) {
            for j in 0..m {
                let step = if let Some(stage) = stage_twiddles {
                    stage[j]
                } else {
                    let angle = sign * std::f64::consts::TAU * j as f64 / len as f64;
                    Complex32::new(angle.cos() as f32, angle.sin() as f32)
                };
                let mut tw = Complex32::new(1.0, 0.0);
                for p in 0..RADIX {
                    let value = cmul_32(chunk[j + p * m], tw);
                    scratch[p] = value;
                    tw = cmul_32(tw, step);
                }

                for (q, out) in output.iter_mut().enumerate() {
                    let mut sum = Complex32::new(0.0, 0.0);
                    for (p, value) in scratch.iter().enumerate() {
                        sum += cmul_32(*value, dft[q][p]);
                    }
                    *out = sum;
                }

                for q in 0..RADIX {
                    chunk[j + q * m] = output[q];
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
