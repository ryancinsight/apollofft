//! Mixed-radix FFT kernels.
//!
//! Strategy for power-of-two lengths:
//! - Use radix-4 sub-kernels for power-of-four branches.
//! - Use one radix-2 split when needed (N = 2 * 4^m), then continue recursively.
//!
//! Non-power-of-two lengths fall back to Bluestein.

use super::{bluestein, radix4};
use num_complex::{Complex32, Complex64};

#[inline]
fn is_power_of_four(n: usize) -> bool {
    n.is_power_of_two() && (n.trailing_zeros() % 2 == 0)
}

#[inline]
fn twiddle64(n: usize, k: usize, inverse: bool) -> Complex64 {
    let s = if inverse { 1.0 } else { -1.0 };
    let a = s * std::f64::consts::TAU * (k as f64) / (n as f64);
    Complex64::new(a.cos(), a.sin())
}

#[inline]
fn twiddle32(n: usize, k: usize, inverse: bool) -> Complex32 {
    let s = if inverse { 1.0 } else { -1.0 };
    let a = s * std::f64::consts::TAU * (k as f64) / (n as f64);
    Complex32::new(a.cos() as f32, a.sin() as f32)
}

#[inline]
fn cmul64(a: Complex64, b: Complex64) -> Complex64 {
    Complex64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

#[inline]
fn cmul32(a: Complex32, b: Complex32) -> Complex32 {
    Complex32::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

fn forward_pow2_recursive_64(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }
    if is_power_of_four(n) {
        let mut v = input.to_vec();
        radix4::forward_inplace_64(&mut v);
        return v;
    }

    let m = n / 2;
    let mut even = Vec::with_capacity(m);
    let mut odd = Vec::with_capacity(m);
    for t in 0..m {
        even.push(input[2 * t]);
        odd.push(input[2 * t + 1]);
    }

    let e = forward_pow2_recursive_64(&even);
    let o = forward_pow2_recursive_64(&odd);

    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..m {
        let tw = cmul64(twiddle64(n, k, false), o[k]);
        out[k] = e[k] + tw;
        out[k + m] = e[k] - tw;
    }
    out
}

fn inverse_pow2_recursive_unnorm_64(input: &[Complex64]) -> Vec<Complex64> {
    let mut conj_in = input.to_vec();
    for x in conj_in.iter_mut() {
        *x = x.conj();
    }
    let mut out = forward_pow2_recursive_64(&conj_in);
    for x in out.iter_mut() {
        *x = x.conj();
    }
    out
}

fn forward_pow2_recursive_32(input: &[Complex32]) -> Vec<Complex32> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }
    if is_power_of_four(n) {
        let mut v = input.to_vec();
        radix4::forward_inplace_32(&mut v);
        return v;
    }

    let m = n / 2;
    let mut even = Vec::with_capacity(m);
    let mut odd = Vec::with_capacity(m);
    for t in 0..m {
        even.push(input[2 * t]);
        odd.push(input[2 * t + 1]);
    }

    let e = forward_pow2_recursive_32(&even);
    let o = forward_pow2_recursive_32(&odd);

    let mut out = vec![Complex32::new(0.0, 0.0); n];
    for k in 0..m {
        let tw = cmul32(twiddle32(n, k, false), o[k]);
        out[k] = e[k] + tw;
        out[k + m] = e[k] - tw;
    }
    out
}

fn inverse_pow2_recursive_unnorm_32(input: &[Complex32]) -> Vec<Complex32> {
    let mut conj_in = input.to_vec();
    for x in conj_in.iter_mut() {
        *x = x.conj();
    }
    let mut out = forward_pow2_recursive_32(&conj_in);
    for x in out.iter_mut() {
        *x = x.conj();
    }
    out
}

/// In-place forward FFT (unnormalized, f64).
///
/// Uses mixed radix-2/radix-4 recursion for power-of-two lengths and Bluestein
/// fallback for non-power-of-two lengths.
pub fn forward_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let out = forward_pow2_recursive_64(data);
        data.copy_from_slice(&out);
    } else {
        bluestein::forward_inplace_64(data);
    }
}

/// In-place inverse FFT (unnormalized, f64).
///
/// Uses mixed radix-2/radix-4 recursion for power-of-two lengths and Bluestein
/// fallback for non-power-of-two lengths.
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let out = inverse_pow2_recursive_unnorm_64(data);
        data.copy_from_slice(&out);
    } else {
        bluestein::inverse_inplace_unnorm_64(data);
    }
}

/// In-place inverse FFT normalized by 1/N (f64).
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    inverse_inplace_unnorm_64(data);
    let scale = 1.0 / data.len() as f64;
    for x in data.iter_mut() {
        *x *= scale;
    }
}

/// In-place forward FFT (unnormalized, f32).
///
/// Uses mixed radix-2/radix-4 recursion for power-of-two lengths and Bluestein
/// fallback for non-power-of-two lengths.
pub fn forward_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let out = forward_pow2_recursive_32(data);
        data.copy_from_slice(&out);
    } else {
        bluestein::forward_inplace_32(data);
    }
}

/// In-place inverse FFT (unnormalized, f32).
///
/// Uses mixed radix-2/radix-4 recursion for power-of-two lengths and Bluestein
/// fallback for non-power-of-two lengths.
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let out = inverse_pow2_recursive_unnorm_32(data);
        data.copy_from_slice(&out);
    } else {
        bluestein::inverse_inplace_unnorm_32(data);
    }
}

/// In-place inverse FFT normalized by 1/N (f32).
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    inverse_inplace_unnorm_32(data);
    let scale = 1.0f32 / data.len() as f32;
    for x in data.iter_mut() {
        *x *= scale;
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
