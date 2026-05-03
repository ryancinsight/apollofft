//! Recursive radix-4 FFT kernels.
//!
//! These kernels target power-of-four lengths and are kept as explicit
//! algorithm variants for benchmarking and planner experimentation.

use num_complex::{Complex32, Complex64};

#[inline]
fn is_power_of_four(n: usize) -> bool {
    n.is_power_of_two() && (n.trailing_zeros() % 2 == 0)
}

#[inline]
fn cmul64(a: Complex64, b: Complex64) -> Complex64 {
    Complex64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

#[inline]
fn cmul32(a: Complex32, b: Complex32) -> Complex32 {
    Complex32::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
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

fn radix4_butterfly4_64(x: &[Complex64], inverse: bool) -> [Complex64; 4] {
    let i_mul = |v: Complex64| {
        if inverse {
            Complex64::new(-v.im, v.re)
        } else {
            Complex64::new(v.im, -v.re)
        }
    };
    let neg_i_mul = |v: Complex64| {
        if inverse {
            Complex64::new(v.im, -v.re)
        } else {
            Complex64::new(-v.im, v.re)
        }
    };

    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];
    let x3 = x[3];

    let t0 = x0 + x2;
    let t1 = x0 - x2;
    let t2 = x1 + x3;
    let t3 = x1 - x3;

    [t0 + t2, t1 + i_mul(t3), t0 - t2, t1 + neg_i_mul(t3)]
}

fn radix4_butterfly4_32(x: &[Complex32], inverse: bool) -> [Complex32; 4] {
    let i_mul = |v: Complex32| {
        if inverse {
            Complex32::new(-v.im, v.re)
        } else {
            Complex32::new(v.im, -v.re)
        }
    };
    let neg_i_mul = |v: Complex32| {
        if inverse {
            Complex32::new(v.im, -v.re)
        } else {
            Complex32::new(-v.im, v.re)
        }
    };

    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];
    let x3 = x[3];

    let t0 = x0 + x2;
    let t1 = x0 - x2;
    let t2 = x1 + x3;
    let t3 = x1 - x3;

    [t0 + t2, t1 + i_mul(t3), t0 - t2, t1 + neg_i_mul(t3)]
}

fn fft_radix4_recursive_64(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = input.len();
    if n == 1 {
        return vec![input[0]];
    }
    debug_assert!(is_power_of_four(n));

    if n == 4 {
        return radix4_butterfly4_64(input, inverse).to_vec();
    }

    let m = n / 4;
    let mut x0 = Vec::with_capacity(m);
    let mut x1 = Vec::with_capacity(m);
    let mut x2 = Vec::with_capacity(m);
    let mut x3 = Vec::with_capacity(m);
    for t in 0..m {
        x0.push(input[4 * t]);
        x1.push(input[4 * t + 1]);
        x2.push(input[4 * t + 2]);
        x3.push(input[4 * t + 3]);
    }

    let y0 = fft_radix4_recursive_64(&x0, inverse);
    let y1 = fft_radix4_recursive_64(&x1, inverse);
    let y2 = fft_radix4_recursive_64(&x2, inverse);
    let y3 = fft_radix4_recursive_64(&x3, inverse);

    let mut out = vec![Complex64::new(0.0, 0.0); n];
    let i_mul = |v: Complex64| {
        if inverse {
            Complex64::new(-v.im, v.re)
        } else {
            Complex64::new(v.im, -v.re)
        }
    };
    let neg_i_mul = |v: Complex64| {
        if inverse {
            Complex64::new(v.im, -v.re)
        } else {
            Complex64::new(-v.im, v.re)
        }
    };

    for k in 0..m {
        let a0 = y0[k];
        let a1 = cmul64(twiddle64(n, k, inverse), y1[k]);
        let a2 = cmul64(twiddle64(n, 2 * k, inverse), y2[k]);
        let a3 = cmul64(twiddle64(n, 3 * k, inverse), y3[k]);

        let t0 = a0 + a2;
        let t1 = a0 - a2;
        let t2 = a1 + a3;
        let t3 = a1 - a3;

        out[k] = t0 + t2;
        out[k + m] = t1 + i_mul(t3);
        out[k + 2 * m] = t0 - t2;
        out[k + 3 * m] = t1 + neg_i_mul(t3);
    }

    out
}

fn fft_radix4_recursive_32(input: &[Complex32], inverse: bool) -> Vec<Complex32> {
    let n = input.len();
    if n == 1 {
        return vec![input[0]];
    }
    debug_assert!(is_power_of_four(n));

    if n == 4 {
        return radix4_butterfly4_32(input, inverse).to_vec();
    }

    let m = n / 4;
    let mut x0 = Vec::with_capacity(m);
    let mut x1 = Vec::with_capacity(m);
    let mut x2 = Vec::with_capacity(m);
    let mut x3 = Vec::with_capacity(m);
    for t in 0..m {
        x0.push(input[4 * t]);
        x1.push(input[4 * t + 1]);
        x2.push(input[4 * t + 2]);
        x3.push(input[4 * t + 3]);
    }

    let y0 = fft_radix4_recursive_32(&x0, inverse);
    let y1 = fft_radix4_recursive_32(&x1, inverse);
    let y2 = fft_radix4_recursive_32(&x2, inverse);
    let y3 = fft_radix4_recursive_32(&x3, inverse);

    let mut out = vec![Complex32::new(0.0, 0.0); n];
    let i_mul = |v: Complex32| {
        if inverse {
            Complex32::new(-v.im, v.re)
        } else {
            Complex32::new(v.im, -v.re)
        }
    };
    let neg_i_mul = |v: Complex32| {
        if inverse {
            Complex32::new(v.im, -v.re)
        } else {
            Complex32::new(-v.im, v.re)
        }
    };

    for k in 0..m {
        let a0 = y0[k];
        let a1 = cmul32(twiddle32(n, k, inverse), y1[k]);
        let a2 = cmul32(twiddle32(n, 2 * k, inverse), y2[k]);
        let a3 = cmul32(twiddle32(n, 3 * k, inverse), y3[k]);

        let t0 = a0 + a2;
        let t1 = a0 - a2;
        let t2 = a1 + a3;
        let t3 = a1 - a3;

        out[k] = t0 + t2;
        out[k + m] = t1 + i_mul(t3);
        out[k + 2 * m] = t0 - t2;
        out[k + 3 * m] = t1 + neg_i_mul(t3);
    }

    out
}

/// In-place forward FFT (unnormalized) for power-of-four lengths.
pub fn forward_inplace_64(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(n));
    let out = fft_radix4_recursive_64(data, false);
    data.copy_from_slice(&out);
}

/// In-place inverse FFT (unnormalized) for power-of-four lengths.
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(n));
    // F^{-1}_unnorm(X) = conj(F(conj(X)))
    let mut conj_in = data.to_vec();
    for x in conj_in.iter_mut() {
        *x = x.conj();
    }
    let out = fft_radix4_recursive_64(&conj_in, false);
    for (dst, v) in data.iter_mut().zip(out.into_iter()) {
        *dst = v.conj();
    }
}

/// In-place inverse FFT normalized by 1/N for power-of-four lengths.
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    inverse_inplace_unnorm_64(data);
    let scale = 1.0 / data.len() as f64;
    for x in data.iter_mut() {
        *x *= scale;
    }
}

/// In-place forward FFT (unnormalized, f32) for power-of-four lengths.
pub fn forward_inplace_32(data: &mut [Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(n));
    let out = fft_radix4_recursive_32(data, false);
    data.copy_from_slice(&out);
}

/// In-place inverse FFT (unnormalized, f32) for power-of-four lengths.
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(is_power_of_four(n));
    // F^{-1}_unnorm(X) = conj(F(conj(X)))
    let mut conj_in = data.to_vec();
    for x in conj_in.iter_mut() {
        *x = x.conj();
    }
    let out = fft_radix4_recursive_32(&conj_in, false);
    for (dst, v) in data.iter_mut().zip(out.into_iter()) {
        *dst = v.conj();
    }
}

/// In-place inverse FFT normalized by 1/N (f32) for power-of-four lengths.
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
