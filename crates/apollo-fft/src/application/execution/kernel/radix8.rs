//! Radix-8 DIT FFT using Winograd DFT-8 inner butterflies.
//!
//! ## Algorithm
//!
//! For an N-point transform where N is a power of 8, we perform an iterative
//! DIT in stages of radix 8.  Each stage processes groups of length `len = 8^s`
//! (for stage s = 1, 2, …, log₈ N):
//!
//! 1. Apply the digit-reverse permutation (base-8) once up front.
//! 2. For each stage, partition the data into non-overlapping groups of `len`
//!    elements.  Within each group, apply inter-group twiddle factors
//!    `W_len^{k·j}` to the non-first sub-blocks, then call the Winograd DFT-8
//!    kernel on each set of 8 elements separated by stride `len/8`.
//!
//! The Winograd DFT-8 kernel requires only 4 real multiplications (the
//! `±√2/2` factors on the odd path) and 26 real additions, versus 64
//! multiplications for the generic DFT-matrix approach.
//!
//! ## References
//!
//! - Winograd, S. (1978). On computing the discrete Fourier transform.
//!   *Mathematics of Computation*, 32(141), 175-199.

use super::{radix2, winograd};
use num_complex::{Complex32, Complex64};

#[inline]
fn is_power_of_eight(n: usize) -> bool {
    n > 0 && n.is_power_of_two() && (n.trailing_zeros() % 3 == 0)
}

// ── digit-reverse permutation (base 8) ───────────────────────────────────────

#[inline]
fn reverse_base8(mut v: usize, digits: u32) -> usize {
    let mut r = 0;
    for _ in 0..digits {
        r = (r << 3) | (v & 7);
        v >>= 3;
    }
    r
}

fn digit_reverse_64(data: &mut [Complex64]) {
    let digits = data.len().trailing_zeros() / 3;
    for i in 0..data.len() {
        let j = reverse_base8(i, digits);
        if j > i {
            data.swap(i, j);
        }
    }
}

fn digit_reverse_32(data: &mut [Complex32]) {
    let digits = data.len().trailing_zeros() / 3;
    for i in 0..data.len() {
        let j = reverse_base8(i, digits);
        if j > i {
            data.swap(i, j);
        }
    }
}

// ── core Winograd-radix-8 in-place kernel ─────────────────────────────────────

fn winograd_r8_inplace_64(data: &mut [Complex64], twiddles: Option<&[Complex64]>, inverse: bool) {
    debug_assert!(is_power_of_eight(data.len()));
    if data.len() <= 1 {
        return;
    }
    digit_reverse_64(data);

    let n = data.len();
    let mut len = 8usize;
    while len <= n {
        let eighth = len >> 3;
        // Stage twiddle slice: same layout as radix2 table (half - 1)..(half - 1 + half)
        let half = len >> 1;
        let stage = twiddles.map(|t| &t[(half - 1)..(half - 1 + half)]);

        for chunk in data.chunks_exact_mut(len) {
            for j in 0..eighth {
                // Apply inter-group twiddles W_len^{p*j} for p = 0..8.
                // W_len^0 = 1 (no-op). For p ≥ 1, compute by recurrence.
                let step = if let Some(s) = stage {
                    s[j]
                } else {
                    let a = if inverse {
                        std::f64::consts::TAU * j as f64 / len as f64
                    } else {
                        -std::f64::consts::TAU * j as f64 / len as f64
                    };
                    Complex64::new(a.cos(), a.sin())
                };
                let mut tw = Complex64::new(1.0, 0.0);
                let mut buf = [Complex64::new(0.0, 0.0); 8];
                for p in 0..8 {
                    buf[p] = winograd::apply_twiddle_64(chunk[j + p * eighth], tw);
                    tw = winograd::apply_twiddle_64(tw, step);
                }
                winograd::dft8_64(&mut buf, inverse);
                for p in 0..8 {
                    chunk[j + p * eighth] = buf[p];
                }
            }
        }
        len <<= 3;
    }
}

fn winograd_r8_inplace_32(data: &mut [Complex32], twiddles: Option<&[Complex32]>, inverse: bool) {
    debug_assert!(is_power_of_eight(data.len()));
    if data.len() <= 1 {
        return;
    }
    digit_reverse_32(data);

    let n = data.len();
    let mut len = 8usize;
    while len <= n {
        let eighth = len >> 3;
        let half = len >> 1;
        let stage = twiddles.map(|t| &t[(half - 1)..(half - 1 + half)]);

        for chunk in data.chunks_exact_mut(len) {
            for j in 0..eighth {
                let step = if let Some(s) = stage {
                    s[j]
                } else {
                    let a = if inverse {
                        std::f64::consts::TAU * j as f64 / len as f64
                    } else {
                        -std::f64::consts::TAU * j as f64 / len as f64
                    };
                    Complex32::new(a.cos() as f32, a.sin() as f32)
                };
                let mut tw = Complex32::new(1.0, 0.0);
                let mut buf = [Complex32::new(0.0, 0.0); 8];
                for p in 0..8 {
                    buf[p] = winograd::apply_twiddle_32(chunk[j + p * eighth], tw);
                    tw = winograd::apply_twiddle_32(tw, step);
                }
                winograd::dft8_32(&mut buf, inverse);
                for p in 0..8 {
                    chunk[j + p * eighth] = buf[p];
                }
            }
        }
        len <<= 3;
    }
}

// ── public API ────────────────────────────────────────────────────────────────

/// Forward FFT (unnormalized) for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn forward_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 { return; }
    debug_assert!(is_power_of_eight(data.len()));
    winograd_r8_inplace_64(data, Some(twiddles), false);
}

/// Inverse FFT (unnormalized) for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_unnorm_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 { return; }
    debug_assert!(is_power_of_eight(data.len()));
    winograd_r8_inplace_64(data, Some(twiddles), true);
}

/// Inverse FFT normalized by 1/N for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 { return; }
    debug_assert!(is_power_of_eight(data.len()));
    winograd_r8_inplace_64(data, Some(twiddles), true);
    let inv_n = 1.0 / data.len() as f64;
    for v in data.iter_mut() { *v *= inv_n; }
}

/// Forward FFT (unnormalized) for power-of-eight lengths.
pub fn forward_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 { return; }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_forward_twiddle_table_64(data.len());
    winograd_r8_inplace_64(data, Some(&twiddles), false);
}

/// Inverse FFT (unnormalized) for power-of-eight lengths.
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    if data.len() <= 1 { return; }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_64(data.len());
    winograd_r8_inplace_64(data, Some(&twiddles), true);
}

/// Inverse FFT normalized by 1/N for power-of-eight lengths.
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 { return; }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_64(data.len());
    winograd_r8_inplace_64(data, Some(&twiddles), true);
    let inv_n = 1.0 / data.len() as f64;
    for v in data.iter_mut() { *v *= inv_n; }
}

/// Forward FFT (unnormalized, f32) for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn forward_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 { return; }
    debug_assert!(is_power_of_eight(data.len()));
    winograd_r8_inplace_32(data, Some(twiddles), false);
}

/// Inverse FFT (unnormalized, f32) for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_unnorm_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 { return; }
    debug_assert!(is_power_of_eight(data.len()));
    winograd_r8_inplace_32(data, Some(twiddles), true);
}

/// Inverse FFT normalized by 1/N (f32) for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 { return; }
    debug_assert!(is_power_of_eight(data.len()));
    winograd_r8_inplace_32(data, Some(twiddles), true);
    let inv_n = 1.0f32 / data.len() as f32;
    for v in data.iter_mut() { *v *= inv_n; }
}

/// Forward FFT (unnormalized, f32) for power-of-eight lengths.
pub fn forward_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 { return; }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_forward_twiddle_table_32(data.len());
    winograd_r8_inplace_32(data, Some(&twiddles), false);
}

/// Inverse FFT (unnormalized, f32) for power-of-eight lengths.
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    if data.len() <= 1 { return; }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_32(data.len());
    winograd_r8_inplace_32(data, Some(&twiddles), true);
}

/// Inverse FFT normalized by 1/N (f32) for power-of-eight lengths.
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 { return; }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_32(data.len());
    winograd_r8_inplace_32(data, Some(&twiddles), true);
    let inv_n = 1.0f32 / data.len() as f32;
    for v in data.iter_mut() { *v *= inv_n; }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward_64, dft_inverse_64};

    fn max_abs_err_64(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (*x - *y).norm()).fold(0.0f64, f64::max)
    }

    #[test]
    fn radix8_forward_n8_matches_direct() {
        let n = 8usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.31).sin(), (k as f64 * 0.17).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-11, "radix8 n=8 forward err={err:.2e}");
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
        assert!(err < 1e-10, "radix8 n=64 forward err={err:.2e}");
    }

    #[test]
    fn radix8_forward_n512_matches_direct() {
        let n = 512usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.07).sin(), (k as f64 * 0.04).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-9, "radix8 n=512 forward err={err:.2e}");
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
            .into_iter().map(|x| x * n as f64).collect::<Vec<_>>();
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "radix8 n=64 inverse unnorm err={err:.2e}");
    }

    #[test]
    fn radix8_roundtrip_n64() {
        let n = 64usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.19).sin(), -(k as f64 * 0.23).cos()))
            .collect();
        let mut buf = input.clone();
        forward_inplace_64(&mut buf);
        inverse_inplace_64(&mut buf);
        let err = max_abs_err_64(&buf, &input);
        assert!(err < 1e-10, "radix8 roundtrip err={err:.2e}");
    }
}
