//! True radix-4 Cooley-Tukey kernels.
//!
//! This module implements in-place radix-4 DIT transforms for power-of-four
//! lengths. The radix-4 modules for 16/32/64 can then build higher-radix
//! behavior on top of this kernel family without routing through radix-2.

use super::kernel_api::radix_kernel_api;
use super::radix_permute::digit_reverse_permute_pow2_radix;
use super::radix_shape::{is_power_of_four, stage_twiddle};
use super::radix2_f16::Cf16;
use num_complex::{Complex32, Complex64};

radix_kernel_api! {
    check       = is_power_of_four,
    inplace64   = radix4_inplace_64,
    inplace32   = radix4_inplace_32,
    description = "power-of-four",
}

fn radix4_inplace_64(data: &mut [Complex64], inverse: bool, twiddles: Option<&[Complex64]>) {
    // Safety: all callers supply Some(twiddles); None is unreachable in the public API.
    let twiddles = twiddles.unwrap();
    debug_assert!(is_power_of_four(data.len()));
    if data.len() <= 1 {
        return;
    }

    digit_reverse_permute_pow2_radix::<4, _>(data);

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
                    let w1 = stage_twiddle(stage_twiddles, half, j);
                    let w2 = stage_twiddle(stage_twiddles, half, 2 * j);
                    let w3 = stage_twiddle(stage_twiddles, half, 3 * j);
                    a1 = a1 * w1;
                    a2 = a2 * w2;
                    a3 = a3 * w3;
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

fn radix4_inplace_32(data: &mut [Complex32], inverse: bool, twiddles: Option<&[Complex32]>) {
    // Safety: all callers supply Some(twiddles); None is unreachable in the public API.
    let twiddles = twiddles.unwrap();
    debug_assert!(is_power_of_four(data.len()));
    if data.len() <= 1 {
        return;
    }

    digit_reverse_permute_pow2_radix::<4, _>(data);

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
                    let w1 = stage_twiddle(stage_twiddles, half, j);
                    let w2 = stage_twiddle(stage_twiddles, half, 2 * j);
                    let w3 = stage_twiddle(stage_twiddles, half, 3 * j);
                    a1 = a1 * w1;
                    a2 = a2 * w2;
                    a3 = a3 * w3;
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
    use super::super::test_utils::max_abs_err_64;

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
