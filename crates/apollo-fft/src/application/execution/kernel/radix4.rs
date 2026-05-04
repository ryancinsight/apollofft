//! True radix-4 Cooley-Tukey kernels.
//!
//! ## Twiddle identity (W_len^{2j} = W_{len/2}^j)
//!
//! prev_stage[j] = W_{len/2}^j = W_len^{2j} gives stride-1 access.
//! W_len^{3j} = curr[j] * prev[j] with one extra multiply.
//! j=0 peel: W_len^0=1+0i eliminates 4 muls per chunk per stage.
use super::kernel_api::radix_kernel_api;
use super::radix2_f16::Cf16;
use super::radix_permute::digit_reverse_permute_pow2_radix;
use super::radix_shape::is_power_of_four;
use super::radix_stage::WinogradComplex;
use num_complex::{Complex32, Complex64};

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
use std::arch::x86_64::{
    _mm256_add_pd, _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd,
    _mm256_setr_pd, _mm256_storeu_pd, _mm256_sub_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd,
};

radix_kernel_api! {
    check       = is_power_of_four,
    inplace64   = radix4_inplace_64,
    inplace32   = radix4_inplace_32,
    description = "power-of-four",
}

#[inline]
fn radix4_inplace<C>(data: &mut [C], inverse: bool, twiddles: Option<&[C]>)
where
    C: WinogradComplex + std::ops::Mul<Output = C> + std::ops::Neg<Output = C> + Copy,
{
    let twiddles = twiddles.unwrap();
    debug_assert!(is_power_of_four(data.len()));
    if data.len() <= 1 { return; }
    digit_reverse_permute_pow2_radix::<4, _>(data);
    let n = data.len();
    let mut len = 4usize;
    while len <= n {
        let quarter = len >> 2;
        let half = len >> 1;
        let (maybe_curr, maybe_prev): (Option<&[C]>, Option<&[C]>) = if len > 4 {
            (Some(&twiddles[(half-1)..(half-1+half)]), Some(&twiddles[(quarter-1)..(half-1)]))
        } else { (None, None) };
        for chunk in data.chunks_exact_mut(len) {
            let (lo, hi) = chunk.split_at_mut(half);
            let (s0, s1) = lo.split_at_mut(quarter);
            let (s2, s3) = hi.split_at_mut(quarter);
            match (maybe_curr, maybe_prev) {
                (Some(curr), Some(prev)) => {
                    // j=0: trivial twiddles, no multiplication needed.
                    { let t0=s0[0]+s2[0]; let t1=s0[0]-s2[0]; let t2=s1[0]+s3[0]; let t3=s1[0]-s3[0];
                      let (y1,y3) = if inverse { (t1+t3.rot_pos_i(),t1-t3.rot_pos_i()) } else { (t1+t3.rot_neg_i(),t1-t3.rot_neg_i()) };
                      s0[0]=t0+t2; s1[0]=y1; s2[0]=t0-t2; s3[0]=y3; }
                    for j in 1..quarter {
                        let tw1=curr[j]; let tw2=prev[j]; let tw3=tw1*tw2;
                        let a1v=s1[j]*tw1; let a2v=s2[j]*tw2; let a3v=s3[j]*tw3;
                        let t0=s0[j]+a2v; let t1=s0[j]-a2v; let t2=a1v+a3v; let t3=a1v-a3v;
                        let (y1,y3) = if inverse { (t1+t3.rot_pos_i(),t1-t3.rot_pos_i()) } else { (t1+t3.rot_neg_i(),t1-t3.rot_neg_i()) };
                        s0[j]=t0+t2; s1[j]=y1; s2[j]=t0-t2; s3[j]=y3;
                    }
                }
                _ => {
                    for j in 0..quarter {
                        let t0=s0[j]+s2[j]; let t1=s0[j]-s2[j]; let t2=s1[j]+s3[j]; let t3=s1[j]-s3[j];
                        let (y1,y3) = if inverse { (t1+t3.rot_pos_i(),t1-t3.rot_pos_i()) } else { (t1+t3.rot_neg_i(),t1-t3.rot_neg_i()) };
                        s0[j]=t0+t2; s1[j]=y1; s2[j]=t0-t2; s3[j]=y3;
                    }
                }
            }
        }
        len <<= 2;
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline(always)]
unsafe fn cmul2_f64(a: std::arch::x86_64::__m256d, b: std::arch::x86_64::__m256d) -> std::arch::x86_64::__m256d {
    let ar = _mm256_unpacklo_pd(a, a);
    let ai = _mm256_unpackhi_pd(a, a);
    let bsw = _mm256_permute_pd(b, 5);
    _mm256_fmaddsub_pd(ar, b, _mm256_mul_pd(ai, bsw))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline(always)]
unsafe fn rot2_f64(v: std::arch::x86_64::__m256d, inverse: bool) -> std::arch::x86_64::__m256d {
    let sw = _mm256_permute_pd(v, 5);
    let mask = if inverse {
        _mm256_setr_pd(-1.0, 1.0, -1.0, 1.0)
    } else {
        _mm256_setr_pd(1.0, -1.0, 1.0, -1.0)
    };
    _mm256_mul_pd(sw, mask)
}

#[inline(always)]
fn radix4_inplace_64(data: &mut [Complex64], inverse: bool, twiddles: Option<&[Complex64]>) {
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
        let (maybe_curr, maybe_prev): (Option<&[Complex64]>, Option<&[Complex64]>) = if len > 4 {
            (
                Some(&twiddles[(half - 1)..(half - 1 + half)]),
                Some(&twiddles[(quarter - 1)..(half - 1)]),
            )
        } else {
            (None, None)
        };

        for chunk in data.chunks_exact_mut(len) {
            let (lo, hi) = chunk.split_at_mut(half);
            let (s0, s1) = lo.split_at_mut(quarter);
            let (s2, s3) = hi.split_at_mut(quarter);

            match (maybe_curr, maybe_prev) {
                (Some(curr), Some(prev)) => {
                    // j=0: trivial twiddles, no multiplication needed.
                    {
                        let t0 = s0[0] + s2[0];
                        let t1 = s0[0] - s2[0];
                        let t2 = s1[0] + s3[0];
                        let t3 = s1[0] - s3[0];
                        let (y1, y3) = if inverse {
                            (t1 + t3.rot_pos_i(), t1 - t3.rot_pos_i())
                        } else {
                            (t1 + t3.rot_neg_i(), t1 - t3.rot_neg_i())
                        };
                        s0[0] = t0 + t2;
                        s1[0] = y1;
                        s2[0] = t0 - t2;
                        s3[0] = y3;
                    }

                    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
                    {
                        // SIMD j-loop: process two Complex64 values at a time.
                        let mut j = 1usize;
                        while j + 1 < quarter {
                            unsafe {
                                let p0 = s0.as_ptr().add(j) as *const f64;
                                let p1 = s1.as_ptr().add(j) as *const f64;
                                let p2 = s2.as_ptr().add(j) as *const f64;
                                let p3 = s3.as_ptr().add(j) as *const f64;
                                let pw1 = curr.as_ptr().add(j) as *const f64;
                                let pw2 = prev.as_ptr().add(j) as *const f64;

                                let a0 = _mm256_loadu_pd(p0);
                                let a1 = _mm256_loadu_pd(p1);
                                let a2 = _mm256_loadu_pd(p2);
                                let a3 = _mm256_loadu_pd(p3);
                                let tw1 = _mm256_loadu_pd(pw1);
                                let tw2 = _mm256_loadu_pd(pw2);
                                let tw3 = cmul2_f64(tw1, tw2);

                                let a1v = cmul2_f64(a1, tw1);
                                let a2v = cmul2_f64(a2, tw2);
                                let a3v = cmul2_f64(a3, tw3);

                                let t0 = _mm256_add_pd(a0, a2v);
                                let t1 = _mm256_sub_pd(a0, a2v);
                                let t2 = _mm256_add_pd(a1v, a3v);
                                let t3 = _mm256_sub_pd(a1v, a3v);
                                let rot = rot2_f64(t3, inverse);

                                let y1 = _mm256_add_pd(t1, rot);
                                let y3 = _mm256_sub_pd(t1, rot);

                                let o0 = _mm256_add_pd(t0, t2);
                                let o2 = _mm256_sub_pd(t0, t2);

                                _mm256_storeu_pd(s0.as_mut_ptr().add(j) as *mut f64, o0);
                                _mm256_storeu_pd(s1.as_mut_ptr().add(j) as *mut f64, y1);
                                _mm256_storeu_pd(s2.as_mut_ptr().add(j) as *mut f64, o2);
                                _mm256_storeu_pd(s3.as_mut_ptr().add(j) as *mut f64, y3);
                            }
                            j += 2;
                        }

                        for j in j..quarter {
                            let tw1 = curr[j];
                            let tw2 = prev[j];
                            let tw3 = tw1 * tw2;
                            let a1v = s1[j] * tw1;
                            let a2v = s2[j] * tw2;
                            let a3v = s3[j] * tw3;
                            let t0 = s0[j] + a2v;
                            let t1 = s0[j] - a2v;
                            let t2 = a1v + a3v;
                            let t3 = a1v - a3v;
                            let (y1, y3) = if inverse {
                                (t1 + t3.rot_pos_i(), t1 - t3.rot_pos_i())
                            } else {
                                (t1 + t3.rot_neg_i(), t1 - t3.rot_neg_i())
                            };
                            s0[j] = t0 + t2;
                            s1[j] = y1;
                            s2[j] = t0 - t2;
                            s3[j] = y3;
                        }
                    }

                    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
                    {
                        for j in 1..quarter {
                            let tw1 = curr[j];
                            let tw2 = prev[j];
                            let tw3 = tw1 * tw2;
                            let a1v = s1[j] * tw1;
                            let a2v = s2[j] * tw2;
                            let a3v = s3[j] * tw3;
                            let t0 = s0[j] + a2v;
                            let t1 = s0[j] - a2v;
                            let t2 = a1v + a3v;
                            let t3 = a1v - a3v;
                            let (y1, y3) = if inverse {
                                (t1 + t3.rot_pos_i(), t1 - t3.rot_pos_i())
                            } else {
                                (t1 + t3.rot_neg_i(), t1 - t3.rot_neg_i())
                            };
                            s0[j] = t0 + t2;
                            s1[j] = y1;
                            s2[j] = t0 - t2;
                            s3[j] = y3;
                        }
                    }
                }
                _ => {
                    for j in 0..quarter {
                        let t0 = s0[j] + s2[j];
                        let t1 = s0[j] - s2[j];
                        let t2 = s1[j] + s3[j];
                        let t3 = s1[j] - s3[j];
                        let (y1, y3) = if inverse {
                            (t1 + t3.rot_pos_i(), t1 - t3.rot_pos_i())
                        } else {
                            (t1 + t3.rot_neg_i(), t1 - t3.rot_neg_i())
                        };
                        s0[j] = t0 + t2;
                        s1[j] = y1;
                        s2[j] = t0 - t2;
                        s3[j] = y3;
                    }
                }
            }
        }
        len <<= 2;
    }
}

#[inline(always)]
fn radix4_inplace_32(data: &mut [Complex32], inverse: bool, twiddles: Option<&[Complex32]>) {
    radix4_inplace(data, inverse, twiddles);
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::max_abs_err_64;
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward_64, dft_inverse_64};

    #[test]
    fn radix4_forward_n16_matches_direct() {
        let n = 16usize;
        let input: Vec<Complex64> = (0..n).map(|k| Complex64::new((k as f64*0.3).sin(),(k as f64*0.11).cos())).collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "radix4 forward mismatch err={err:.2e}");
    }

    #[test]
    fn radix4_inverse_unnorm_n16_matches_direct() {
        let n = 16usize;
        let input: Vec<Complex64> = (0..n).map(|k| Complex64::new((k as f64*0.27).cos(),(k as f64*0.17).sin())).collect();
        let mut got = input.clone();
        inverse_inplace_unnorm_64(&mut got);
        let expected = dft_inverse_64(&input).into_iter().map(|x| x * n as f64).collect::<Vec<_>>();
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "radix4 inverse mismatch err={err:.2e}");
    }
}
