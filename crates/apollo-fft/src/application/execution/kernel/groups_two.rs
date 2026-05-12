use num_complex::Complex32;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stage_pair32_groups_two_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    radix: usize,
    first_twiddles: &[Complex32],
    second_twiddles: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_castpd_ps, _mm256_castps_pd, _mm256_loadu_ps, _mm256_movehdup_ps,
        _mm256_moveldup_ps, _mm256_permute2f128_pd, _mm256_storeu_ps, _mm256_sub_ps,
        _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    };

    let n = src.len();
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    let vector_end = radix & !3usize;

    let mut j = 0usize;
    while j < vector_end {
        let d0 = _mm256_loadu_ps(src.as_ptr().add(j * 4).cast::<f32>());
        let d1 = _mm256_loadu_ps(src.as_ptr().add((j + 1) * 4).cast::<f32>());
        let d2 = _mm256_loadu_ps(src.as_ptr().add((j + 2) * 4).cast::<f32>());
        let d3 = _mm256_loadu_ps(src.as_ptr().add((j + 3) * 4).cast::<f32>());

        let t0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(d0), _mm256_castps_pd(d1)));
        let t1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(d0), _mm256_castps_pd(d1)));
        let t2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(d2), _mm256_castps_pd(d3)));
        let t3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(d2), _mm256_castps_pd(d3)));

        let x0 = _mm256_castpd_ps(_mm256_permute2f128_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t2), 0x20));
        let x2 = _mm256_castpd_ps(_mm256_permute2f128_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t2), 0x31));
        let x1 = _mm256_castpd_ps(_mm256_permute2f128_pd(_mm256_castps_pd(t1), _mm256_castps_pd(t3), 0x20));
        let x3 = _mm256_castpd_ps(_mm256_permute2f128_pd(_mm256_castps_pd(t1), _mm256_castps_pd(t3), 0x31));

        let w1 = _mm256_loadu_ps(first_twiddles.as_ptr().add(j).cast::<f32>());
        let w1r = _mm256_moveldup_ps(w1);
        let w1i = _mm256_movehdup_ps(w1);

        let x2_mul = cmul_vec32(w1r, w1i, x2);
        let x3_mul = cmul_vec32(w1r, w1i, x3);

        let a0 = _mm256_add_ps(x0, x2_mul);
        let a1 = _mm256_add_ps(x1, x3_mul);
        let b0 = _mm256_sub_ps(x0, x2_mul);
        let b1 = _mm256_sub_ps(x1, x3_mul);

        let w2 = _mm256_loadu_ps(second_twiddles.as_ptr().add(j).cast::<f32>());
        let w3 = _mm256_loadu_ps(second_twiddles.as_ptr().add(j + radix).cast::<f32>());

        let w2r = _mm256_moveldup_ps(w2);
        let w2i = _mm256_movehdup_ps(w2);
        let c0 = cmul_vec32(w2r, w2i, a1);

        let w3r = _mm256_moveldup_ps(w3);
        let w3i = _mm256_movehdup_ps(w3);
        let c1 = cmul_vec32(w3r, w3i, b1);

        _mm256_storeu_ps(dst.as_mut_ptr().add(j).cast::<f32>(), _mm256_add_ps(a0, c0));
        _mm256_storeu_ps(dst.as_mut_ptr().add(j + half_n).cast::<f32>(), _mm256_sub_ps(a0, c0));
        _mm256_storeu_ps(dst.as_mut_ptr().add(j + quarter_n).cast::<f32>(), _mm256_add_ps(b0, c1));
        _mm256_storeu_ps(dst.as_mut_ptr().add(j + half_n + quarter_n).cast::<f32>(), _mm256_sub_ps(b0, c1));

        j += 4;
    }

    while j < radix {
        let w1 = first_twiddles[j];
        let w2 = second_twiddles[j];
        let w3 = second_twiddles[j + radix];
        
        let src_base = j * 4;
        let x0 = src[src_base];
        let x1 = src[src_base + 1];
        let x2 = src[src_base + 2] * w1;
        let x3 = src[src_base + 3] * w1;
        
        let a0 = x0 + x2;
        let a1 = x1 + x3;
        let b0 = x0 - x2;
        let b1 = x1 - x3;
        
        let c0 = a1 * w2;
        let c1 = b1 * w3;
        
        dst[j] = a0 + c0;
        dst[j + half_n] = a0 - c0;
        dst[j + quarter_n] = b0 + c1;
        dst[j + half_n + quarter_n] = b0 - c1;
        
        j += 1;
    }
}
