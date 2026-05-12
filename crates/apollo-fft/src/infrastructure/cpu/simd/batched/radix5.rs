//! Batched radix-5 CPU kernel.

use crate::domain::storage::FftStorage;

use super::real_sweep::RealSweep;

const LANES: usize = 8;
const C1: f32 = 0.309_016_97;
const C2: f32 = -0.809_017;
const S1: f32 = 0.951_056_54;
const S2: f32 = 0.587_785_24;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
};

#[inline(always)]
fn radix5_scalar_values(xr: [f32; 5], xi: [f32; 5], inverse: bool) -> ([f32; 5], [f32; 5]) {
    let x0r = xr[0];
    let x1r = xr[1];
    let x2r = xr[2];
    let x3r = xr[3];
    let x4r = xr[4];
    let x0i = xi[0];
    let x1i = xi[1];
    let x2i = xi[2];
    let x3i = xi[3];
    let x4i = xi[4];

    let a1r = x1r + x4r;
    let a1i = x1i + x4i;
    let a2r = x2r + x3r;
    let a2i = x2i + x3i;
    let b1r = x1r - x4r;
    let b1i = x1i - x4i;
    let b2r = x2r - x3r;
    let b2i = x2i - x3i;

    let y0r = x0r + a1r + a2r;
    let y0i = x0i + a1i + a2i;

    let p1r = C2.mul_add(a2r, C1.mul_add(a1r, x0r));
    let p1i = C2.mul_add(a2i, C1.mul_add(a1i, x0i));
    let q1r = S2.mul_add(b2r, S1 * b1r);
    let q1i = S2.mul_add(b2i, S1 * b1i);

    let p2r = C1.mul_add(a2r, C2.mul_add(a1r, x0r));
    let p2i = C1.mul_add(a2i, C2.mul_add(a1i, x0i));
    let q2r = (-S1).mul_add(b2r, S2 * b1r);
    let q2i = (-S1).mul_add(b2i, S2 * b1i);

    if inverse {
        (
            [y0r, p1r - q1i, p2r - q2i, p2r + q2i, p1r + q1i],
            [y0i, p1i + q1r, p2i + q2r, p2i - q2r, p1i - q1r],
        )
    } else {
        (
            [y0r, p1r + q1i, p2r + q2i, p2r - q2i, p1r - q1i],
            [y0i, p1i - q1r, p2i - q2r, p2i + q2r, p1i + q1r],
        )
    }
}

#[inline(always)]
fn radix5_lane_scalar(
    re: &mut [[f32; LANES]; 5],
    im: &mut [[f32; LANES]; 5],
    lane: usize,
    inverse: bool,
) {
    let (yr, yi) = radix5_scalar_values(
        [
            re[0][lane],
            re[1][lane],
            re[2][lane],
            re[3][lane],
            re[4][lane],
        ],
        [
            im[0][lane],
            im[1][lane],
            im[2][lane],
            im[3][lane],
            im[4][lane],
        ],
        inverse,
    );
    for point in 0..5 {
        re[point][lane] = yr[point];
        im[point][lane] = yi[point];
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn load(row: &[f32; LANES]) -> __m256 {
    // SAFETY: every row has exactly LANES contiguous f32 values.
    unsafe { _mm256_loadu_ps(row.as_ptr()) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn store(row: &mut [f32; LANES], value: __m256) {
    // SAFETY: every row has exactly LANES contiguous f32 values.
    unsafe { _mm256_storeu_ps(row.as_mut_ptr(), value) };
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn load_sweep_re<const INVERSE: bool, S>(sweep: &S, point: usize, col: usize) -> __m256
where
    S: RealSweep<INVERSE, 5>,
{
    unsafe { _mm256_loadu_ps(sweep.re_ptr(point, col)) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn load_sweep_im<const INVERSE: bool, S>(sweep: &S, point: usize, col: usize) -> __m256
where
    S: RealSweep<INVERSE, 5>,
{
    unsafe { _mm256_loadu_ps(sweep.im_ptr(point, col)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn fmadd(a: f32, b: __m256, c: __m256) -> __m256 {
    _mm256_fmadd_ps(_mm256_set1_ps(a), b, c)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn radix5_avx_fma_core(
    xr: [__m256; 5],
    xi: [__m256; 5],
    inverse: bool,
) -> ([__m256; 5], [__m256; 5]) {
    // SAFETY: target features are checked by the caller before entering this
    // function. Lanes are independent FFT instances; the algorithm uses only
    // lane-wise AVX/FMA operations and performs no cross-lane shuffle.
    let x0r = xr[0];
    let x1r = xr[1];
    let x2r = xr[2];
    let x3r = xr[3];
    let x4r = xr[4];
    let x0i = xi[0];
    let x1i = xi[1];
    let x2i = xi[2];
    let x3i = xi[3];
    let x4i = xi[4];

    unsafe {
        let a1r = _mm256_add_ps(x1r, x4r);
        let a1i = _mm256_add_ps(x1i, x4i);
        let a2r = _mm256_add_ps(x2r, x3r);
        let a2i = _mm256_add_ps(x2i, x3i);
        let b1r = _mm256_sub_ps(x1r, x4r);
        let b1i = _mm256_sub_ps(x1i, x4i);
        let b2r = _mm256_sub_ps(x2r, x3r);
        let b2i = _mm256_sub_ps(x2i, x3i);

        let y0r = _mm256_add_ps(x0r, _mm256_add_ps(a1r, a2r));
        let y0i = _mm256_add_ps(x0i, _mm256_add_ps(a1i, a2i));

        let p1r = fmadd(C2, a2r, fmadd(C1, a1r, x0r));
        let p1i = fmadd(C2, a2i, fmadd(C1, a1i, x0i));
        let q1r = fmadd(S2, b2r, _mm256_mul_ps(_mm256_set1_ps(S1), b1r));
        let q1i = fmadd(S2, b2i, _mm256_mul_ps(_mm256_set1_ps(S1), b1i));

        let p2r = fmadd(C1, a2r, fmadd(C2, a1r, x0r));
        let p2i = fmadd(C1, a2i, fmadd(C2, a1i, x0i));
        let q2r = fmadd(-S1, b2r, _mm256_mul_ps(_mm256_set1_ps(S2), b1r));
        let q2i = fmadd(-S1, b2i, _mm256_mul_ps(_mm256_set1_ps(S2), b1i));

        if inverse {
            (
                [
                    y0r,
                    _mm256_sub_ps(p1r, q1i),
                    _mm256_sub_ps(p2r, q2i),
                    _mm256_add_ps(p2r, q2i),
                    _mm256_add_ps(p1r, q1i),
                ],
                [
                    y0i,
                    _mm256_add_ps(p1i, q1r),
                    _mm256_add_ps(p2i, q2r),
                    _mm256_sub_ps(p2i, q2r),
                    _mm256_sub_ps(p1i, q1r),
                ],
            )
        } else {
            (
                [
                    y0r,
                    _mm256_add_ps(p1r, q1i),
                    _mm256_add_ps(p2r, q2i),
                    _mm256_sub_ps(p2r, q2i),
                    _mm256_sub_ps(p1r, q1i),
                ],
                [
                    y0i,
                    _mm256_sub_ps(p1i, q1r),
                    _mm256_sub_ps(p2i, q2r),
                    _mm256_add_ps(p2i, q2r),
                    _mm256_add_ps(p1i, q1r),
                ],
            )
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn radix5_batch_avx_fma(
    re: &mut [[f32; LANES]; 5],
    im: &mut [[f32; LANES]; 5],
    inverse: bool,
) {
    unsafe {
        let (yr, yi) = radix5_avx_fma_core(
            [
                load(&re[0]),
                load(&re[1]),
                load(&re[2]),
                load(&re[3]),
                load(&re[4]),
            ],
            [
                load(&im[0]),
                load(&im[1]),
                load(&im[2]),
                load(&im[3]),
                load(&im[4]),
            ],
            inverse,
        );
        for point in 0..5 {
            store(&mut re[point], yr[point]);
            store(&mut im[point], yi[point]);
        }
    }
}

/// Execute up to eight independent radix-5 DFT instances with native f32 arithmetic.
///
/// # Theorem
///
/// For `omega = exp(-2*pi*i/5)`, the DFT is
/// `Y_k = sum_j x_j omega^(j*k)`. Pairing conjugate exponents gives
/// `x1*omega^k + x4*omega^(4k) = cos(2*pi*k/5)(x1+x4)
/// - i sin(2*pi*k/5)(x1-x4)` and the analogous `(x2,x3)` pair. The kernel
/// names these pair sums `a1`, `a2` and pair differences `b1`, `b2`; `p*`
/// terms are cosine projections and `q*` terms are sine projections.
/// Substitution reconstructs every DFT matrix row.
///
/// # Precision contract
///
/// This f32 entry point accepts only `FftStorage<f32>`. f16 is not routed
/// through this function. CPU f16 support is unavailable until a stable
/// AVX-512-FP16 implementation exists, because widening f16 to f32 violates
/// Apollo's native-precision contract.
pub fn radix5_batched_f32<S: FftStorage<f32>>(storage: &mut S, inverse: bool) {
    assert_eq!(storage.rows(), 5, "radix-5 storage row count mismatch");
    assert!(storage.cols() <= LANES, "radix-5 batch width exceeds lanes");
    let cols = storage.cols();
    let mut re = [[0.0_f32; LANES]; 5];
    let mut im = [[0.0_f32; LANES]; 5];
    for lane in 0..cols {
        for point in 0..5 {
            re[point][lane] = storage.load_re(point, lane);
            im[point][lane] = storage.load_im(point, lane);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
        // SAFETY: runtime feature detection proves the required AVX/FMA
        // instructions are available for the native f32 path.
        unsafe {
            radix5_batch_avx_fma(&mut re, &mut im, inverse);
        }
    } else {
        for lane in 0..cols {
            radix5_lane_scalar(&mut re, &mut im, lane, inverse);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    for lane in 0..cols {
        radix5_lane_scalar(&mut re, &mut im, lane, inverse);
    }
    for lane in 0..cols {
        for point in 0..5 {
            storage.store(point, lane, re[point][lane], im[point][lane]);
        }
    }
}

/// Execute a radix-5 real column sweep without stack lane-tile pack/unpack.
pub(crate) fn radix5_real_sweep<const INVERSE: bool, S>(sweep: &mut S)
where
    S: RealSweep<INVERSE, 5>,
{
    let n2 = sweep.n2();
    let mut col = 0usize;
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
        while col + LANES <= n2 {
            // SAFETY: runtime feature detection proves AVX/FMA availability.
            // The loop only enters for full eight-column spans; each row slice
            // has length `n2`, and all row offsets are within checked buffers.
            unsafe {
                let zero = _mm256_setzero_ps();
                let (yr, yi) = radix5_avx_fma_core(
                    [
                        load_sweep_re(sweep, 0, col),
                        load_sweep_re(sweep, 1, col),
                        load_sweep_re(sweep, 2, col),
                        load_sweep_re(sweep, 3, col),
                        load_sweep_re(sweep, 4, col),
                    ],
                    if INVERSE {
                        [
                            load_sweep_im(sweep, 0, col),
                            load_sweep_im(sweep, 1, col),
                            load_sweep_im(sweep, 2, col),
                            load_sweep_im(sweep, 3, col),
                            load_sweep_im(sweep, 4, col),
                        ]
                    } else {
                        [zero, zero, zero, zero, zero]
                    },
                    INVERSE,
                );
                for point in 0..5 {
                    sweep.store_avx(point, col, yr[point], yi[point]);
                }
            }
            col += LANES;
        }
    }

    while col < n2 {
        let (yr, yi) = radix5_scalar_values(
            [
                sweep.load_re(0, col),
                sweep.load_re(1, col),
                sweep.load_re(2, col),
                sweep.load_re(3, col),
                sweep.load_re(4, col),
            ],
            [
                sweep.load_im(0, col),
                sweep.load_im(1, col),
                sweep.load_im(2, col),
                sweep.load_im(3, col),
                sweep.load_im(4, col),
            ],
            INVERSE,
        );
        for point in 0..5 {
            sweep.store(point, col, yr[point], yi[point]);
        }
        col += 1;
    }
}
