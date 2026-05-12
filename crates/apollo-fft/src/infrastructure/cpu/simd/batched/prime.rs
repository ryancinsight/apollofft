//! Batched prime-radix native-precision kernels.

use crate::domain::storage::FftStorage;

use super::real_sweep::RealSweep;

const LANES: usize = 8;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
};

const COS3: [[f32; 3]; 3] = [[1.0, 1.0, 1.0], [1.0, -0.5, -0.5], [1.0, -0.5, -0.5]];
const SIN3: [[f32; 3]; 3] = [
    [0.0, 0.0, 0.0],
    [0.0, 0.866_025_4, -0.866_025_4],
    [0.0, -0.866_025_4, 0.866_025_4],
];

const COS5: [[f32; 5]; 5] = [
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.309_016_97, -0.809_017, -0.809_017, 0.309_016_97],
    [1.0, -0.809_017, 0.309_016_97, 0.309_016_97, -0.809_017],
    [1.0, -0.809_017, 0.309_016_97, 0.309_016_97, -0.809_017],
    [1.0, 0.309_016_97, -0.809_017, -0.809_017, 0.309_016_97],
];
const SIN5: [[f32; 5]; 5] = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [
        0.0,
        0.951_056_54,
        0.587_785_24,
        -0.587_785_24,
        -0.951_056_54,
    ],
    [
        0.0,
        0.587_785_24,
        -0.951_056_54,
        0.951_056_54,
        -0.587_785_24,
    ],
    [
        0.0,
        -0.587_785_24,
        0.951_056_54,
        -0.951_056_54,
        0.587_785_24,
    ],
    [
        0.0,
        -0.951_056_54,
        -0.587_785_24,
        0.587_785_24,
        0.951_056_54,
    ],
];

const COS7: [[f32; 7]; 7] = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [
        1.0,
        0.623_489_8,
        -0.222_520_93,
        -0.900_968_85,
        -0.900_968_85,
        -0.222_520_93,
        0.623_489_8,
    ],
    [
        1.0,
        -0.222_520_93,
        -0.900_968_85,
        0.623_489_8,
        0.623_489_8,
        -0.900_968_85,
        -0.222_520_93,
    ],
    [
        1.0,
        -0.900_968_85,
        0.623_489_8,
        -0.222_520_93,
        -0.222_520_93,
        0.623_489_8,
        -0.900_968_85,
    ],
    [
        1.0,
        -0.900_968_85,
        0.623_489_8,
        -0.222_520_93,
        -0.222_520_93,
        0.623_489_8,
        -0.900_968_85,
    ],
    [
        1.0,
        -0.222_520_93,
        -0.900_968_85,
        0.623_489_8,
        0.623_489_8,
        -0.900_968_85,
        -0.222_520_93,
    ],
    [
        1.0,
        0.623_489_8,
        -0.222_520_93,
        -0.900_968_85,
        -0.900_968_85,
        -0.222_520_93,
        0.623_489_8,
    ],
];
const SIN7: [[f32; 7]; 7] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [
        0.0,
        0.781_831_5,
        0.974_927_9,
        0.433_883_73,
        -0.433_883_73,
        -0.974_927_9,
        -0.781_831_5,
    ],
    [
        0.0,
        0.974_927_9,
        -0.433_883_73,
        -0.781_831_5,
        0.781_831_5,
        0.433_883_73,
        -0.974_927_9,
    ],
    [
        0.0,
        0.433_883_73,
        -0.781_831_5,
        0.974_927_9,
        -0.974_927_9,
        0.781_831_5,
        -0.433_883_73,
    ],
    [
        0.0,
        -0.433_883_73,
        0.781_831_5,
        -0.974_927_9,
        0.974_927_9,
        -0.781_831_5,
        0.433_883_73,
    ],
    [
        0.0,
        -0.974_927_9,
        0.433_883_73,
        0.781_831_5,
        -0.781_831_5,
        -0.433_883_73,
        0.974_927_9,
    ],
    [
        0.0,
        -0.781_831_5,
        -0.974_927_9,
        -0.433_883_73,
        0.433_883_73,
        0.974_927_9,
        0.781_831_5,
    ],
];

const COS11: [[f32; 11]; 11] = [
    [1.0; 11],
    [
        1.0,
        0.841_253_5,
        0.415_415,
        -0.142_314_84,
        -0.654_860_73,
        -0.959_493,
        -0.959_493,
        -0.654_860_73,
        -0.142_314_84,
        0.415_415,
        0.841_253_5,
    ],
    [
        1.0,
        0.415_415,
        -0.654_860_73,
        -0.959_493,
        -0.142_314_84,
        0.841_253_5,
        0.841_253_5,
        -0.142_314_84,
        -0.959_493,
        -0.654_860_73,
        0.415_415,
    ],
    [
        1.0,
        -0.142_314_84,
        -0.959_493,
        0.415_415,
        0.841_253_5,
        -0.654_860_73,
        -0.654_860_73,
        0.841_253_5,
        0.415_415,
        -0.959_493,
        -0.142_314_84,
    ],
    [
        1.0,
        -0.654_860_73,
        -0.142_314_84,
        0.841_253_5,
        -0.959_493,
        0.415_415,
        0.415_415,
        -0.959_493,
        0.841_253_5,
        -0.142_314_84,
        -0.654_860_73,
    ],
    [
        1.0,
        -0.959_493,
        0.841_253_5,
        -0.654_860_73,
        0.415_415,
        -0.142_314_84,
        -0.142_314_84,
        0.415_415,
        -0.654_860_73,
        0.841_253_5,
        -0.959_493,
    ],
    [
        1.0,
        -0.959_493,
        0.841_253_5,
        -0.654_860_73,
        0.415_415,
        -0.142_314_84,
        -0.142_314_84,
        0.415_415,
        -0.654_860_73,
        0.841_253_5,
        -0.959_493,
    ],
    [
        1.0,
        -0.654_860_73,
        -0.142_314_84,
        0.841_253_5,
        -0.959_493,
        0.415_415,
        0.415_415,
        -0.959_493,
        0.841_253_5,
        -0.142_314_84,
        -0.654_860_73,
    ],
    [
        1.0,
        -0.142_314_84,
        -0.959_493,
        0.415_415,
        0.841_253_5,
        -0.654_860_73,
        -0.654_860_73,
        0.841_253_5,
        0.415_415,
        -0.959_493,
        -0.142_314_84,
    ],
    [
        1.0,
        0.415_415,
        -0.654_860_73,
        -0.959_493,
        -0.142_314_84,
        0.841_253_5,
        0.841_253_5,
        -0.142_314_84,
        -0.959_493,
        -0.654_860_73,
        0.415_415,
    ],
    [
        1.0,
        0.841_253_5,
        0.415_415,
        -0.142_314_84,
        -0.654_860_73,
        -0.959_493,
        -0.959_493,
        -0.654_860_73,
        -0.142_314_84,
        0.415_415,
        0.841_253_5,
    ],
];
const SIN11: [[f32; 11]; 11] = [
    [0.0; 11],
    [
        0.0,
        0.540_640_83,
        0.909_632,
        0.989_821_43,
        0.755_749_6,
        0.281_732_56,
        -0.281_732_56,
        -0.755_749_6,
        -0.989_821_43,
        -0.909_632,
        -0.540_640_83,
    ],
    [
        0.0,
        0.909_632,
        0.755_749_6,
        -0.281_732_56,
        -0.989_821_43,
        -0.540_640_83,
        0.540_640_83,
        0.989_821_43,
        0.281_732_56,
        -0.755_749_6,
        -0.909_632,
    ],
    [
        0.0,
        0.989_821_43,
        -0.281_732_56,
        -0.909_632,
        0.540_640_83,
        0.755_749_6,
        -0.755_749_6,
        -0.540_640_83,
        0.909_632,
        0.281_732_56,
        -0.989_821_43,
    ],
    [
        0.0,
        0.755_749_6,
        -0.989_821_43,
        0.540_640_83,
        0.281_732_56,
        -0.909_632,
        0.909_632,
        -0.281_732_56,
        -0.540_640_83,
        0.989_821_43,
        -0.755_749_6,
    ],
    [
        0.0,
        0.281_732_56,
        -0.540_640_83,
        0.755_749_6,
        -0.909_632,
        0.989_821_43,
        -0.989_821_43,
        0.909_632,
        -0.755_749_6,
        0.540_640_83,
        -0.281_732_56,
    ],
    [
        0.0,
        -0.281_732_56,
        0.540_640_83,
        -0.755_749_6,
        0.909_632,
        -0.989_821_43,
        0.989_821_43,
        -0.909_632,
        0.755_749_6,
        -0.540_640_83,
        0.281_732_56,
    ],
    [
        0.0,
        -0.755_749_6,
        0.989_821_43,
        -0.540_640_83,
        -0.281_732_56,
        0.909_632,
        -0.909_632,
        0.281_732_56,
        0.540_640_83,
        -0.989_821_43,
        0.755_749_6,
    ],
    [
        0.0,
        -0.989_821_43,
        0.281_732_56,
        0.909_632,
        -0.540_640_83,
        -0.755_749_6,
        0.755_749_6,
        0.540_640_83,
        -0.909_632,
        -0.281_732_56,
        0.989_821_43,
    ],
    [
        0.0,
        -0.909_632,
        -0.755_749_6,
        0.281_732_56,
        0.989_821_43,
        0.540_640_83,
        -0.540_640_83,
        -0.989_821_43,
        -0.281_732_56,
        0.755_749_6,
        0.909_632,
    ],
    [
        0.0,
        -0.540_640_83,
        -0.909_632,
        -0.989_821_43,
        -0.755_749_6,
        -0.281_732_56,
        0.281_732_56,
        0.755_749_6,
        0.989_821_43,
        0.909_632,
        0.540_640_83,
    ],
];

pub(crate) fn prime_batched_f32<const R: usize, S>(storage: &mut S, inverse: bool)
where
    S: FftStorage<f32>,
{
    assert!(matches!(R, 3 | 7 | 11), "unsupported prime radix");
    assert_eq!(storage.rows(), R, "prime radix storage row count mismatch");
    assert!(
        storage.cols() <= LANES,
        "prime radix batch width exceeds lanes"
    );

    let cols = storage.cols();
    let mut re = [[0.0_f32; LANES]; R];
    let mut im = [[0.0_f32; LANES]; R];
    let mut out_re = [[0.0_f32; LANES]; R];
    let mut out_im = [[0.0_f32; LANES]; R];
    for lane in 0..cols {
        for point in 0..R {
            re[point][lane] = storage.load_re(point, lane);
            im[point][lane] = storage.load_im(point, lane);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
        // SAFETY: runtime feature detection proves AVX/FMA availability.
        unsafe {
            prime_batched_f32_avx_fma::<R>(&re, &im, &mut out_re, &mut out_im, inverse);
        }
    } else {
        prime_batched_f32_scalar::<R>(&re, &im, &mut out_re, &mut out_im, inverse, cols);
    }
    #[cfg(not(target_arch = "x86_64"))]
    prime_batched_f32_scalar::<R>(&re, &im, &mut out_re, &mut out_im, inverse, cols);
    store_outputs::<R, S>(storage, cols, &out_re, &out_im);
}

/// Execute a real-input prime-radix column sweep directly over row-major
/// sweep storage.
///
/// # Proof
///
/// For prime radix `R`, the DFT kernel is
/// `Y[k] = sum_j X[j] * (cos(2*pi*j*k/R) - i sin(2*pi*j*k/R))`; inverse uses
/// the conjugate kernel. The const `INVERSE` selects the sign before
/// monomorphization, so the forward real-input adapter supplies zero imaginary
/// samples and the inverse adapter ignores the computed imaginary component.
pub(crate) fn prime_real_sweep<const R: usize, const INVERSE: bool, S>(sweep: &mut S)
where
    S: RealSweep<INVERSE, R>,
{
    assert!(matches!(R, 3 | 7 | 11), "unsupported prime sweep radix");
    let n2 = sweep.n2();
    let mut col = 0usize;
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
        // SAFETY: runtime feature detection proves AVX/FMA availability. The
        // helper only reads and writes full eight-column spans inside checked
        // row-major buffers.
        col = unsafe { prime_real_sweep_avx_fma::<R, INVERSE, S>(sweep, n2) };
    }

    prime_real_sweep_scalar::<R, INVERSE, S>(sweep, n2, col);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn prime_real_sweep_avx_fma<const R: usize, const INVERSE: bool, S>(
    sweep: &mut S,
    n2: usize,
) -> usize
where
    S: RealSweep<INVERSE, R>,
{
    let mut col = 0usize;
    unsafe {
        while col + LANES <= n2 {
            for k in 0..R {
                let mut yr = _mm256_setzero_ps();
                let mut yi = _mm256_setzero_ps();
                for j in 0..R {
                    let xr = load_sweep_re::<R, INVERSE, S>(sweep, j, col);
                    let xi = load_sweep_im::<R, INVERSE, S>(sweep, j, col);
                    let (c, s) = coefficient::<R>(k, j);
                    let signed_s = if INVERSE { -s } else { s };
                    yr = fmadd(signed_s, xi, fmadd(c, xr, yr));
                    yi = fmadd(-signed_s, xr, fmadd(c, xi, yi));
                }
                sweep.store_avx(k, col, yr, yi);
            }
            col += LANES;
        }
    }
    col
}

fn prime_real_sweep_scalar<const R: usize, const INVERSE: bool, S>(
    sweep: &mut S,
    n2: usize,
    start_col: usize,
) where
    S: RealSweep<INVERSE, R>,
{
    for col in start_col..n2 {
        for k in 0..R {
            let mut yr = 0.0_f32;
            let mut yi = 0.0_f32;
            for j in 0..R {
                let xr = sweep.load_re(j, col);
                let xi = sweep.load_im(j, col);
                let (c, s) = coefficient::<R>(k, j);
                let signed_s = if INVERSE { -s } else { s };
                yr = signed_s.mul_add(xi, c.mul_add(xr, yr));
                yi = (-signed_s).mul_add(xr, c.mul_add(xi, yi));
            }
            sweep.store(k, col, yr, yi);
        }
    }
}

fn store_outputs<const R: usize, S>(
    storage: &mut S,
    cols: usize,
    re: &[[f32; LANES]; R],
    im: &[[f32; LANES]; R],
) where
    S: FftStorage<f32>,
{
    for lane in 0..cols {
        for point in 0..R {
            storage.store(point, lane, re[point][lane], im[point][lane]);
        }
    }
}

#[inline(always)]
fn coefficient<const R: usize>(k: usize, j: usize) -> (f32, f32) {
    match R {
        3 => (COS3[k][j], SIN3[k][j]),
        5 => (COS5[k][j], SIN5[k][j]),
        7 => (COS7[k][j], SIN7[k][j]),
        11 => (COS11[k][j], SIN11[k][j]),
        _ => unreachable!("unsupported prime radix"),
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
unsafe fn load_sweep_re<const R: usize, const INVERSE: bool, S>(
    sweep: &S,
    point: usize,
    col: usize,
) -> __m256
where
    S: RealSweep<INVERSE, R>,
{
    unsafe { _mm256_loadu_ps(sweep.re_ptr(point, col)) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn load_sweep_im<const R: usize, const INVERSE: bool, S>(
    sweep: &S,
    point: usize,
    col: usize,
) -> __m256
where
    S: RealSweep<INVERSE, R>,
{
    unsafe { _mm256_loadu_ps(sweep.im_ptr(point, col)) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn store(row: &mut [f32; LANES], value: __m256) {
    // SAFETY: every row has exactly LANES contiguous f32 values.
    unsafe { _mm256_storeu_ps(row.as_mut_ptr(), value) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn fmadd(a: f32, b: __m256, c: __m256) -> __m256 {
    _mm256_fmadd_ps(_mm256_set1_ps(a), b, c)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn prime_batched_f32_avx_fma<const R: usize>(
    re: &[[f32; LANES]; R],
    im: &[[f32; LANES]; R],
    out_re: &mut [[f32; LANES]; R],
    out_im: &mut [[f32; LANES]; R],
    inverse: bool,
) {
    // SAFETY: caller checked AVX/FMA. Each vector lane is an independent FFT
    // instance, so all operations are lane-wise and require no cross-lane
    // shuffle or permutation.
    unsafe {
        for k in 0..R {
            let mut yr = _mm256_setzero_ps();
            let mut yi = _mm256_setzero_ps();
            for j in 0..R {
                let xr = load(&re[j]);
                let xi = load(&im[j]);
                let (c, s) = coefficient::<R>(k, j);
                let signed_s = if inverse { -s } else { s };
                yr = fmadd(signed_s, xi, fmadd(c, xr, yr));
                yi = fmadd(-signed_s, xr, fmadd(c, xi, yi));
            }
            store(&mut out_re[k], yr);
            store(&mut out_im[k], yi);
        }
    }
}

fn prime_batched_f32_scalar<const R: usize>(
    re: &[[f32; LANES]; R],
    im: &[[f32; LANES]; R],
    out_re: &mut [[f32; LANES]; R],
    out_im: &mut [[f32; LANES]; R],
    inverse: bool,
    cols: usize,
) {
    for lane in 0..cols {
        for k in 0..R {
            let mut yr = 0.0_f32;
            let mut yi = 0.0_f32;
            for j in 0..R {
                let xr = re[j][lane];
                let xi = im[j][lane];
                let (c, s) = coefficient::<R>(k, j);
                let signed_s = if inverse { -s } else { s };
                yr = signed_s.mul_add(xi, c.mul_add(xr, yr));
                yi = (-signed_s).mul_add(xr, c.mul_add(xi, yi));
            }
            out_re[k][lane] = yr;
            out_im[k][lane] = yi;
        }
    }
}
