//! Mathematical primitives and interpolations for NUFFT Kaiser-Bessel operations.

use ndarray::Array1;
use std::f64::consts::PI;

/// Approximates the modified Bessel function of order zero (I0).
/// This implementation relies on a highly optimized polynomial fit.
#[inline]
pub fn i0(z: f64) -> f64 {
    let t = z.abs();
    if t < 3.75 {
        let y = (t / 3.75) * (t / 3.75);
        1.0 + y
            * (3.515_623_7
                + y * (3.089_942_4
                    + y * (1.206_749_2 + y * (0.265_973_2 + y * (0.036_076_8 + y * 0.004_581_3)))))
    } else {
        let y = 3.75 / t;
        (t.exp() / t.sqrt())
            * (0.398_942_28
                + y * (0.013_285_92
                    + y * (0.002_253_19
                        + y * (-0.001_575_65
                            + y * (0.009_162_81
                                + y * (-0.020_577_06
                                    + y * (0.026_355_37
                                        + y * (-0.016_476_33 + y * 0.003_923_77))))))))
    }
}

/// Evaluates the Kaiser-Bessel spreading kernel.
#[inline]
pub fn kb_kernel(x: f64, w: f64, beta: f64, i0_beta: f64) -> f64 {
    let u2 = (x / w) * (x / w);
    if u2 >= 1.0 {
        0.0
    } else {
        i0(beta * f64::sqrt(1.0 - u2)) / i0_beta
    }
}

/// Evaluates the analytical Fourier Transform of the Kaiser-Bessel kernel.
pub fn kb_kernel_ft(xi: f64, w: usize, beta: f64, i0_beta: f64) -> f64 {
    let two_pi_w_xi = 2.0 * PI * w as f64 * xi;
    let z_sq = beta * beta - two_pi_w_xi * two_pi_w_xi;
    let prefix = 2.0 * w as f64 / i0_beta;
    if z_sq.abs() < 1e-30 {
        prefix
    } else if z_sq > 0.0 {
        let s = z_sq.sqrt();
        prefix * s.sinh() / s
    } else {
        let s = (-z_sq).sqrt();
        prefix * s.sin() / s
    }
}

/// Precomputes the 1D deconvolution array across the active dimension.
pub fn axis_deconv(
    n: usize,
    m: usize,
    kernel_width: usize,
    beta: f64,
    i0_beta: f64,
) -> Array1<f64> {
    Array1::from_shape_fn(n, |k| {
        let xi = fft_signed_index(k, n) as f64 / m as f64;
        1.0 / kb_kernel_ft(xi, kernel_width, beta, i0_beta)
    })
}

/// Maps an unsigned index from `[0..N]` to a signed Fourier-centric integer inside `[-N/2..N/2]`.
pub fn fft_signed_index(index: usize, len: usize) -> i64 {
    if index <= len / 2 {
        index as i64
    } else {
        index as i64 - len as i64
    }
}

/// Calculates the ideal bucket allocation length mapping kernel domains.
pub fn bucket_count(len: usize, kernel_width: usize) -> usize {
    len.max(1).div_ceil((2 * kernel_width + 1).max(1)).max(1)
}
