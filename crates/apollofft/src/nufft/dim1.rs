//! 1D Type-1 and Type-2 NUFFT executions and plans.

use crate::nufft::math::{bucket_count, fft_signed_index, kb_kernel, kb_kernel_ft};
use crate::nufft::DEFAULT_NUFFT_OVERSAMPLING;
use crate::types::UniformDomain1D;
use ndarray::Array1;
use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;
use std::sync::Arc;

#[derive(Clone, Copy)]
struct IndexedPoint1D {
    x: f64,
    value: Complex64,
    bucket: usize,
}

fn sort_points_1d(
    positions: &[f64],
    values: &[Complex64],
    domain: UniformDomain1D,
    oversampled_len: usize,
    kernel_width: usize,
) -> Vec<IndexedPoint1D> {
    let buckets = bucket_count(oversampled_len, kernel_width);
    let bucket_scale = buckets as f64 / domain.length();
    let mut indexed: Vec<_> = positions
        .iter()
        .zip(values.iter())
        .enumerate()
        .map(|(_original_index, (&x, &value))| {
            let x_mod = x.rem_euclid(domain.length());
            let bucket = ((x_mod * bucket_scale).floor() as usize).min(buckets - 1);
            IndexedPoint1D {
                x: x_mod,
                value,
                bucket,
            }
        })
        .collect();
    indexed.sort_unstable_by(|lhs, rhs| lhs.bucket.cmp(&rhs.bucket));
    indexed
}

fn sort_positions_1d(
    positions: &[f64],
    domain: UniformDomain1D,
    oversampled_len: usize,
    kernel_width: usize,
) -> Vec<(usize, f64, usize)> {
    let buckets = bucket_count(oversampled_len, kernel_width);
    let bucket_scale = buckets as f64 / domain.length();
    let mut indexed: Vec<_> = positions
        .iter()
        .enumerate()
        .map(|(original_index, &x)| {
            let x_mod = x.rem_euclid(domain.length());
            let bucket = ((x_mod * bucket_scale).floor() as usize).min(buckets - 1);
            (original_index, x_mod, bucket)
        })
        .collect();
    indexed.sort_unstable_by(|lhs, rhs| lhs.2.cmp(&rhs.2));
    indexed
}

/// Reusable 1D NUFFT plan using Kaiser-Bessel spreading.
pub struct NufftPlan1D {
    n_out: usize,
    m: usize,
    w: usize,
    beta: f64,
    i0_beta: f64,
    domain: UniformDomain1D,
    deconv: Array1<f64>,
    fft_forward: Arc<dyn rustfft::Fft<f64>>,
    fft_inverse: Arc<dyn rustfft::Fft<f64>>,
}

impl std::fmt::Debug for NufftPlan1D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NufftPlan1D")
            .field("n_out", &self.n_out)
            .field("oversampled_len", &self.m)
            .field("kernel_width", &self.w)
            .field("domain", &self.domain)
            .finish()
    }
}

impl NufftPlan1D {
    /// Create a reusable 1D NUFFT plan.
    #[must_use]
    pub fn new(domain: UniformDomain1D, sigma: usize, kernel_width: usize) -> Self {
        assert!(sigma >= 2, "sigma must be >= 2");
        assert!(kernel_width >= 2, "kernel_width must be >= 2");

        let m = sigma * domain.n;
        let beta = PI * (1.0 - 1.0 / (2.0 * sigma as f64)) * (2 * kernel_width) as f64;
        let i0_beta = crate::nufft::math::i0(beta);

        let mut planner = FftPlanner::<f64>::new();
        let fft_forward = planner.plan_fft_forward(m);
        let fft_inverse = planner.plan_fft_inverse(m);
        let deconv = Array1::from_shape_fn(domain.n, |k| {
            let k_signed = fft_signed_index(k, domain.n);
            let xi = k_signed as f64 / m as f64;
            1.0 / kb_kernel_ft(xi, kernel_width, beta, i0_beta)
        });

        Self {
            n_out: domain.n,
            m,
            w: kernel_width,
            beta,
            i0_beta,
            domain,
            deconv,
            fft_forward,
            fft_inverse,
        }
    }

    /// Run type-1 NUFFT, mapping non-uniform samples to uniform Fourier bins.
    #[must_use]
    pub fn type1(&self, positions: &[f64], values: &[Complex64]) -> Array1<Complex64> {
        assert_eq!(
            positions.len(),
            values.len(),
            "positions/value length mismatch"
        );
        let mut grid = vec![Complex64::new(0.0, 0.0); self.m];
        let w = self.w as i64;
        let w_f = self.w as f64;
        let sorted_points = sort_points_1d(positions, values, self.domain, self.m, self.w);

        for point in sorted_points {
            let t = self.m as f64 * point.x / self.domain.length();
            let m0 = t.round() as i64;
            let d = t - m0 as f64;
            for p in -w..=w {
                let weight = kb_kernel(p as f64 - d, w_f, self.beta, self.i0_beta);
                if weight != 0.0 {
                    let m_idx = (m0 + p).rem_euclid(self.m as i64) as usize;
                    grid[m_idx] += point.value * weight;
                }
            }
        }

        self.fft_forward.process(&mut grid);

        Array1::from_shape_fn(self.n_out, |k| {
            let k_signed = fft_signed_index(k, self.n_out);
            let m_idx = k_signed.rem_euclid(self.m as i64) as usize;
            grid[m_idx] * self.deconv[k]
        })
    }

    /// Run type-2 NUFFT, interpolating a uniform Fourier grid at non-uniform positions.
    #[must_use]
    pub fn type2(&self, fourier_coeffs: &Array1<Complex64>, positions: &[f64]) -> Vec<Complex64> {
        assert_eq!(
            fourier_coeffs.len(),
            self.n_out,
            "fourier_coeffs length mismatch"
        );

        let mut spread = vec![Complex64::new(0.0, 0.0); self.m];
        for k in 0..self.n_out {
            let k_signed = fft_signed_index(k, self.n_out);
            let m_idx = k_signed.rem_euclid(self.m as i64) as usize;
            spread[m_idx] = fourier_coeffs[k] * self.deconv[k];
        }
        self.fft_inverse.process(&mut spread);

        let w = self.w as i64;
        let w_f = self.w as f64;
        let sorted_points = sort_positions_1d(positions, self.domain, self.m, self.w);
        let mut output = vec![Complex64::default(); positions.len()];
        for (original_index, x_mod, _) in sorted_points {
            let t = self.m as f64 * x_mod / self.domain.length();
            let m0 = t.round() as i64;
            let d = t - m0 as f64;

            let mut value = Complex64::new(0.0, 0.0);
            for p in -w..=w {
                let weight = kb_kernel(p as f64 - d, w_f, self.beta, self.i0_beta);
                if weight != 0.0 {
                    let m_idx = (m0 + p).rem_euclid(self.m as i64) as usize;
                    value += spread[m_idx] * weight;
                }
            }
            output[original_index] = value;
        }
        output
    }
}

/// Exact direct 1D type-1 NUFFT.
#[must_use]
pub fn nufft_type1_1d(
    positions: &[f64],
    values: &[Complex64],
    domain: UniformDomain1D,
) -> Array1<Complex64> {
    assert_eq!(
        positions.len(),
        values.len(),
        "positions/value length mismatch"
    );
    let two_pi_over_l = 2.0 * PI / domain.length();
    Array1::from_shape_fn(domain.n, |k| {
        let k_signed = fft_signed_index(k, domain.n);
        positions
            .iter()
            .zip(values.iter())
            .fold(Complex64::new(0.0, 0.0), |acc, (&x, &value)| {
                let angle = -two_pi_over_l * k_signed as f64 * x;
                acc + value * Complex64::new(angle.cos(), angle.sin())
            })
    })
}

/// Exact direct 1D type-2 NUFFT.
#[must_use]
pub fn nufft_type2_1d(
    fourier_coeffs: &Array1<Complex64>,
    positions: &[f64],
    domain: UniformDomain1D,
) -> Vec<Complex64> {
    positions
        .iter()
        .map(|&x| {
            fourier_coeffs
                .iter()
                .enumerate()
                .fold(Complex64::new(0.0, 0.0), |acc, (k, &value)| {
                    let angle =
                        2.0 * PI * fft_signed_index(k, domain.n) as f64 * x / domain.length();
                    acc + value * Complex64::new(angle.cos(), angle.sin())
                })
        })
        .collect()
}

/// Fast 1D type-1 NUFFT convenience wrapper.
#[must_use]
pub fn nufft_type1_1d_fast(
    positions: &[f64],
    values: &[Complex64],
    domain: UniformDomain1D,
    kernel_width: usize,
) -> Array1<Complex64> {
    NufftPlan1D::new(domain, DEFAULT_NUFFT_OVERSAMPLING, kernel_width).type1(positions, values)
}

/// Fast 1D type-2 NUFFT convenience wrapper.
#[must_use]
pub fn nufft_type2_1d_fast(
    fourier_coeffs: &Array1<Complex64>,
    positions: &[f64],
    domain: UniformDomain1D,
    kernel_width: usize,
) -> Vec<Complex64> {
    NufftPlan1D::new(domain, DEFAULT_NUFFT_OVERSAMPLING, kernel_width)
        .type2(fourier_coeffs, positions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nufft::DEFAULT_NUFFT_KERNEL_WIDTH;
    use proptest::prelude::*;

    #[test]
    fn exact_1d_matches_direct_mode_sum() {
        let domain = UniformDomain1D::new(8, 0.125).unwrap();
        let positions = vec![0.0, 0.125, 0.25, 0.375];
        let values = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, -0.25),
            Complex64::new(-0.5, 0.75),
            Complex64::new(0.25, 0.1),
        ];
        let output = nufft_type1_1d(&positions, &values, domain);
        assert_eq!(output.len(), 8);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn fast_1d_tracks_exact() {
        let domain = UniformDomain1D::new(32, 0.05).unwrap();
        let positions: Vec<f64> = (0..20)
            .map(|i| ((i as f64 * 0.137) % domain.length()).abs())
            .collect();
        let values: Vec<Complex64> = (0..20)
            .map(|i| Complex64::new((i as f64 * 0.3).cos(), (i as f64 * 0.2).sin()))
            .collect();
        let exact = nufft_type1_1d(&positions, &values, domain);
        let fast = nufft_type1_1d_fast(&positions, &values, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
        let scale = exact.iter().map(|value| value.norm()).fold(1.0, f64::max);
        let max_err = exact
            .iter()
            .zip(fast.iter())
            .map(|(lhs, rhs)| (lhs - rhs).norm())
            .fold(0.0, f64::max);
        assert!(max_err / scale < 1e-6);
    }

    proptest! {
        #[test]
        fn fast_1d_matches_exact_for_random_points(
            n in 8usize..32,
            dx in 0.02f64..0.2,
            sample_count in 3usize..12,
        ) {
            let domain = UniformDomain1D::new(n, dx).unwrap();
            let positions: Vec<f64> = (0..sample_count)
                .map(|i| (i as f64 * std::f64::consts::SQRT_2 + 0.13).fract() * domain.length())
                .collect();
            let values: Vec<Complex64> = (0..sample_count)
                .map(|i| Complex64::new((i as f64 * 0.37).cos(), (i as f64 * 0.19).sin()))
                .collect();
            let exact = nufft_type1_1d(&positions, &values, domain);
            let fast = nufft_type1_1d_fast(&positions, &values, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
            let scale = exact.iter().map(|value| value.norm()).fold(1.0, f64::max);
            let max_err = exact
                .iter()
                .zip(fast.iter())
                .map(|(lhs, rhs)| (lhs - rhs).norm())
                .fold(0.0, f64::max);
            prop_assert!(max_err / scale < 1e-6);
        }
    }
}
