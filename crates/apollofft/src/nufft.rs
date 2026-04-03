//! Non-uniform FFT utilities and reusable NUFFT plans.

use crate::types::{UniformDomain1D, UniformGrid3D};
use ndarray::{Array1, Array3};
use num_complex::Complex64;
use rustfft::FftPlanner;
use std::cmp::Ordering;
use std::f64::consts::PI;
use std::sync::Arc;

/// Default kernel half-width.
pub const DEFAULT_NUFFT_KERNEL_WIDTH: usize = 6;

/// Default oversampling factor.
pub const DEFAULT_NUFFT_OVERSAMPLING: usize = 2;

#[inline]
fn i0(z: f64) -> f64 {
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

#[inline]
fn kb_kernel(x: f64, w: f64, beta: f64, i0_beta: f64) -> f64 {
    let u2 = (x / w) * (x / w);
    if u2 >= 1.0 {
        0.0
    } else {
        i0(beta * f64::sqrt(1.0 - u2)) / i0_beta
    }
}

fn kb_kernel_ft(xi: f64, w: usize, beta: f64, i0_beta: f64) -> f64 {
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

#[derive(Clone, Copy)]
struct IndexedPoint1D {
    original_index: usize,
    x: f64,
    value: Complex64,
    bucket: usize,
}

#[derive(Clone, Copy)]
struct IndexedPoint3D {
    x: f64,
    y: f64,
    z: f64,
    value: Complex64,
    bucket: usize,
}

fn bucket_count(len: usize, kernel_width: usize) -> usize {
    len.max(1).div_ceil((2 * kernel_width + 1).max(1)).max(1)
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
        .map(|(original_index, (&x, &value))| {
            let x_mod = x.rem_euclid(domain.length());
            let bucket = ((x_mod * bucket_scale).floor() as usize).min(buckets - 1);
            IndexedPoint1D {
                original_index,
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

fn sort_points_3d(
    positions: &[(f64, f64, f64)],
    values: &[Complex64],
    grid: UniformGrid3D,
    oversampled_shape: (usize, usize, usize),
    kernel_width: usize,
) -> Vec<IndexedPoint3D> {
    let (lx, ly, lz) = grid.lengths();
    let bx = bucket_count(oversampled_shape.0, kernel_width);
    let by = bucket_count(oversampled_shape.1, kernel_width);
    let bz = bucket_count(oversampled_shape.2, kernel_width);
    let sx = bx as f64 / lx;
    let sy = by as f64 / ly;
    let sz = bz as f64 / lz;
    let mut indexed: Vec<_> = positions
        .iter()
        .zip(values.iter())
        .map(|(&(x, y, z), &value)| {
            let x_mod = x.rem_euclid(lx);
            let y_mod = y.rem_euclid(ly);
            let z_mod = z.rem_euclid(lz);
            let ix = ((x_mod * sx).floor() as usize).min(bx - 1);
            let iy = ((y_mod * sy).floor() as usize).min(by - 1);
            let iz = ((z_mod * sz).floor() as usize).min(bz - 1);
            IndexedPoint3D {
                x: x_mod,
                y: y_mod,
                z: z_mod,
                value,
                bucket: (ix * by + iy) * bz + iz,
            }
        })
        .collect();
    indexed.sort_unstable_by(|lhs, rhs| {
        lhs.bucket
            .cmp(&rhs.bucket)
            .then_with(|| lhs.x.partial_cmp(&rhs.x).unwrap_or(Ordering::Equal))
            .then_with(|| lhs.y.partial_cmp(&rhs.y).unwrap_or(Ordering::Equal))
            .then_with(|| lhs.z.partial_cmp(&rhs.z).unwrap_or(Ordering::Equal))
    });
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
        let i0_beta = i0(beta);

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

/// Reusable 3D type-1 NUFFT plan using separable Kaiser-Bessel spreading.
pub struct NufftPlan3D {
    grid: UniformGrid3D,
    mx: usize,
    my: usize,
    mz: usize,
    w: usize,
    beta: f64,
    i0_beta: f64,
    deconv_x: Array1<f64>,
    deconv_y: Array1<f64>,
    deconv_z: Array1<f64>,
    fft_x: Arc<dyn rustfft::Fft<f64>>,
    fft_y: Arc<dyn rustfft::Fft<f64>>,
    fft_z: Arc<dyn rustfft::Fft<f64>>,
}

impl std::fmt::Debug for NufftPlan3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NufftPlan3D")
            .field("grid", &self.grid)
            .field("oversampled_shape", &(self.mx, self.my, self.mz))
            .field("kernel_width", &self.w)
            .finish()
    }
}

impl NufftPlan3D {
    /// Create a reusable 3D NUFFT plan.
    #[must_use]
    pub fn new(grid: UniformGrid3D, sigma: usize, kernel_width: usize) -> Self {
        assert!(sigma >= 2, "sigma must be >= 2");
        assert!(kernel_width >= 2, "kernel_width must be >= 2");

        let mx = sigma * grid.nx;
        let my = sigma * grid.ny;
        let mz = sigma * grid.nz;
        let beta = PI * (1.0 - 1.0 / (2.0 * sigma as f64)) * (2 * kernel_width) as f64;
        let i0_beta = i0(beta);
        let mut planner = FftPlanner::<f64>::new();
        let fft_x = planner.plan_fft_forward(mx);
        let fft_y = planner.plan_fft_forward(my);
        let fft_z = planner.plan_fft_forward(mz);
        let deconv_x = axis_deconv(grid.nx, mx, kernel_width, beta, i0_beta);
        let deconv_y = axis_deconv(grid.ny, my, kernel_width, beta, i0_beta);
        let deconv_z = axis_deconv(grid.nz, mz, kernel_width, beta, i0_beta);

        Self {
            grid,
            mx,
            my,
            mz,
            w: kernel_width,
            beta,
            i0_beta,
            deconv_x,
            deconv_y,
            deconv_z,
            fft_x,
            fft_y,
            fft_z,
        }
    }

    /// Run type-1 3D NUFFT.
    #[must_use]
    pub fn type1(&self, positions: &[(f64, f64, f64)], values: &[Complex64]) -> Array3<Complex64> {
        assert_eq!(
            positions.len(),
            values.len(),
            "positions/value length mismatch"
        );
        let (lx, ly, lz) = self.grid.lengths();
        let mut grid = Array3::<Complex64>::zeros((self.mx, self.my, self.mz));
        let w = self.w as i64;
        let w_f = self.w as f64;

        let sorted_points = sort_points_3d(
            positions,
            values,
            self.grid,
            (self.mx, self.my, self.mz),
            self.w,
        );
        let mut wx = vec![0.0_f64; 2 * self.w + 1];
        let mut wy = vec![0.0_f64; 2 * self.w + 1];
        let mut wz = vec![0.0_f64; 2 * self.w + 1];

        for point in sorted_points {
            let tx = self.mx as f64 * point.x / lx;
            let ty = self.my as f64 * point.y / ly;
            let tz = self.mz as f64 * point.z / lz;
            let m0x = tx.round() as i64;
            let m0y = ty.round() as i64;
            let m0z = tz.round() as i64;
            let dx = tx - m0x as f64;
            let dy = ty - m0y as f64;
            let dz = tz - m0z as f64;
            for (offset, p) in (-w..=w).enumerate() {
                wx[offset] = kb_kernel(p as f64 - dx, w_f, self.beta, self.i0_beta);
                wy[offset] = kb_kernel(p as f64 - dy, w_f, self.beta, self.i0_beta);
                wz[offset] = kb_kernel(p as f64 - dz, w_f, self.beta, self.i0_beta);
            }

            for (px, &wxv) in wx.iter().enumerate() {
                if wxv == 0.0 {
                    continue;
                }
                let ix = (m0x + px as i64 - w).rem_euclid(self.mx as i64) as usize;
                for (py, &wyv) in wy.iter().enumerate() {
                    if wyv == 0.0 {
                        continue;
                    }
                    let iy = (m0y + py as i64 - w).rem_euclid(self.my as i64) as usize;
                    let wxy = wxv * wyv;
                    for (pz, &wzv) in wz.iter().enumerate() {
                        if wzv == 0.0 {
                            continue;
                        }
                        let iz = (m0z + pz as i64 - w).rem_euclid(self.mz as i64) as usize;
                        grid[[ix, iy, iz]] += point.value * (wxy * wzv);
                    }
                }
            }
        }

        let mut row_z = vec![Complex64::new(0.0, 0.0); self.mz];
        for ix in 0..self.mx {
            for iy in 0..self.my {
                for iz in 0..self.mz {
                    row_z[iz] = grid[[ix, iy, iz]];
                }
                self.fft_z.process(&mut row_z);
                for iz in 0..self.mz {
                    grid[[ix, iy, iz]] = row_z[iz];
                }
            }
        }

        let mut row_y = vec![Complex64::new(0.0, 0.0); self.my];
        for ix in 0..self.mx {
            for iz in 0..self.mz {
                for iy in 0..self.my {
                    row_y[iy] = grid[[ix, iy, iz]];
                }
                self.fft_y.process(&mut row_y);
                for iy in 0..self.my {
                    grid[[ix, iy, iz]] = row_y[iy];
                }
            }
        }

        let mut row_x = vec![Complex64::new(0.0, 0.0); self.mx];
        for iy in 0..self.my {
            for iz in 0..self.mz {
                for ix in 0..self.mx {
                    row_x[ix] = grid[[ix, iy, iz]];
                }
                self.fft_x.process(&mut row_x);
                for ix in 0..self.mx {
                    grid[[ix, iy, iz]] = row_x[ix];
                }
            }
        }

        Array3::from_shape_fn(
            (self.grid.nx, self.grid.ny, self.grid.nz),
            |(kx, ky, kz)| {
                let kx_idx = fft_signed_index(kx, self.grid.nx).rem_euclid(self.mx as i64) as usize;
                let ky_idx = fft_signed_index(ky, self.grid.ny).rem_euclid(self.my as i64) as usize;
                let kz_idx = fft_signed_index(kz, self.grid.nz).rem_euclid(self.mz as i64) as usize;
                grid[[kx_idx, ky_idx, kz_idx]]
                    * (self.deconv_x[kx] * self.deconv_y[ky] * self.deconv_z[kz])
            },
        )
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

/// Exact direct 3D type-1 NUFFT.
#[must_use]
pub fn nufft_type1_3d(
    positions: &[(f64, f64, f64)],
    values: &[Complex64],
    grid: UniformGrid3D,
) -> Array3<Complex64> {
    assert_eq!(
        positions.len(),
        values.len(),
        "positions/value length mismatch"
    );
    let (lx, ly, lz) = grid.lengths();
    let two_pi_lx = 2.0 * PI / lx;
    let two_pi_ly = 2.0 * PI / ly;
    let two_pi_lz = 2.0 * PI / lz;

    Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(px, py, pz)| {
        let kx = fft_signed_index(px, grid.nx);
        let ky = fft_signed_index(py, grid.ny);
        let kz = fft_signed_index(pz, grid.nz);
        positions.iter().zip(values.iter()).fold(
            Complex64::new(0.0, 0.0),
            |acc, (&(x, y, z), &value)| {
                let angle = -(two_pi_lx * kx as f64 * x
                    + two_pi_ly * ky as f64 * y
                    + two_pi_lz * kz as f64 * z);
                acc + value * Complex64::new(angle.cos(), angle.sin())
            },
        )
    })
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

/// Fast 3D type-1 NUFFT convenience wrapper.
#[must_use]
pub fn nufft_type1_3d_fast(
    positions: &[(f64, f64, f64)],
    values: &[Complex64],
    grid: UniformGrid3D,
    kernel_width: usize,
) -> Array3<Complex64> {
    NufftPlan3D::new(grid, DEFAULT_NUFFT_OVERSAMPLING, kernel_width).type1(positions, values)
}

fn axis_deconv(n: usize, m: usize, kernel_width: usize, beta: f64, i0_beta: f64) -> Array1<f64> {
    Array1::from_shape_fn(n, |k| {
        let xi = fft_signed_index(k, n) as f64 / m as f64;
        1.0 / kb_kernel_ft(xi, kernel_width, beta, i0_beta)
    })
}

fn fft_signed_index(index: usize, len: usize) -> i64 {
    if index <= len / 2 {
        index as i64
    } else {
        index as i64 - len as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert!(
            max_err / scale < 1e-6,
            "max relative error = {}",
            max_err / scale
        );
    }

    #[test]
    fn fast_3d_tracks_exact() {
        let grid = UniformGrid3D::new(8, 8, 8, 0.1, 0.1, 0.1).unwrap();
        let positions: Vec<(f64, f64, f64)> = (0..10)
            .map(|i| {
                let f = i as f64;
                (
                    (f * 0.17).rem_euclid(grid.nx as f64 * grid.dx),
                    (f * 0.11).rem_euclid(grid.ny as f64 * grid.dy),
                    (f * 0.07).rem_euclid(grid.nz as f64 * grid.dz),
                )
            })
            .collect();
        let values: Vec<Complex64> = (0..10)
            .map(|i| Complex64::new((i as f64 * 0.4).cos(), (i as f64 * 0.3).sin()))
            .collect();
        let exact = nufft_type1_3d(&positions, &values, grid);
        let fast = nufft_type1_3d_fast(&positions, &values, grid, DEFAULT_NUFFT_KERNEL_WIDTH);
        let scale = exact.iter().map(|value| value.norm()).fold(1.0, f64::max);
        let max_err = exact
            .iter()
            .zip(fast.iter())
            .map(|(lhs, rhs)| (lhs - rhs).norm())
            .fold(0.0, f64::max);
        assert!(
            max_err / scale < 1e-6,
            "max relative error = {}",
            max_err / scale
        );
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
                .map(|i| ((i as f64 * std::f64::consts::SQRT_2 + 0.13).fract() * domain.length()))
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
