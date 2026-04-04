//! 3D Type-1 NUFFT executions and plans.

use crate::nufft::math::{axis_deconv, bucket_count, fft_signed_index, kb_kernel};
use crate::nufft::DEFAULT_NUFFT_OVERSAMPLING;
use crate::types::UniformGrid3D;
use ndarray::Array3;
use num_complex::Complex64;
use rustfft::FftPlanner;
use std::cmp::Ordering;
use std::f64::consts::PI;
use std::sync::Arc;

#[derive(Clone, Copy)]
struct IndexedPoint3D {
    x: f64,
    y: f64,
    z: f64,
    value: Complex64,
    bucket: usize,
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

/// Reusable 3D type-1 NUFFT plan using separable Kaiser-Bessel spreading.
pub struct NufftPlan3D {
    grid: UniformGrid3D,
    mx: usize,
    my: usize,
    mz: usize,
    w: usize,
    beta: f64,
    i0_beta: f64,
    deconv_x: ndarray::Array1<f64>,
    deconv_y: ndarray::Array1<f64>,
    deconv_z: ndarray::Array1<f64>,
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
        let i0_beta = crate::nufft::math::i0(beta);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nufft::DEFAULT_NUFFT_KERNEL_WIDTH;

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
        assert!(max_err / scale < 1e-6);
    }
}
