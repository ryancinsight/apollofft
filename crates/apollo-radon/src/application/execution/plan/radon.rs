//! Reusable Radon transform plan.

use crate::domain::contracts::error::{RadonError, RadonResult};
use crate::domain::geometry::parallel_beam::ParallelBeamGeometry;
use crate::domain::projection::sinogram::Sinogram;
use crate::infrastructure::kernel::direct::{adjoint_backproject, forward_project};
use crate::infrastructure::kernel::filter::ramp_filter_projection_into;
use ndarray::{Array2, Axis};
use rayon::prelude::*;

/// Reusable 2D parallel-beam Radon plan.
#[derive(Debug, Clone, PartialEq)]
pub struct RadonPlan {
    geometry: ParallelBeamGeometry,
}

impl RadonPlan {
    /// Create a plan from validated geometry parameters.
    pub fn new(
        rows: usize,
        cols: usize,
        angles: Vec<f64>,
        detector_count: usize,
        detector_spacing: f64,
    ) -> RadonResult<Self> {
        Ok(Self {
            geometry: ParallelBeamGeometry::new(
                rows,
                cols,
                angles,
                detector_count,
                detector_spacing,
            )?,
        })
    }

    /// Create a plan from an existing geometry value.
    #[must_use]
    pub const fn from_geometry(geometry: ParallelBeamGeometry) -> Self {
        Self { geometry }
    }

    /// Borrow the plan geometry.
    #[must_use]
    pub const fn geometry(&self) -> &ParallelBeamGeometry {
        &self.geometry
    }

    /// Execute the forward Radon transform.
    pub fn forward(&self, image: &Array2<f64>) -> RadonResult<Sinogram> {
        if image.dim() != (self.geometry.rows(), self.geometry.cols()) {
            return Err(RadonError::ImageShapeMismatch);
        }
        Ok(Sinogram::new(forward_project(image, &self.geometry)))
    }

    /// Execute adjoint backprojection.
    pub fn backproject(&self, sinogram: &Sinogram) -> RadonResult<Array2<f64>> {
        if sinogram.shape() != (self.geometry.angle_count(), self.geometry.detector_count()) {
            return Err(RadonError::SinogramShapeMismatch);
        }
        Ok(adjoint_backproject(sinogram.values(), &self.geometry))
    }

    /// Execute ramp-filtered backprojection.
    ///
    /// ## Normalization
    ///
    /// The backprojection scale factor PI / angle_count approximates the continuous FBP integral
    /// (1/2pi) integral_0^pi d_theta discretized with uniform angular step PI / angle_count.
    ///
    /// **Limitation**: this normalization is only correct for uniformly spaced angles.
    /// Non-uniform angle distributions require weighted integration (e.g., Parker weighting).
    /// The Fourier slice theorem guarantees exact reconstruction in the limit of dense uniform
    /// angular sampling and ideal ramp filtering.
    pub fn filtered_backprojection(&self, sinogram: &Sinogram) -> RadonResult<Array2<f64>> {
        if sinogram.shape() != (self.geometry.angle_count(), self.geometry.detector_count()) {
            return Err(RadonError::SinogramShapeMismatch);
        }
        let sinogram_values = sinogram.values();
        let mut filtered = Array2::zeros(sinogram_values.dim());
        let det_spacing = self.geometry.detector_spacing();
        let det_count = self.geometry.detector_count();
        // Parallel ramp filter per angle: each row is independent.
        filtered
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(angle_idx, mut dest_row)| {
                let source_row = sinogram_values.row(angle_idx);
                let projection: Vec<f64> = source_row.iter().copied().collect();
                let mut filtered_buf = vec![0.0_f64; det_count];
                ramp_filter_projection_into(&projection, det_spacing, &mut filtered_buf);
                for (dst, src) in dest_row.iter_mut().zip(filtered_buf.iter()) {
                    *dst = *src;
                }
            });
        let mut image = adjoint_backproject(&filtered, &self.geometry);
        let scale = std::f64::consts::PI / self.geometry.angle_count() as f64;
        image.iter_mut().for_each(|value| *value *= scale);
        Ok(image)
    }
}
