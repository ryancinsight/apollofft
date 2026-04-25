//! Validated parallel-beam Radon geometry.

use crate::domain::contracts::error::{RadonError, RadonResult};
use serde::{Deserialize, Serialize};

/// Parallel-beam geometry for a 2D discrete image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParallelBeamGeometry {
    rows: usize,
    cols: usize,
    angles: Vec<f64>,
    detector_count: usize,
    detector_spacing: f64,
}

impl ParallelBeamGeometry {
    /// Create validated parallel-beam geometry.
    pub fn new(
        rows: usize,
        cols: usize,
        angles: Vec<f64>,
        detector_count: usize,
        detector_spacing: f64,
    ) -> RadonResult<Self> {
        if rows == 0 {
            return Err(RadonError::EmptyRows);
        }
        if cols == 0 {
            return Err(RadonError::EmptyCols);
        }
        if angles.is_empty() {
            return Err(RadonError::EmptyAngles);
        }
        if angles.iter().any(|angle| !angle.is_finite()) {
            return Err(RadonError::InvalidAngle);
        }
        if detector_count == 0 {
            return Err(RadonError::EmptyDetectors);
        }
        if !detector_spacing.is_finite() || detector_spacing <= 0.0 {
            return Err(RadonError::InvalidDetectorSpacing);
        }
        Ok(Self {
            rows,
            cols,
            angles,
            detector_count,
            detector_spacing,
        })
    }

    /// Return image row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Return image column count.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Borrow projection angles in radians.
    #[must_use]
    pub fn angles(&self) -> &[f64] {
        &self.angles
    }

    /// Return projection angle count.
    #[must_use]
    pub fn angle_count(&self) -> usize {
        self.angles.len()
    }

    /// Return detector bin count.
    #[must_use]
    pub const fn detector_count(&self) -> usize {
        self.detector_count
    }

    /// Return detector bin spacing.
    #[must_use]
    pub const fn detector_spacing(&self) -> f64 {
        self.detector_spacing
    }

    /// Convert a physical detector coordinate to fractional detector index.
    #[must_use]
    pub fn detector_index(&self, coordinate: f64) -> f64 {
        coordinate / self.detector_spacing + 0.5 * (self.detector_count as f64 - 1.0)
    }

    /// Return centered x-coordinate for a column index.
    #[must_use]
    pub fn x(&self, col: usize) -> f64 {
        col as f64 - 0.5 * (self.cols as f64 - 1.0)
    }

    /// Return centered y-coordinate for a row index.
    #[must_use]
    pub fn y(&self, row: usize) -> f64 {
        row as f64 - 0.5 * (self.rows as f64 - 1.0)
    }
}
