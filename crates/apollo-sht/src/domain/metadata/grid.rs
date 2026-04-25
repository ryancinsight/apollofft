//! Spherical surface sampling descriptors.

use crate::domain::contracts::error::{ShtError, ShtResult};
use serde::{Deserialize, Serialize};

/// Validated spherical grid and harmonic bandlimit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SphericalGridSpec {
    latitudes: usize,
    longitudes: usize,
    max_degree: usize,
}

impl SphericalGridSpec {
    /// Create a validated spherical grid specification.
    pub fn new(latitudes: usize, longitudes: usize, max_degree: usize) -> ShtResult<Self> {
        if latitudes == 0 {
            return Err(ShtError::EmptyLatitudeCount);
        }
        if longitudes == 0 {
            return Err(ShtError::EmptyLongitudeCount);
        }
        if max_degree >= latitudes || 2 * max_degree + 1 > longitudes {
            return Err(ShtError::DegreeExceedsSampling);
        }
        Ok(Self {
            latitudes,
            longitudes,
            max_degree,
        })
    }

    /// Return latitude sample count.
    #[must_use]
    pub const fn latitudes(self) -> usize {
        self.latitudes
    }

    /// Return longitude sample count.
    #[must_use]
    pub const fn longitudes(self) -> usize {
        self.longitudes
    }

    /// Return maximum spherical harmonic degree.
    #[must_use]
    pub const fn max_degree(self) -> usize {
        self.max_degree
    }
}
