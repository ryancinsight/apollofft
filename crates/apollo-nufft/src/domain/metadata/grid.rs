//! Uniform periodic domain descriptors for NUFFT plans.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// NUFFT metadata validation error.
#[derive(Debug, Error, Clone, PartialEq)]
#[error("invalid NUFFT metadata field {field}: {value} ({reason})")]
pub struct NufftMetadataError {
    field: &'static str,
    value: String,
    reason: &'static str,
}

impl NufftMetadataError {
    fn validation(field: &'static str, value: impl ToString, reason: &'static str) -> Self {
        Self {
            field,
            value: value.to_string(),
            reason,
        }
    }
}

/// Uniform 1D periodic domain descriptor used by NUFFT surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UniformDomain1D {
    /// Number of uniform samples or Fourier bins.
    pub n: usize,
    /// Spatial step size.
    pub dx: f64,
}

impl UniformDomain1D {
    /// Create a validated 1D uniform domain descriptor.
    pub fn new(n: usize, dx: f64) -> Result<Self, NufftMetadataError> {
        if n == 0 {
            return Err(NufftMetadataError::validation("n", n, "must be > 0"));
        }
        if !dx.is_finite() || dx <= 0.0 {
            return Err(NufftMetadataError::validation(
                "dx",
                dx,
                "must be finite and > 0",
            ));
        }
        Ok(Self { n, dx })
    }

    /// Return the physical periodic domain length.
    #[must_use]
    pub fn length(self) -> f64 {
        self.n as f64 * self.dx
    }
}

/// Uniform 3D periodic grid descriptor used by NUFFT surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UniformGrid3D {
    /// X dimension.
    pub nx: usize,
    /// Y dimension.
    pub ny: usize,
    /// Z dimension.
    pub nz: usize,
    /// X spacing.
    pub dx: f64,
    /// Y spacing.
    pub dy: f64,
    /// Z spacing.
    pub dz: f64,
}

impl UniformGrid3D {
    /// Create a validated 3D uniform grid descriptor.
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> Result<Self, NufftMetadataError> {
        if nx == 0 {
            return Err(NufftMetadataError::validation("nx", nx, "must be > 0"));
        }
        if ny == 0 {
            return Err(NufftMetadataError::validation("ny", ny, "must be > 0"));
        }
        if nz == 0 {
            return Err(NufftMetadataError::validation("nz", nz, "must be > 0"));
        }
        for (field, value) in [("dx", dx), ("dy", dy), ("dz", dz)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(NufftMetadataError::validation(
                    field,
                    value,
                    "must be finite and > 0",
                ));
            }
        }
        Ok(Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
        })
    }

    /// Return the physical domain lengths.
    #[must_use]
    pub fn lengths(self) -> (f64, f64, f64) {
        (
            self.nx as f64 * self.dx,
            self.ny as f64 * self.dy,
            self.nz as f64 * self.dz,
        )
    }
}
