//! Shared shape and grid descriptor types.

use crate::domain::contracts::error::{ApolloError, ApolloResult};
use serde::{Deserialize, Serialize};

/// Shape descriptor for 1D plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape1D {
    /// Length of the signal.
    pub n: usize,
}

impl Shape1D {
    /// Create a validated 1D shape descriptor.
    pub fn new(n: usize) -> ApolloResult<Self> {
        if n == 0 {
            return Err(ApolloError::validation("n", n.to_string(), "must be > 0"));
        }
        Ok(Self { n })
    }
}

/// Shape descriptor for 2D plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape2D {
    /// X dimension.
    pub nx: usize,
    /// Y dimension.
    pub ny: usize,
}

impl Shape2D {
    /// Create a validated 2D shape descriptor.
    pub fn new(nx: usize, ny: usize) -> ApolloResult<Self> {
        if nx == 0 {
            return Err(ApolloError::validation("nx", nx.to_string(), "must be > 0"));
        }
        if ny == 0 {
            return Err(ApolloError::validation("ny", ny.to_string(), "must be > 0"));
        }
        Ok(Self { nx, ny })
    }
}

/// Shape descriptor for 3D plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape3D {
    /// X dimension.
    pub nx: usize,
    /// Y dimension.
    pub ny: usize,
    /// Z dimension.
    pub nz: usize,
}

impl Shape3D {
    /// Create a validated 3D shape descriptor.
    pub fn new(nx: usize, ny: usize, nz: usize) -> ApolloResult<Self> {
        if nx == 0 {
            return Err(ApolloError::validation("nx", nx.to_string(), "must be > 0"));
        }
        if ny == 0 {
            return Err(ApolloError::validation("ny", ny.to_string(), "must be > 0"));
        }
        if nz == 0 {
            return Err(ApolloError::validation("nz", nz.to_string(), "must be > 0"));
        }
        Ok(Self { nx, ny, nz })
    }

    /// Return the total number of points.
    #[must_use]
    pub fn volume(self) -> usize {
        self.nx * self.ny * self.nz
    }
}

/// Half-spectrum descriptor for R2C 3D transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HalfSpectrum3D {
    /// Full real-domain shape.
    pub full: Shape3D,
    /// Number of independent complex bins along Z.
    pub nz_complex: usize,
}

impl HalfSpectrum3D {
    /// Construct the half-spectrum descriptor implied by a real-domain shape.
    #[must_use]
    pub fn from_shape(full: Shape3D) -> Self {
        Self {
            full,
            nz_complex: full.nz / 2 + 1,
        }
    }
}
