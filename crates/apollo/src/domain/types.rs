//! Shared shape, normalization, and backend-identification types.

use crate::domain::error::{ApolloError, ApolloResult};
use serde::{Deserialize, Serialize};

/// Supported backend families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    /// CPU backend backed by `rustfft` and `realfft`.
    Cpu,
    /// WGPU backend.
    Wgpu,
    /// cudatile adapter backend.
    Cudatile,
}

/// Apollo's normalization convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Normalization {
    /// Forward unnormalized, inverse normalized by the transformed volume.
    FftwCompatible,
}

/// Storage precision used by an execution profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StoragePrecision {
    /// 16-bit floating point storage.
    F16,
    /// 32-bit floating point storage.
    F32,
    /// 64-bit floating point storage.
    F64,
}

/// Arithmetic precision used internally by an execution profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputePrecision {
    /// Mixed `f16` storage with `f32` compute.
    MixedF16F32,
    /// 32-bit floating point compute.
    F32,
    /// 64-bit floating point compute.
    F64,
}

/// High-level precision intent selected by a caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionMode {
    /// Highest-accuracy backend path.
    HighAccuracy,
    /// Explicit mixed-precision execution.
    MixedPrecision,
    /// Explicit low-precision execution.
    LowPrecision,
}

/// Precision contract advertised by a backend or plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrecisionProfile {
    /// Precision mode intent.
    pub mode: PrecisionMode,
    /// Input/output storage precision.
    pub storage: StoragePrecision,
    /// Internal arithmetic precision.
    pub compute: ComputePrecision,
}

impl PrecisionProfile {
    /// CPU-style high-accuracy profile.
    pub const HIGH_ACCURACY_F64: Self = Self {
        mode: PrecisionMode::HighAccuracy,
        storage: StoragePrecision::F64,
        compute: ComputePrecision::F64,
    };

    /// Explicit low-precision f32 profile.
    pub const LOW_PRECISION_F32: Self = Self {
        mode: PrecisionMode::LowPrecision,
        storage: StoragePrecision::F32,
        compute: ComputePrecision::F32,
    };

    /// Mixed-precision profile with `f16` storage and `f32` compute.
    pub const MIXED_PRECISION_F16_F32: Self = Self {
        mode: PrecisionMode::MixedPrecision,
        storage: StoragePrecision::F16,
        compute: ComputePrecision::MixedF16F32,
    };
}

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
    pub fn new(n: usize, dx: f64) -> ApolloResult<Self> {
        if n == 0 {
            return Err(ApolloError::validation("n", n.to_string(), "must be > 0"));
        }
        if !dx.is_finite() || dx <= 0.0 {
            return Err(ApolloError::validation(
                "dx",
                dx.to_string(),
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
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> ApolloResult<Self> {
        Shape3D::new(nx, ny, nz)?;
        for (field, value) in [("dx", dx), ("dy", dy), ("dz", dz)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ApolloError::validation(
                    field,
                    value.to_string(),
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

    /// Return the underlying shape descriptor.
    #[must_use]
    pub fn shape(self) -> Shape3D {
        Shape3D {
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
        }
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
