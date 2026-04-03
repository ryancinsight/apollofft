//! CPU backend adapter.

use crate::application::plan::{FftPlan1D, FftPlan2D, FftPlan3D};
use crate::backend::FftBackend;
use crate::domain::backend::BackendCapabilities;
use crate::domain::error::ApolloResult;
use crate::domain::types::{BackendKind, Normalization, Shape1D, Shape2D, Shape3D};

/// CPU backend backed by `rustfft` and `realfft`.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl FftBackend for CpuBackend {
    type Plan1D = FftPlan1D;
    type Plan2D = FftPlan2D;
    type Plan3D = FftPlan3D;

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            kind: BackendKind::Cpu,
            normalization: Normalization::FftwCompatible,
            supports_1d: true,
            supports_2d: true,
            supports_3d: true,
            supports_real_to_complex: true,
        }
    }

    fn plan_1d(&self, shape: Shape1D) -> ApolloResult<Self::Plan1D> {
        Ok(FftPlan1D::new(shape.n))
    }

    fn plan_2d(&self, shape: Shape2D) -> ApolloResult<Self::Plan2D> {
        Ok(FftPlan2D::new(shape.nx, shape.ny))
    }

    fn plan_3d(&self, shape: Shape3D) -> ApolloResult<Self::Plan3D> {
        Ok(FftPlan3D::new(shape.nx, shape.ny, shape.nz))
    }
}
