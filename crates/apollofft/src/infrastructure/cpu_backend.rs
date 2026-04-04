//! CPU backend adapter.

use crate::application::plan::{FftPlan1D, FftPlan2D, FftPlan3D};
use crate::backend::FftBackend;
use crate::domain::backend::BackendCapabilities;
use crate::domain::error::ApolloResult;
use crate::domain::types::{
    BackendKind, Normalization, PrecisionProfile, Shape1D, Shape2D, Shape3D,
};

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
            supports_mixed_precision: true,
            default_precision_profile: PrecisionProfile::HIGH_ACCURACY_F64,
            supported_precision_profiles: vec![
                PrecisionProfile::HIGH_ACCURACY_F64,
                PrecisionProfile::LOW_PRECISION_F32,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
            ],
        }
    }

    fn plan_1d(&self, shape: Shape1D) -> ApolloResult<Self::Plan1D> {
        Ok(FftPlan1D::with_precision(
            shape.n,
            PrecisionProfile::HIGH_ACCURACY_F64,
        ))
    }

    fn plan_2d(&self, shape: Shape2D) -> ApolloResult<Self::Plan2D> {
        Ok(FftPlan2D::with_precision(
            shape.nx,
            shape.ny,
            PrecisionProfile::HIGH_ACCURACY_F64,
        ))
    }

    fn plan_3d(&self, shape: Shape3D) -> ApolloResult<Self::Plan3D> {
        Ok(FftPlan3D::with_precision(
            shape.nx,
            shape.ny,
            shape.nz,
            PrecisionProfile::HIGH_ACCURACY_F64,
        ))
    }
}
