//! Backend traits and capability descriptors.

use crate::domain::error::ApolloResult;
use crate::domain::types::{
    BackendKind, Normalization, PrecisionProfile, Shape1D, Shape2D, Shape3D,
};

/// Capability descriptor advertised by a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapabilities {
    /// Backend family.
    pub kind: BackendKind,
    /// Normalization convention implemented by the backend.
    pub normalization: Normalization,
    /// Whether the backend can plan 1D transforms.
    pub supports_1d: bool,
    /// Whether the backend can plan 2D transforms.
    pub supports_2d: bool,
    /// Whether the backend can plan 3D transforms.
    pub supports_3d: bool,
    /// Whether the backend supports real-to-complex half-spectrum transforms.
    pub supports_real_to_complex: bool,
    /// Whether the backend implements at least one mixed-precision profile.
    pub supports_mixed_precision: bool,
    /// Default precision profile used by compatibility constructors.
    pub default_precision_profile: PrecisionProfile,
    /// Precision profiles truthfully implemented by this backend.
    pub supported_precision_profiles: Vec<PrecisionProfile>,
}

/// Backend trait used by consumers that want backend selection via dependency inversion.
pub trait FftBackend {
    /// 1D plan type returned by the backend.
    type Plan1D;
    /// 2D plan type returned by the backend.
    type Plan2D;
    /// 3D plan type returned by the backend.
    type Plan3D;

    /// Identify the backend family.
    fn backend_kind(&self) -> BackendKind;

    /// Report backend capabilities.
    fn capabilities(&self) -> BackendCapabilities;

    /// Construct a 1D plan.
    fn plan_1d(&self, shape: Shape1D) -> ApolloResult<Self::Plan1D>;

    /// Construct a 2D plan.
    fn plan_2d(&self, shape: Shape2D) -> ApolloResult<Self::Plan2D>;

    /// Construct a 3D plan.
    fn plan_3d(&self, shape: Shape3D) -> ApolloResult<Self::Plan3D>;
}
