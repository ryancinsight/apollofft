#![warn(missing_docs)]
//! cudatile backend contract surface for Apollo FFT.

pub mod application;
pub mod domain;
pub mod infrastructure;

use apollofft::backend::FftBackend;
use apollofft::domain::backend::BackendCapabilities;
use apollofft::error::{ApolloError, ApolloResult};
use apollofft::types::{BackendKind, Normalization, Shape1D, Shape2D, Shape3D};
pub use domain::config::CudatileDeviceConfig;

/// cudatile backend descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudatileBackend {
    config: CudatileDeviceConfig,
}

impl CudatileBackend {
    /// Create a new cudatile backend descriptor.
    #[must_use]
    pub fn new(config: CudatileDeviceConfig) -> Self {
        Self { config }
    }

    fn unavailable(&self) -> ApolloError {
        ApolloError::BackendUnavailable {
            backend: format!(
                "cudatile backend unavailable for device {} (enable runtime integration)",
                self.config.device_id
            ),
        }
    }
}

impl Default for CudatileBackend {
    fn default() -> Self {
        Self::new(CudatileDeviceConfig::default())
    }
}

impl FftBackend for CudatileBackend {
    type Plan1D = ();
    type Plan2D = ();
    type Plan3D = ();

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Cudatile
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            kind: BackendKind::Cudatile,
            normalization: Normalization::FftwCompatible,
            supports_1d: false,
            supports_2d: false,
            supports_3d: false,
            supports_real_to_complex: false,
        }
    }

    fn plan_1d(&self, _shape: Shape1D) -> ApolloResult<Self::Plan1D> {
        Err(self.unavailable())
    }

    fn plan_2d(&self, _shape: Shape2D) -> ApolloResult<Self::Plan2D> {
        Err(self.unavailable())
    }

    fn plan_3d(&self, _shape: Shape3D) -> ApolloResult<Self::Plan3D> {
        Err(self.unavailable())
    }
}

use serde::{Deserialize, Serialize};

