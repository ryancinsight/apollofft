//! Application-layer execution and orchestration modules.

/// Executable kernels and transform plans.
pub mod execution;
/// Plan-cache orchestration.
pub mod orchestration;

/// Compatibility module for FFT plans.
pub mod plan {
    pub use crate::application::execution::plan::fft::{
        dimension_1d::FftPlan1D, dimension_2d::FftPlan2D, dimension_3d::FftPlan3D,
        real_storage::RealFftData,
    };
}

/// Compatibility module for reusable FFT plan caches.
pub mod cache {
    pub use crate::application::orchestration::cache::plans::*;
}

/// FFT utility functions (fftfreq, rfftfreq, fftshift, ifftshift).
pub mod utilities;
