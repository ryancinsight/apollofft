//! WGPU capability contracts.

use apollo_fft::PrecisionProfile;

/// Truthful WGPU transform capability descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuCapabilities {
    /// Whether a WGPU adapter and device can be acquired.
    pub device_available: bool,
    /// Whether forward transform execution is implemented.
    pub supports_forward: bool,
    /// Whether inverse or adjoint transform execution is implemented.
    pub supports_inverse: bool,
    /// Whether mixed-precision (f16/f32/f64) typed storage dispatch is supported.
    pub supports_mixed_precision: bool,
    /// Whether exact narrowed integer residue storage dispatch is supported.
    pub supports_quantized_storage: bool,
    /// Default precision profile for GPU execution.
    pub default_precision_profile: PrecisionProfile,
}

impl WgpuCapabilities {
    /// Construct capabilities for a boundary-only backend.
    #[must_use]
    pub const fn detected(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: false,
            supports_inverse: false,
            supports_mixed_precision: false,
            supports_quantized_storage: false,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }

    /// Construct capabilities for full forward and inverse execution support.
    #[must_use]
    pub const fn full(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: device_available,
            supports_inverse: device_available,
            supports_mixed_precision: false,
            supports_quantized_storage: device_available,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }
}
