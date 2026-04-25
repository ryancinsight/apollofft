//! WGPU capability contracts for the STFT backend.

/// Truthful WGPU transform capability descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuCapabilities {
    /// Whether a WGPU adapter and device can be acquired.
    pub device_available: bool,
    /// Whether forward STFT execution is implemented.
    pub supports_forward: bool,
    /// Whether inverse STFT execution is implemented.
    pub supports_inverse: bool,
}

impl WgpuCapabilities {
    /// Construct capabilities reflecting zero-kernel state (both false).
    #[must_use]
    pub const fn detected(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: false,
            supports_inverse: false,
        }
    }

    /// Construct capabilities for a forward-only implementation.
    ///
    /// `supports_forward` is true iff `device_available` is true.
    /// `supports_inverse` is always false (inverse STFT not implemented on GPU).
    #[must_use]
    pub const fn forward_only(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: device_available,
            supports_inverse: false,
        }
    }
}
