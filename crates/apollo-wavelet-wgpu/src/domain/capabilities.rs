//! WGPU capability contracts for the Haar DWT backend.

/// Truthful WGPU transform capability descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuCapabilities {
    /// Whether a WGPU adapter and device can be acquired.
    pub device_available: bool,
    /// Whether forward DWT execution is implemented.
    pub supports_forward: bool,
    /// Whether inverse DWT execution is implemented.
    pub supports_inverse: bool,
}

impl WgpuCapabilities {
    /// Construct capabilities reflecting zero-kernel state.
    #[must_use]
    pub const fn detected(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: false,
            supports_inverse: false,
        }
    }

    /// Construct capabilities for a fully implemented forward+inverse kernel.
    ///
    /// Both `supports_forward` and `supports_inverse` are true iff
    /// `device_available` is true.
    #[must_use]
    pub const fn implemented(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: device_available,
            supports_inverse: device_available,
        }
    }
}
