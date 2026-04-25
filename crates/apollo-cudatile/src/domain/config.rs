//! cudatile configuration types.

use serde::{Deserialize, Serialize};

/// Device and launch configuration for future cudatile integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudatileDeviceConfig {
    /// Logical CUDA device identifier.
    pub device_id: u32,
    /// Preferred tile size for future kernel launches.
    pub tile_width: u32,
    /// Preferred tile size for future kernel launches.
    pub tile_height: u32,
}

impl Default for CudatileDeviceConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            tile_width: 16,
            tile_height: 16,
        }
    }
}
