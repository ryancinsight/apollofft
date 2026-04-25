//! WGPU error contracts.

use thiserror::Error;

/// Result alias for WGPU operations.
pub type WgpuResult<T> = Result<T, WgpuError>;

/// Errors produced by WGPU backend operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum WgpuError {
    /// Adapter acquisition failed.
    #[error("wgpu adapter unavailable: {message}")]
    AdapterUnavailable {
        /// Adapter failure context.
        message: String,
    },
    /// Device acquisition failed.
    #[error("wgpu device unavailable: {message}")]
    DeviceUnavailable {
        /// Device failure context.
        message: String,
    },
    /// Plan length is empty.
    #[error("invalid DHT WGPU plan length {len}: {message}")]
    InvalidLength {
        /// Requested logical length.
        len: usize,
        /// Failure explanation.
        message: &'static str,
    },
    /// Input length does not match the plan.
    #[error("input length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Plan length.
        expected: usize,
        /// Input length.
        actual: usize,
    },
    /// Host readback from the staging buffer failed.
    #[error("wgpu buffer map failed: {message}")]
    BufferMapFailed {
        /// Mapping failure context.
        message: String,
    },
    /// Numerical execution is unsupported for the requested operation.
    #[error("{operation} is unsupported by the current WGPU capability set")]
    UnsupportedExecution {
        /// Requested operation name.
        operation: &'static str,
    },
}
