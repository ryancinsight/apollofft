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
    /// Plan parameters are invalid.
    #[error(
        "invalid SDFT WGPU plan: window_len={window_len}, bin_count={bin_count}, reason={message}"
    )]
    InvalidPlan {
        /// Window length.
        window_len: usize,
        /// Bin count.
        bin_count: usize,
        /// Failure explanation.
        message: &'static str,
    },
    /// Window input length does not match the plan.
    #[error("window length mismatch: expected {expected}, got {actual}")]
    WindowLengthMismatch {
        /// Expected window length.
        expected: usize,
        /// Actual window length.
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
