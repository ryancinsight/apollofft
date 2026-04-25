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
    /// Numerical execution is unsupported for the requested operation.
    #[error("{operation} is unsupported by the current WGPU capability set")]
    UnsupportedExecution {
        /// Requested operation name.
        operation: &'static str,
    },
    /// Plan carries an invalid length.
    #[error("invalid plan length {len}: {message}")]
    InvalidPlan {
        /// Invalid length value.
        len: usize,
        /// Failure description.
        message: &'static str,
    },
    /// Signal or spectrum slice length does not match the plan length.
    #[error("length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Length required by the plan.
        expected: usize,
        /// Length supplied by the caller.
        actual: usize,
    },
    /// Basis slice length does not equal len*len.
    #[error("basis length mismatch: expected {expected}, got {actual}")]
    BasisLengthMismatch {
        /// Expected basis length (len*len).
        expected: usize,
        /// Actual basis slice length.
        actual: usize,
    },
    /// GPU buffer map operation failed.
    #[error("buffer map failed: {message}")]
    BufferMapFailed {
        /// Failure context.
        message: String,
    },
}
