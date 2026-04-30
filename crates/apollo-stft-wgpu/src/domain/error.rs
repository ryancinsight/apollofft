//! WGPU error contracts for the STFT backend.

use thiserror::Error;

/// Result alias for STFT WGPU operations.
pub type WgpuResult<T> = Result<T, WgpuError>;

/// Enumeration of all failure modes for the STFT WGPU backend.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum WgpuError {
    /// Plan parameters are invalid (zero lengths or hop exceeds frame).
    #[error("invalid STFT plan: frame_len={frame_len}, hop_len={hop_len}: {message}")]
    InvalidPlan {
        /// Frame length supplied to the plan.
        frame_len: usize,
        /// Hop length supplied to the plan.
        hop_len: usize,
        /// Human-readable description of the violated constraint.
        message: &'static str,
    },
    /// Input signal is shorter than the minimum required length.
    #[error("input too short: required {min} samples, actual {actual}")]
    InputTooShort {
        /// Minimum required signal length.
        min: usize,
        /// Actual signal length supplied.
        actual: usize,
    },
    /// Buffer length does not match the plan expectation.
    #[error("length mismatch: expected {expected}, actual {actual}")]
    LengthMismatch {
        /// Expected length derived from the plan.
        expected: usize,
        /// Actual length supplied by the caller.
        actual: usize,
    },
    /// GPU buffer mapping failed.
    #[error("buffer map failed: {message}")]
    BufferMapFailed {
        /// Error detail from the WGPU runtime.
        message: String,
    },
    /// No WGPU adapter could be acquired.
    #[error("wgpu adapter unavailable: {message}")]
    AdapterUnavailable {
        /// Error detail from the WGPU runtime.
        message: String,
    },
    /// No WGPU device could be acquired.
    #[error("wgpu device unavailable: {message}")]
    DeviceUnavailable {
        /// Error detail from the WGPU runtime.
        message: String,
    },
    /// The requested operation is not implemented.
    #[error("{operation} is unsupported by the current WGPU capability set")]
    UnsupportedExecution {
        /// Name of the unsupported operation.
        operation: &'static str,
    },
    /// Requested precision profile does not match the typed storage.
    #[error("precision profile does not match typed STFT WGPU storage")]
    InvalidPrecisionProfile,
    /// The inverse FFT path requires `frame_len` to be a power of two.
    #[error("frame_len {frame_len} is not a power of two; the GPU inverse FFT path requires a power-of-two frame length")]
    FrameLenNotPowerOfTwo {
        /// The non-power-of-two frame length supplied by the caller.
        frame_len: usize,
    },
}
