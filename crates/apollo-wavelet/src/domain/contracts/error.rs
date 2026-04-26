//! Error contracts for wavelet transforms.

use thiserror::Error;

/// Result alias for wavelet operations.
pub type WaveletResult<T> = Result<T, WaveletError>;

/// Errors produced by wavelet plan construction or execution.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum WaveletError {
    /// Signal length is zero.
    #[error("signal length must be > 0")]
    EmptySignal,
    /// Signal length must be a power of two for this DWT plan.
    #[error("DWT signal length must be a power of two")]
    NonPowerOfTwoLength,
    /// Decomposition level count is zero.
    #[error("level count must be > 0")]
    EmptyLevelCount,
    /// Decomposition level exceeds the signal length.
    #[error("level count exceeds signal length")]
    LevelExceedsLength,
    /// Input length does not match the plan.
    #[error("input length does not match the plan")]
    LengthMismatch,
    /// CWT scale list is empty.
    #[error("CWT scale list must be non-empty")]
    EmptyScales,
    /// CWT scale is non-finite or non-positive.
    #[error("CWT scales must be finite and > 0")]
    InvalidScale,
    /// Coefficient storage does not match the plan.
    #[error("coefficient storage does not match the plan")]
    CoefficientShapeMismatch,
    /// Precision profile does not match the storage type.
    #[error("precision profile does not match wavelet storage type")]
    PrecisionMismatch,
}
