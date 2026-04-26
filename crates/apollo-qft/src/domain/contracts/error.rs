//! Error contracts for quantum Fourier transforms.

use thiserror::Error;

/// Result alias for QFT operations.
pub type QftResult<T> = Result<T, QftError>;

/// Errors produced by QFT execution.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum QftError {
    /// Input length is zero.
    #[error("length must be > 0")]
    EmptyLength,
    /// Input length does not match the plan.
    #[error("input length mismatch")]
    LengthMismatch,
    /// Precision profile does not match the requested storage type.
    #[error("precision profile does not match storage type")]
    PrecisionMismatch,
}
