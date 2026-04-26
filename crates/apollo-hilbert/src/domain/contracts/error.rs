//! Error contracts for Hilbert transform operations.

use thiserror::Error;

/// Result alias for Hilbert operations.
pub type HilbertResult<T> = Result<T, HilbertError>;

/// Errors produced by Hilbert plan construction or execution.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum HilbertError {
    /// Signal length is zero.
    #[error("signal length must be > 0")]
    EmptySignal,
    /// Input length does not match the plan.
    #[error("input length does not match the plan")]
    LengthMismatch,
    /// Precision profile does not match the requested storage type.
    #[error("precision profile does not match storage type")]
    PrecisionMismatch,
}
