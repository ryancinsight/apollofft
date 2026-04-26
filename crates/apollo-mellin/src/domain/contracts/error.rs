//! Error contracts for Mellin plans.

use thiserror::Error;

/// Result alias for Mellin operations.
pub type MellinResult<T> = Result<T, MellinError>;

/// Errors produced by Mellin plan construction or execution.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum MellinError {
    /// Sample count is zero.
    #[error("sample count must be > 0")]
    EmptySampleCount,
    /// Scale bounds must be finite and positive.
    #[error("scale bounds must be finite and > 0")]
    InvalidScaleBound,
    /// Minimum scale must be less than maximum scale.
    #[error("min scale must be < max scale")]
    InvalidScaleOrder,
    /// Output buffer length must match.
    #[error("output length mismatch")]
    LengthMismatch,
    /// Input signal must be non-empty.
    #[error("input signal must be non-empty")]
    EmptySignal,
    /// Signal domain bounds must be finite and positive.
    #[error("signal domain bounds must be finite and > 0")]
    InvalidSignalBound,
    /// Signal domain minimum must be less than maximum.
    #[error("signal domain min must be < max")]
    InvalidSignalOrder,
    /// Mellin exponent must be finite.
    #[error("Mellin exponent must be finite")]
    InvalidExponent,
    /// Precision profile does not match the requested storage type.
    #[error("precision profile does not match storage type")]
    PrecisionMismatch,
}
