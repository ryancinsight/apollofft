//! CZT contracts and capabilities.

use thiserror::Error;

/// Errors produced by CZT plan creation or execution.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum CztError {
    /// The requested length is zero.
    #[error("length must be > 0")]
    EmptyLength,
    /// The plan parameters are not valid for the requested transform.
    #[error("invalid transform parameters")]
    InvalidParameters,
    /// Input length mismatch.
    #[error("input length mismatch")]
    LengthMismatch,
    /// Requested precision profile does not match the selected storage type.
    #[error("precision profile does not match the selected storage type")]
    PrecisionMismatch,
}
