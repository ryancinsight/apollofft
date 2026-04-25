//! NTT contracts and capabilities.

use thiserror::Error;

/// Errors produced by NTT plan creation or execution.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NttError {
    /// Length is zero.
    #[error("length must be > 0")]
    EmptyLength,
    /// Length is not a power of two.
    #[error("length must be a power of two")]
    NonPowerOfTwo,
    /// Input length does not match the plan.
    #[error("input length mismatch")]
    LengthMismatch,
    /// Modulus is less than 2.
    #[error("modulus must be at least 2")]
    InvalidModulus,
    /// Selected modulus does not support the requested transform length.
    #[error("transform length is not supported by the modulus")]
    UnsupportedLength,
}
