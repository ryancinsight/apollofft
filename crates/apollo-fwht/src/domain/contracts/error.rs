//! FWHT contracts and capabilities.

use thiserror::Error;

/// Errors returned by FWHT execution.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum FwhtError {
    /// Input length is zero.
    #[error("length must be > 0")]
    EmptyInput,
    /// Input length is not a power of two.
    #[error("length must be a power of two")]
    NonPowerOfTwo,
    /// Input length does not match the plan.
    #[error("input length does not match the plan")]
    LengthMismatch,
    /// Requested precision profile does not match the selected storage type.
    #[error("precision profile does not match the selected storage type")]
    PrecisionMismatch,
}
