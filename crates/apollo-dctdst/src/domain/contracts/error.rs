//! Error contracts for DCT/DST plans.

use thiserror::Error;

/// Result alias for DCT/DST operations.
pub type DctDstResult<T> = Result<T, DctDstError>;

/// Errors produced by DCT/DST plan construction.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum DctDstError {
    /// Transform length is zero.
    #[error("transform length must be > 0")]
    EmptyLength,
    /// Input length does not match the plan.
    #[error("input length does not match the plan")]
    LengthMismatch,
}
