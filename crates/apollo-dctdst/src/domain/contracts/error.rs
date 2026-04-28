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
    /// Precision profile does not match the selected storage type.
    #[error("precision profile does not match the selected DCT/DST storage type")]
    PrecisionMismatch,
    /// Transform length does not satisfy the minimum required by the selected kind.
    ///
    /// DCT-I (Type-I discrete cosine transform) requires N ≥ 2 because the
    /// definition references both boundary samples x₀ and x_{N−1}; a length-1
    /// sequence leaves x_{N−1} = x₀ and the boundary-weighted sum degenerates.
    #[error(
        "transform length is too small for the selected transform kind (DCT-I requires N >= 2)"
    )]
    UnsupportedLength,
}
