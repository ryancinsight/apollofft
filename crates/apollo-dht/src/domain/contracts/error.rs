//! Error contracts for DHT operations.

use thiserror::Error;

/// Result alias for DHT operations.
pub type DhtResult<T> = Result<T, DhtError>;

/// Errors produced by DHT plan construction or execution.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum DhtError {
    /// Signal length is zero.
    #[error("signal length must be > 0")]
    EmptySignal,
    /// Input length does not match the plan.
    #[error("input length does not match the plan")]
    LengthMismatch,
    /// Precision profile does not match the selected storage type.
    #[error("precision profile does not match the selected DHT storage type")]
    PrecisionMismatch,
    /// 2D input is not N\u{d7}N for the plan length N.
    #[error("2D DHT shape mismatch: expected {expected}\u{d7}{expected}, got {rows}\u{d7}{cols}")]
    ShapeMismatch2d {
        /// Expected side length.
        expected: usize,
        /// Actual row count.
        rows: usize,
        /// Actual column count.
        cols: usize,
    },
    /// 3D input is not N\u{d7}N\u{d7}N for the plan length N.
    #[error("3D DHT shape mismatch: expected {expected}^3, got {d0}\u{d7}{d1}\u{d7}{d2}")]
    ShapeMismatch3d {
        /// Expected side length.
        expected: usize,
        /// Actual dimension 0.
        d0: usize,
        /// Actual dimension 1.
        d1: usize,
        /// Actual dimension 2.
        d2: usize,
    },
}
