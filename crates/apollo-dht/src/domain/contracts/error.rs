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
}
