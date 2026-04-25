//! FrFT error contracts.

use thiserror::Error;

/// Errors produced by FrFT plan creation or execution.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum FrftError {
    /// Transform length is zero.
    #[error("signal length must be > 0")]
    EmptySignal,
    /// Fractional order is not finite.
    #[error("FrFT order must be finite")]
    NonFiniteOrder,
    /// Input length does not match the plan.
    #[error("input length {input} does not match plan length {plan}")]
    LengthMismatch {
        /// Observed input or output length.
        input: usize,
        /// Required plan length.
        plan: usize,
    },
}
