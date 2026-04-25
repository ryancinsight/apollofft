//! Error contracts for sliding DFT plans.

use thiserror::Error;

/// Result alias for SDFT operations.
pub type SdftResult<T> = Result<T, SdftError>;

/// Errors produced by SDFT plan construction.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum SdftError {
    /// Window length is zero.
    #[error("window length must be > 0")]
    EmptyWindow,
    /// Bin count is zero.
    #[error("bin count must be > 0")]
    EmptyBinCount,
    /// Bin count exceeds window length.
    #[error("bin count must be <= window length")]
    BinCountExceedsWindow,
    /// Initial window length does not match the plan.
    #[error("initial window length does not match the plan")]
    InitialWindowLengthMismatch,
}
