//! Error contracts for spherical harmonic transform plans.

use thiserror::Error;

/// Result alias for SHT operations.
pub type ShtResult<T> = Result<T, ShtError>;

/// Errors produced by SHT plan construction.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ShtError {
    /// Latitude count is zero.
    #[error("latitude count must be > 0")]
    EmptyLatitudeCount,
    /// Longitude count is zero.
    #[error("longitude count must be > 0")]
    EmptyLongitudeCount,
    /// Harmonic degree cannot be represented by the sampling grid.
    #[error("max degree exceeds spherical grid sampling support")]
    DegreeExceedsSampling,
    /// Sample grid shape does not match the plan.
    #[error("sample grid shape does not match the plan")]
    SampleShapeMismatch,
    /// Coefficient storage does not match the plan.
    #[error("coefficient shape does not match the plan")]
    CoefficientShapeMismatch,
}
