//! Error contracts for Radon operations.

use thiserror::Error;

/// Result alias for Radon operations.
pub type RadonResult<T> = Result<T, RadonError>;

/// Errors produced by Radon plan construction or execution.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum RadonError {
    /// Image row count is zero.
    #[error("image row count must be > 0")]
    EmptyRows,
    /// Image column count is zero.
    #[error("image column count must be > 0")]
    EmptyCols,
    /// Angle list is empty.
    #[error("angle list must be non-empty")]
    EmptyAngles,
    /// Angle is non-finite.
    #[error("angles must be finite radians")]
    InvalidAngle,
    /// Detector count is zero.
    #[error("detector count must be > 0")]
    EmptyDetectors,
    /// Detector spacing is non-finite or non-positive.
    #[error("detector spacing must be finite and > 0")]
    InvalidDetectorSpacing,
    /// Image shape does not match the plan.
    #[error("image shape does not match the plan")]
    ImageShapeMismatch,
    /// Sinogram shape does not match the plan.
    #[error("sinogram shape does not match the plan")]
    SinogramShapeMismatch,
}
