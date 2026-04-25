//! Error contracts for graph Fourier transforms.

use thiserror::Error;

/// Result alias for GFT operations.
pub type GftResult<T> = Result<T, GftError>;

/// Errors produced by graph Fourier plan creation or execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum GftError {
    /// Graph has no vertices.
    #[error("graph has no vertices")]
    EmptyGraph,
    /// Adjacency matrix is not square.
    #[error("adjacency matrix is not square")]
    NonSquareAdjacency,
    /// Adjacency matrix is not symmetric.
    #[error("adjacency matrix is not symmetric")]
    NonSymmetricAdjacency,
    /// Adjacency matrix contains a non-finite edge weight.
    #[error("adjacency matrix contains a non-finite edge weight")]
    NonFiniteWeight,
    /// Input length does not match the graph order.
    #[error("input length does not match graph order")]
    LengthMismatch,
}
