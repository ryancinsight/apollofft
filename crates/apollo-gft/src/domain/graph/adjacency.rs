//! Validated undirected weighted graph adjacency matrices.
//!
//! A graph Fourier transform over an undirected weighted graph uses the
//! eigensystem of the combinatorial Laplacian `L = D - A`. This descriptor is
//! the single validation boundary for matrix shape, symmetry, and finite edge
//! weights.

use crate::domain::contracts::error::{GftError, GftResult};
use nalgebra::DMatrix;

const SYMMETRY_TOLERANCE: f64 = 1.0e-12;

/// Validated adjacency matrix for an undirected weighted graph.
#[derive(Debug, Clone)]
pub struct GraphAdjacency {
    matrix: DMatrix<f64>,
}

impl GraphAdjacency {
    /// Validate and store an undirected weighted adjacency matrix.
    pub fn new(matrix: DMatrix<f64>) -> GftResult<Self> {
        if matrix.nrows() == 0 {
            return Err(GftError::EmptyGraph);
        }
        if matrix.nrows() != matrix.ncols() {
            return Err(GftError::NonSquareAdjacency);
        }
        for row in 0..matrix.nrows() {
            for col in 0..matrix.ncols() {
                let value = matrix[(row, col)];
                if !value.is_finite() {
                    return Err(GftError::NonFiniteWeight);
                }
                if (value - matrix[(col, row)]).abs() > SYMMETRY_TOLERANCE {
                    return Err(GftError::NonSymmetricAdjacency);
                }
            }
        }
        Ok(Self { matrix })
    }

    /// Return the graph order.
    #[must_use]
    pub fn len(&self) -> usize {
        self.matrix.nrows()
    }

    /// Return true when the graph order is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the validated adjacency matrix.
    #[must_use]
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }
}
