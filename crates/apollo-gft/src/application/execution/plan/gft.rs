//! Reusable graph Fourier transform plan.
//!
//! For an undirected weighted graph with combinatorial Laplacian `L = D - A`,
//! the graph Fourier basis is the orthonormal eigenvector matrix `U` of `L`.
//! The forward transform is `U^T x`; the inverse transform is `U X`.

use crate::domain::contracts::error::{GftError, GftResult};
use crate::domain::graph::adjacency::GraphAdjacency;
use crate::infrastructure::kernel::laplacian::spectral_basis;
use nalgebra::DMatrix;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Reusable graph Fourier plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GftPlan {
    n: usize,
    eigenvalues: Vec<f64>,
    basis: Vec<f64>,
}

impl GftPlan {
    /// Create a graph Fourier plan from an undirected weighted adjacency matrix.
    pub fn from_adjacency(adjacency: &DMatrix<f64>) -> GftResult<Self> {
        let graph = GraphAdjacency::new(adjacency.clone())?;
        Self::from_graph(&graph)
    }

    /// Create a graph Fourier plan from a validated graph adjacency descriptor.
    pub fn from_graph(graph: &GraphAdjacency) -> GftResult<Self> {
        let basis = spectral_basis(graph);
        Ok(Self {
            n: graph.len(),
            eigenvalues: basis.eigenvalues,
            basis: basis.eigenvectors,
        })
    }

    /// Return graph order.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.n
    }

    /// Return true when graph order is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Return Laplacian eigenvalues.
    #[must_use]
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Return the column-major graph Fourier basis.
    #[must_use]
    pub fn basis(&self) -> &[f64] {
        &self.basis
    }

    /// Forward graph Fourier transform `U^T x`.
    pub fn forward(&self, signal: &Array1<f64>) -> GftResult<Array1<f64>> {
        if signal.len() != self.n {
            return Err(GftError::LengthMismatch);
        }
        Ok(Array1::from_shape_fn(self.n, |k| {
            (0..self.n)
                .map(|i| self.basis[i + k * self.n] * signal[i])
                .sum()
        }))
    }

    /// Inverse graph Fourier transform `U X`.
    pub fn inverse(&self, spectrum: &Array1<f64>) -> GftResult<Array1<f64>> {
        if spectrum.len() != self.n {
            return Err(GftError::LengthMismatch);
        }
        Ok(Array1::from_shape_fn(self.n, |i| {
            (0..self.n)
                .map(|k| self.basis[i + k * self.n] * spectrum[k])
                .sum()
        }))
    }
}
