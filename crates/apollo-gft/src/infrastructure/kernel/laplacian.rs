//! Combinatorial Laplacian and graph spectral basis construction.

use crate::domain::graph::adjacency::GraphAdjacency;
use nalgebra::{DMatrix, SymmetricEigen};

/// Laplacian eigensystem stored for application-layer plans.
#[derive(Debug, Clone)]
pub struct GraphSpectralBasis {
    /// Laplacian eigenvalues.
    pub eigenvalues: Vec<f64>,
    /// Column-major eigenvector matrix.
    pub eigenvectors: Vec<f64>,
}

/// Build the combinatorial Laplacian `L = D - A`.
#[must_use]
pub fn combinatorial_laplacian(graph: &GraphAdjacency) -> DMatrix<f64> {
    let adjacency = graph.matrix();
    let n = graph.len();
    let mut laplacian = DMatrix::<f64>::zeros(n, n);
    for row in 0..n {
        let degree: f64 = (0..n).map(|col| adjacency[(row, col)]).sum();
        laplacian[(row, row)] = degree;
        for col in 0..n {
            laplacian[(row, col)] -= adjacency[(row, col)];
        }
    }
    laplacian
}

/// Compute the graph Fourier basis from the combinatorial Laplacian.
#[must_use]
pub fn spectral_basis(graph: &GraphAdjacency) -> GraphSpectralBasis {
    let decomposition = SymmetricEigen::new(combinatorial_laplacian(graph));
    let n = graph.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&lhs, &rhs| {
        decomposition.eigenvalues[lhs]
            .partial_cmp(&decomposition.eigenvalues[rhs])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = Vec::with_capacity(n * n);
    for &column in &order {
        eigenvalues.push(decomposition.eigenvalues[column]);
        for row in 0..n {
            eigenvectors.push(decomposition.eigenvectors[(row, column)]);
        }
    }

    GraphSpectralBasis {
        eigenvalues,
        eigenvectors,
    }
}
