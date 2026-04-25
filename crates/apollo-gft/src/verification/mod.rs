//! Verification modules for graph Fourier transforms.

#[cfg(test)]
mod tests {
    use crate::{GftError, GftPlan, GraphAdjacency};
    use approx::assert_abs_diff_eq;
    use nalgebra::DMatrix;
    use ndarray::Array1;
    use proptest::prelude::*;

    #[test]
    fn two_vertex_graph_has_known_spectrum_and_roundtrips() {
        let adjacency = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let plan = GftPlan::from_adjacency(&adjacency).expect("plan");

        assert_eq!(plan.len(), 2);
        assert_abs_diff_eq!(plan.eigenvalues()[0], 0.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(plan.eigenvalues()[1], 2.0, epsilon = 1.0e-12);

        let input = Array1::from_vec(vec![2.0, -1.0]);
        let recovered = plan
            .inverse(&plan.forward(&input).expect("forward"))
            .expect("inverse");
        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn path_three_graph_roundtrips_and_has_zero_constant_mode() {
        let adjacency =
            DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let graph = GraphAdjacency::new(adjacency).expect("graph");
        let plan = GftPlan::from_graph(&graph).expect("plan");

        assert_abs_diff_eq!(plan.eigenvalues()[0], 0.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(plan.eigenvalues()[1], 1.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(plan.eigenvalues()[2], 3.0, epsilon = 1.0e-12);

        let constant = Array1::from_vec(vec![4.0, 4.0, 4.0]);
        let spectrum = plan.forward(&constant).expect("forward");
        assert_abs_diff_eq!(spectrum[1], 0.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(spectrum[2], 0.0, epsilon = 1.0e-12);

        let signal = Array1::from_vec(vec![1.0, -2.0, 0.5]);
        let recovered = plan
            .inverse(&plan.forward(&signal).expect("forward"))
            .expect("inverse");
        for (actual, expected) in recovered.iter().zip(signal.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn rejects_invalid_graphs_and_lengths() {
        let empty = DMatrix::<f64>::zeros(0, 0);
        assert_eq!(
            GftPlan::from_adjacency(&empty).unwrap_err(),
            GftError::EmptyGraph
        );

        let rectangular = DMatrix::from_row_slice(2, 3, &[0.0, 1.0, 2.0, 1.0, 0.0, 3.0]);
        assert_eq!(
            GftPlan::from_adjacency(&rectangular).unwrap_err(),
            GftError::NonSquareAdjacency
        );

        let asymmetric = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 2.0, 0.0]);
        assert_eq!(
            GftPlan::from_adjacency(&asymmetric).unwrap_err(),
            GftError::NonSymmetricAdjacency
        );

        let non_finite = DMatrix::from_row_slice(2, 2, &[0.0, f64::NAN, f64::NAN, 0.0]);
        assert_eq!(
            GftPlan::from_adjacency(&non_finite).unwrap_err(),
            GftError::NonFiniteWeight
        );

        let adjacency = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let plan = GftPlan::from_adjacency(&adjacency).expect("plan");
        assert_eq!(
            plan.forward(&Array1::from_vec(vec![1.0])).unwrap_err(),
            GftError::LengthMismatch
        );
        assert_eq!(
            plan.inverse(&Array1::from_vec(vec![1.0])).unwrap_err(),
            GftError::LengthMismatch
        );
    }
    proptest! {
        /// GFT roundtrip holds for randomly weighted symmetric adjacency matrices.
        #[test]
        fn gft_roundtrip_random_graph(
            n in 2usize..8usize,
            seed in 0u64..100u64,
        ) {
            let mut adj = DMatrix::<f64>::zeros(n, n);
            let mut rng_val = seed;
            for i in 0..n {
                for j in (i+1)..n {
                    rng_val = rng_val.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let w = rng_val as f64 / u64::MAX as f64;
                    adj[(i, j)] = w;
                    adj[(j, i)] = w;
                }
            }
            let plan = GftPlan::from_adjacency(&adj).unwrap();
            let signal = Array1::from_vec((0..n).map(|i| i as f64 + 1.0).collect::<Vec<_>>());
            let spectrum = plan.forward(&signal).unwrap();
            let recovered = plan.inverse(&spectrum).unwrap();
            let err: f64 = signal.iter().zip(recovered.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            prop_assert!(err < 1e-8, "GFT roundtrip err={}", err);
        }
    }

    #[test]
    fn eigenvector_basis_is_orthonormal() {
        // Path graph 0-1-2-3. Laplacian eigensystem is real-symmetric.
        // U^T U = I: inner product of distinct eigenvector columns = 0, self = 1.
        let n = 4usize;
        let mut adj = DMatrix::<f64>::zeros(n, n);
        adj[(0, 1)] = 1.0;
        adj[(1, 0)] = 1.0;
        adj[(1, 2)] = 1.0;
        adj[(2, 1)] = 1.0;
        adj[(2, 3)] = 1.0;
        adj[(3, 2)] = 1.0;
        let plan = GftPlan::from_adjacency(&adj).expect("plan");
        let basis = plan.basis();
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..n).map(|k| basis[k + i * n] * basis[k + j * n]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Eigenvectors {i} and {j} not orthonormal: dot = {dot}"
                );
            }
        }
    }

    #[test]
    fn weighted_graph_forward_inverse_roundtrip() {
        let n = 3usize;
        let mut adj = DMatrix::<f64>::zeros(n, n);
        adj[(0, 1)] = 2.5;
        adj[(1, 0)] = 2.5;
        adj[(1, 2)] = 0.7;
        adj[(2, 1)] = 0.7;
        adj[(0, 2)] = 1.3;
        adj[(2, 0)] = 1.3;
        let plan = GftPlan::from_adjacency(&adj).expect("plan");
        let signal = Array1::from_vec(vec![1.0, -2.0, 0.5]);
        let spectrum = plan.forward(&signal).expect("forward");
        let recovered = plan.inverse(&spectrum).expect("inverse");
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "Weighted GFT roundtrip failed: expected {a}, got {b}"
            );
        }
    }
}
