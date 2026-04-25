//! Verification helpers for NUFFT.
//!
//! This module contains standalone verification tests that prove fundamental
//! mathematical invariants of the NUFFT implementation. Each test is derived
//! from a formal theorem and asserts on computed values, not merely on
//! `Result`/`Option` variants.
//!
//! ## Invariant hierarchy
//!
//! 1. **DC identity**: `F[0] = Σ_j c_j` (exact arithmetic consequence of `exp(0)=1`)
//! 2. **Adjoint identity**: `⟨A·c, f⟩ = ⟨c, A*·f⟩` (inner-product duality)
//! 3. **Kernel-width error ordering**: `err(W=2) > err(W=4) > err(W=6)` (Fessler–Sutton bound)
//! 4. **3D DC identity**: `F[0,0,0] = Σ_j c_j` (separable extension of invariant 1)
//! 5. **`type1_into` consistency**: allocating and in-place paths agree to machine precision
//! 6. **Monotone error decrease**: KB approximation error strictly decreases with `W`

#[cfg(test)]
mod tests {
    use crate::{
        nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d, nufft_type2_1d, NufftPlan1D,
        UniformDomain1D, UniformGrid3D,
    };
    use ndarray::Array1;
    use num_complex::Complex64;

    /// Theorem: Type-1 DC mode identity.
    ///
    /// For any positions `x_j` and complex amplitudes `c_j`, the `k = 0`
    /// Fourier coefficient of the type-1 NUFFT satisfies
    ///
    /// ```text
    /// F[0] = Σ_j c_j · exp(-2πi · 0 · x_j / L) = Σ_j c_j
    /// ```
    ///
    /// because `exp(0) = 1` for every `j` regardless of position. This identity
    /// holds exactly for the direct transform and approximately for the fast path,
    /// with the approximation error bounded by the KB spreading error.
    ///
    /// **Verification data:**
    /// - positions: `[0.1, 0.5, 1.3, 1.8]` (non-uniform, deliberately off-grid)
    /// - values: `[(1.0,0), (0.5,-0.3), (-0.25,0.8), (0.1,0.1)]`
    /// - `dc_exact = (1.0+0.5-0.25+0.1, 0-0.3+0.8+0.1) = (1.35, 0.6)`
    #[test]
    fn type1_dc_mode_is_sum_of_all_values_1d() {
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let positions = vec![0.1_f64, 0.5, 1.3, 1.8];
        let values = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, -0.3),
            Complex64::new(-0.25, 0.8),
            Complex64::new(0.1, 0.1),
        ];

        // Analytical DC = sum(c_j) = (1.35, 0.6)
        let dc_exact = Complex64::new(1.35, 0.6);

        let f_exact = nufft_type1_1d(&positions, &values, domain);
        let err_exact = (f_exact[0] - dc_exact).norm();
        assert!(
            err_exact < 1e-10,
            "exact DC mode error {err_exact}: got {:?}, expected {:?}",
            f_exact[0],
            dc_exact
        );
        // Verify the full output is finite.
        for (k, v) in f_exact.iter().enumerate() {
            assert!(
                v.norm().is_finite(),
                "exact output mode {k} is non-finite: {v:?}"
            );
        }

        // Fast path: kernel_width=6, oversampling=DEFAULT(2).
        let f_fast = nufft_type1_1d_fast(&positions, &values, domain, 6);
        let err_fast = (f_fast[0] - dc_exact).norm();
        assert!(
            err_fast < 1e-4,
            "fast DC mode error {err_fast}: got {:?}, expected {:?}",
            f_fast[0],
            dc_exact
        );
        for (k, v) in f_fast.iter().enumerate() {
            assert!(
                v.norm().is_finite(),
                "fast output mode {k} is non-finite: {v:?}"
            );
        }
    }

    /// Theorem: Type-1 / Type-2 adjoint identity.
    ///
    /// Define the type-1 operator `A` and type-2 operator `A*` by
    ///
    /// ```text
    /// (A·c)_k  = Σ_j  c_j  exp(-2πi k x_j / L)   (type-1)
    /// (A*·f)_j = Σ_k  f_k  exp(+2πi k x_j / L)   (type-2)
    /// ```
    ///
    /// They satisfy the real inner-product adjoint identity:
    ///
    /// ```text
    /// Re(⟨A·c, f⟩) = Re(⟨c, A*·f⟩)
    /// ```
    ///
    /// **Proof:** Expand `(A·c)_k`, swap summation order, and factor:
    ///
    /// ```text
    /// Re(Σ_k conj((A·c)_k) · f_k)
    ///   = Re(Σ_k Σ_j conj(c_j) exp(+2πi k x_j/L) · f_k)
    ///   = Re(Σ_j conj(c_j) · Σ_k f_k exp(+2πi k x_j/L))
    ///   = Re(Σ_j conj(c_j) · (A*·f)_j)  □
    /// ```
    ///
    /// The residual `|LHS - RHS|` must be below round-off for the exact
    /// direct transform (asserted at `< 1e-10`).
    #[test]
    fn type1_and_type2_adjoint_relationship_1d() {
        let domain = UniformDomain1D::new(4, 0.5).expect("domain");
        // 3 non-uniform positions in [0, L=2.0)
        let positions = vec![0.1_f64, 0.5, 1.3];
        let values_c = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(-0.5, 0.25),
            Complex64::new(0.25, -0.1),
        ];

        // F = A·c, length = domain.n = 4
        let f = nufft_type1_1d(&positions, &values_c, domain);

        // Arbitrary frequency-domain vector of length 4.
        let freq_domain_data = [
            Complex64::new(2.0, 0.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(0.5, 0.3),
            Complex64::new(-0.25, 0.1),
        ];
        let freq_domain_array = Array1::from_iter(freq_domain_data.iter().copied());

        // G = A*·f, length = positions.len() = 3
        let g = nufft_type2_1d(&freq_domain_array, &positions, domain);

        // LHS = Re(⟨F, freq_domain⟩) = Re(Σ_k conj(F[k]) · freq_domain[k])
        let lhs: f64 = f
            .iter()
            .zip(freq_domain_data.iter())
            .map(|(f_k, fd_k)| (f_k.conj() * fd_k).re)
            .sum();

        // RHS = Re(⟨c, G⟩) = Re(Σ_j conj(c_j) · G[j])
        let rhs: f64 = values_c
            .iter()
            .zip(g.iter())
            .map(|(c_j, g_j)| (c_j.conj() * g_j).re)
            .sum();

        let residual = (lhs - rhs).abs();
        assert!(
            residual < 1e-10,
            "adjoint identity failed: LHS={lhs:.15e}, RHS={rhs:.15e}, residual={residual:.3e}"
        );
    }

    /// Theorem: KB approximation error decreases monotonically with kernel half-width
    /// at fixed oversampling, and decreases further with higher oversampling.
    ///
    /// For fixed `σ`, increasing `W` by one divides the Fessler–Sutton error bound
    /// (2003, eq. 13) by a factor `> 1`, because the factor
    /// `sinh(β·√(1-(π/β)²)) / β` grows super-linearly with `β ∝ W`.
    ///
    /// **Cases verified (sigma, W, tolerance):**
    /// - `(2, 4, 1e-4)`: conservative, `β = π·0.75·8 ≈ 18.85`
    /// - `(2, 6, 1e-6)`: standard NUFFT setting, `β = π·0.75·12 ≈ 28.27`
    /// - `(4, 6, 1e-8)`: higher oversampling, `β = π·0.875·12 ≈ 32.99`
    #[test]
    fn fast_1d_tracks_exact_at_varying_kernel_widths() {
        let domain = UniformDomain1D::new(16, 0.125).expect("domain");
        let positions = vec![0.1_f64, 0.5, 0.9, 1.4, 1.7];
        let values = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(-0.5, 0.5),
            Complex64::new(0.75, -0.25),
            Complex64::new(-0.25, 0.75),
            Complex64::new(0.5, -0.5),
        ];
        let exact = nufft_type1_1d(&positions, &values, domain);

        // (sigma, kernel_width, max_absolute_error_tolerance)
        let cases = [(2_usize, 4_usize, 1e-4_f64), (2, 6, 1e-6), (4, 6, 1e-8)];
        for &(sigma, w, tol) in cases.iter() {
            let fast = NufftPlan1D::new(domain, sigma, w).type1(&positions, &values);
            assert_eq!(
                fast.len(),
                exact.len(),
                "sigma={sigma}, W={w}: output length mismatch"
            );
            let max_err = exact
                .iter()
                .zip(fast.iter())
                .map(|(e, f)| (e - f).norm())
                .fold(0.0_f64, f64::max);
            assert!(
                max_err <= tol,
                "sigma={sigma}, W={w}: max_err={max_err:.3e} > tol={tol:.0e}"
            );
        }
    }

    /// Theorem: 3D type-1 DC mode identity.
    ///
    /// The DC mode `(k_x=0, k_y=0, k_z=0)` of the 3D type-1 NUFFT satisfies
    ///
    /// ```text
    /// F[0,0,0] = Σ_j c_j
    /// ```
    ///
    /// because `exp(-2πi·(0·x_j/Lx + 0·y_j/Ly + 0·z_j/Lz)) = 1` for all `j`.
    /// This is the separable extension of the 1D DC identity to the tensor-product
    /// frequency lattice.
    ///
    /// **Verification data:**
    /// - values: `[(1.0,0), (0.5,0.5), (-0.75,0.25)]`
    /// - `dc_exact = (1.0+0.5-0.75, 0+0.5+0.25) = (0.75, 0.75)`
    #[test]
    fn type1_3d_dc_mode_is_sum_of_values() {
        let grid = UniformGrid3D::new(2, 2, 2, 1.0, 1.0, 1.0).expect("grid");
        let positions = vec![(0.1_f64, 0.2, 0.3), (0.5, 0.6, 0.7), (0.8, 0.1, 0.5)];
        let values = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.5),
            Complex64::new(-0.75, 0.25),
        ];

        // Analytical: dc = (1.0+0.5-0.75, 0.0+0.5+0.25) = (0.75, 0.75)
        let dc_exact = Complex64::new(0.75, 0.75);

        let f = nufft_type1_3d(&positions, &values, grid);
        assert_eq!(f.dim(), (2, 2, 2), "output shape mismatch");

        let err = (f[[0, 0, 0]] - dc_exact).norm();
        assert!(
            err < 1e-10,
            "3D DC mode error {err:.3e}: got {:?}, expected {:?}",
            f[[0, 0, 0]],
            dc_exact
        );

        for v in f.iter() {
            assert!(v.norm().is_finite(), "non-finite 3D output: {v:?}");
        }
    }

    /// Invariant: `NufftPlan1D::type1_into` produces bit-identical output to the
    /// allocating `NufftPlan1D::type1`.
    ///
    /// Both methods execute the same spreading → FFT → deconvolution pipeline.
    /// `type1` allocates internal buffers and calls `type1_into`; the outputs
    /// must therefore agree to within floating-point round-off (`< 1e-14`).
    ///
    /// **Buffer sizes for domain.n=8, sigma=2:**
    /// - `scratch_grid.len() = sigma * domain.n = 16`
    /// - `output.len() = domain.n = 8`
    #[test]
    fn plan_1d_type1_into_matches_type1_allocating() {
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        // sigma=2, kernel_width=6 ⟹ m = 2*8 = 16, n_out = 8
        let plan = NufftPlan1D::new(domain, 2, 6);

        let positions = vec![0.1_f64, 0.5, 1.3];
        let values = vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(-0.5, 0.25),
            Complex64::new(0.25, -0.1),
        ];

        // Allocating path.
        let out_alloc = plan.type1(&positions, &values);
        assert_eq!(
            out_alloc.len(),
            domain.n,
            "allocating output length mismatch"
        );

        // In-place path: scratch_grid length = sigma * domain.n = 16.
        let sigma: usize = 2;
        let mut scratch_grid = vec![Complex64::new(0.0, 0.0); sigma * domain.n];
        let mut output = vec![Complex64::new(0.0, 0.0); domain.n];
        plan.type1_into(&positions, &values, &mut scratch_grid, &mut output);

        for (k, (a, b)) in out_alloc.iter().zip(output.iter()).enumerate() {
            let err = (a - b).norm();
            assert!(
                err < 1e-14,
                "k={k}: type1={a:?} vs type1_into={b:?}, err={err:.3e}"
            );
        }
    }

    /// Theorem: KB NUFFT approximation error decreases strictly with kernel
    /// half-width `W` for fixed oversampling `σ = 2`.
    ///
    /// The Fessler–Sutton error bound (2003, eq. 13) is an exponentially
    /// decreasing function of `W` for the optimal `β = π·(1 - 1/(2σ))·2W`.
    /// This test validates the strict ordering
    ///
    /// ```text
    /// err(W=2) > err(W=4) > err(W=6)
    /// ```
    ///
    /// for a bandlimited signal with 8 unit-magnitude sources at uniform positions
    /// in `[0, L=8.0)`, and confirms `err(W=6) < 1e-5` (practical accuracy bound).
    ///
    /// **Source amplitudes:** `exp(i·k·π/4)` for `k = 0, …, 7` (unit magnitude,
    /// uniformly spaced phases — a standard benchmark for NUFFT error studies).
    #[test]
    fn nufft_approximation_error_decreases_with_kernel_width() {
        let domain = UniformDomain1D::new(32, 0.25).expect("domain");
        // 8 uniformly spaced positions in [0, domain.length() = 8.0)
        let positions: Vec<f64> = (0..8).map(|i| i as f64).collect();
        // Unit-magnitude values at angles k·π/4 for k = 0..8
        let values: Vec<Complex64> = (0..8)
            .map(|k| {
                let angle = k as f64 * std::f64::consts::PI / 4.0;
                Complex64::new(angle.cos(), angle.sin())
            })
            .collect();

        let exact = nufft_type1_1d(&positions, &values, domain);

        let max_abs_err = |w: usize| -> f64 {
            let fast = NufftPlan1D::new(domain, 2, w).type1(&positions, &values);
            exact
                .iter()
                .zip(fast.iter())
                .map(|(e, f)| (e - f).norm())
                .fold(0.0_f64, f64::max)
        };

        let err_2 = max_abs_err(2);
        let err_4 = max_abs_err(4);
        let err_6 = max_abs_err(6);

        assert!(
            err_2 > err_4,
            "error must decrease: err(W=2)={err_2:.3e} should exceed err(W=4)={err_4:.3e}"
        );
        assert!(
            err_4 > err_6,
            "error must decrease: err(W=4)={err_4:.3e} should exceed err(W=6)={err_6:.3e}"
        );
        assert!(
            err_6 < 1e-5,
            "err(W=6)={err_6:.3e} must be below practical accuracy threshold 1e-5"
        );
    }
}
