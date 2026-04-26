//! 1D Chirp Z-Transform Plan

use crate::application::execution::kernel::bluestein::{
    czt_bluestein_forward, czt_bluestein_forward_into,
};
use crate::application::execution::kernel::direct::czt_direct_forward;
use crate::domain::contracts::error::CztError;
use apollo_fft::application::plan::FftPlan1D;
use apollo_fft::types::{PrecisionProfile, Shape1D};
use ndarray::Array1;
use num_complex::Complex64;

/// Return whether a CZT input or output length satisfies the non-zero contract.
#[must_use]
pub fn is_valid_length(n: usize) -> bool {
    n > 0
}

/// Reusable chirp z-transform plan.
///
/// The plan is the single source of truth for CZT dimensions, spiral
/// parameters, chirp factors, convolution kernel, and FFT plan. Construction
/// validates non-zero lengths, finite non-zero `a` and `w`, and precomputes
/// all input-independent terms.
///
/// # Mathematical contract
///
/// For input `x[0..n)` this plan computes
/// `X[k] = sum_n x[n] a^-n w^(n k)` for `k in 0..m`.
/// `forward_direct` evaluates that definition directly. `forward` and
/// `forward_into` evaluate the same map through Bluestein convolution.
///
/// # Complexity
///
/// `forward_direct` costs `O(nm)` time. `forward` and `forward_into` cost
/// `O(p log p)` time with one `O(p)` convolution workspace, where
/// `p = convolution_len() >= n + m - 1`.
#[derive(Debug)]
pub struct CztPlan {
    n: usize,
    m: usize,
    a: Complex64,
    w: Complex64,
    convolution_len: usize,
    chirp_n: Vec<Complex64>,
    chirp_k: Vec<Complex64>,
    fft_kernel: Array1<Complex64>,
    fft_plan: FftPlan1D,
}

impl CztPlan {
    /// Create a validated CZT plan precomputing and caching the $O(P \log P)$ convolution kernel.
    pub fn new(n: usize, m: usize, a: Complex64, w: Complex64) -> Result<Self, CztError> {
        if n == 0 || m == 0 {
            return Err(CztError::EmptyLength);
        }
        if !a.re.is_finite() || !a.im.is_finite() || !w.re.is_finite() || !w.im.is_finite() {
            return Err(CztError::InvalidParameters);
        }
        if a == Complex64::new(0.0, 0.0) || w == Complex64::new(0.0, 0.0) {
            return Err(CztError::InvalidParameters);
        }

        let convolution_len = (n + m - 1).next_power_of_two();
        let fft_plan = FftPlan1D::with_precision(
            Shape1D::new(convolution_len).expect("CZT convolution length must be valid"),
            PrecisionProfile::HIGH_ACCURACY_F64,
        );

        let mut chirp_n = Vec::with_capacity(n);
        let mut chirp_k = Vec::with_capacity(m);
        let mut kernel = vec![Complex64::new(0.0, 0.0); convolution_len];

        for n_idx in 0..n {
            let nn = n_idx as f64;
            let phase = 0.5 * nn * nn;
            chirp_n.push((a.powf(-nn)) * w.powf(phase));
        }

        for k_idx in 0..m {
            let kk = k_idx as f64;
            let phase = 0.5 * kk * kk;
            chirp_k.push(w.powf(phase));
        }

        for k_idx in 0..m {
            kernel[k_idx] = w.powf(-0.5 * (k_idx as f64) * (k_idx as f64));
        }
        for k_idx in 1..n {
            kernel[convolution_len - k_idx] = w.powf(-0.5 * (k_idx as f64) * (k_idx as f64));
        }

        let mut fft_kernel = Array1::from_vec(kernel);
        fft_plan.forward_complex_inplace(&mut fft_kernel);

        Ok(Self {
            n,
            m,
            a,
            w,
            convolution_len,
            chirp_n,
            chirp_k,
            fft_kernel,
            fft_plan,
        })
    }

    /// Return the input length.
    #[must_use]
    pub const fn input_len(&self) -> usize {
        self.n
    }

    /// Return the output length.
    #[must_use]
    pub const fn output_len(&self) -> usize {
        self.m
    }

    /// Return the convolution length used by the fast path.
    #[must_use]
    pub const fn convolution_len(&self) -> usize {
        self.convolution_len
    }

    /// Forward direct CZT evaluation.
    pub fn forward_direct(&self, input: &Array1<Complex64>) -> Result<Array1<Complex64>, CztError> {
        if input.len() != self.n {
            return Err(CztError::LengthMismatch);
        }
        czt_direct_forward(input, self.m, self.a, self.w)
    }

    /// Forward CZT using Bluestein's convolution identity with precomputed caching.
    pub fn forward(&self, input: &Array1<Complex64>) -> Result<Array1<Complex64>, CztError> {
        if input.len() != self.n {
            return Err(CztError::LengthMismatch);
        }

        Ok(czt_bluestein_forward(
            input,
            self.m,
            self.convolution_len,
            &self.chirp_n,
            &self.chirp_k,
            &self.fft_kernel,
            &self.fft_plan,
        ))
    }

    /// Forward CZT into caller-owned output storage.
    pub fn forward_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<Complex64>,
    ) -> Result<(), CztError> {
        if input.len() != self.n || output.len() != self.m {
            return Err(CztError::LengthMismatch);
        }

        czt_bluestein_forward_into(
            input,
            output,
            self.convolution_len,
            &self.chirp_n,
            &self.chirp_k,
            &self.fft_kernel,
            &self.fft_plan,
        );
        Ok(())
    }

    /// In-place forward CZT.
    pub fn forward_inplace(&self, data: &mut Array1<Complex64>) -> Result<(), CztError> {
        let transformed = self.forward(data)?;
        *data = transformed;
        Ok(())
    }
}

#[cfg(test)]
mod spiral_collapse_tests {
    use super::*;

    /// CZT with A=1, W=exp(-2pi*i/N), M=N equals the N-point DFT (spiral-collapse theorem).
    #[test]
    fn czt_with_dft_parameters_equals_dft() {
        let n = 8usize;
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
        let input = Array1::from_shape_fn(n, |i| {
            Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos())
        });
        let plan = CztPlan::new(n, n, a, w).expect("valid DFT plan");
        let fast = plan.forward(&input).expect("fast");
        let direct = plan.forward_direct(&input).expect("direct");
        for (k, (cv, dv)) in fast.iter().zip(direct.iter()).enumerate() {
            let err = (cv - dv).norm();
            assert!(err < 1e-10, "CZT != DFT at k={k}: err={err}");
        }
    }

    /// Bluestein fast path matches direct for multiple non-trivial (N, M) pairs.
    #[test]
    fn czt_bluestein_equals_direct_for_fixed_inputs() {
        for (n, m) in [(3usize, 3usize), (5, 7), (11, 11), (13, 8)] {
            let a = Complex64::from_polar(0.9, 0.3);
            let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64 * 0.8);
            let inp = Array1::from_shape_fn(n, |i| {
                Complex64::new((i as f64 * 0.31).sin(), -(i as f64 * 0.19).cos())
            });
            let plan = CztPlan::new(n, m, a, w).expect("valid");
            let fast = plan.forward(&inp).expect("fast");
            let direct = plan.forward_direct(&inp).expect("direct");
            for (k, (f, d)) in fast.iter().zip(direct.iter()).enumerate() {
                let err = (f - d).norm();
                assert!(err < 1e-9, "n={n} m={m} k={k} err={err}");
            }
        }
    }

    /// Theorem (Spiral-Collapse, independent cross-check): CZT(x, N, exp(-2πi/N), 1)
    /// equals the N-point DFT, verified here against `apollo_fft::fft_1d_complex`
    /// which is entirely independent of the CZT implementation path.
    ///
    /// Proof: By Bluestein (1970), CZT with A=1, W=exp(-2πi/N), M=N reduces
    /// identically to the DFT summation Σ_n x[n] exp(-2πikn/N). The apollo_fft
    /// implementation uses a separate Cooley-Tukey / Bluestein kernel path; any
    /// shared sign or index bug in the CZT path would produce a measurable mismatch.
    #[test]
    fn czt_dft_parameters_match_independent_fft_implementation() {
        let n = 8usize;
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
        let input = Array1::from_shape_fn(n, |i| {
            Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos())
        });
        let plan = CztPlan::new(n, n, a, w).expect("valid DFT plan");
        let czt_output = plan.forward(&input).expect("CZT forward");
        // Independent DFT via apollo_fft (separate Cooley-Tukey / Bluestein path)
        let fft_output = apollo_fft::fft_1d_complex(&input);
        for (k, (cv, fv)) in czt_output.iter().zip(fft_output.iter()).enumerate() {
            let err = (cv - fv).norm();
            assert!(
                err < 1e-9,
                "CZT does not match independent FFT at k={k}: czt={cv:?}, fft={fv:?}, err={err:.3e}"
            );
        }
    }

    #[test]
    fn rejects_zero_length() {
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::new(0.9, 0.1);
        assert!(matches!(
            CztPlan::new(0, 5, a, w),
            Err(CztError::EmptyLength)
        ));
        assert!(matches!(
            CztPlan::new(5, 0, a, w),
            Err(CztError::EmptyLength)
        ));
    }

    #[test]
    fn rejects_zero_a() {
        let a = Complex64::new(0.0, 0.0);
        let w = Complex64::new(0.9, 0.1);
        assert!(matches!(
            CztPlan::new(4, 4, a, w),
            Err(CztError::InvalidParameters)
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn direct_matches_reference_for_small_sequence() {
        let input = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.5, 0.25),
            Complex64::new(-0.75, 0.5),
        ]);
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / 8.0);
        let plan = CztPlan::new(input.len(), 4, a, w).expect("valid plan");
        let direct = plan.forward_direct(&input).expect("direct");
        let fast = plan.forward(&input).expect("fast");
        for (lhs, rhs) in direct.iter().zip(fast.iter()) {
            assert_relative_eq!(lhs.re, rhs.re, epsilon = 1.0e-9);
            assert_relative_eq!(lhs.im, rhs.im, epsilon = 1.0e-9);
        }
    }

    #[test]
    fn rejects_invalid_lengths() {
        assert!(matches!(
            CztPlan::new(0, 4, Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)),
            Err(CztError::EmptyLength)
        ));
        assert!(matches!(
            CztPlan::new(4, 0, Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)),
            Err(CztError::EmptyLength)
        ));
    }

    #[test]
    fn length_mismatch_is_rejected() {
        let plan = CztPlan::new(
            4,
            4,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / 8.0),
        )
        .expect("valid plan");
        let bad = Array1::from_vec(vec![Complex64::new(0.0, 0.0); 3]);
        assert!(matches!(plan.forward(&bad), Err(CztError::LengthMismatch)));
        let good = Array1::from_vec(vec![Complex64::new(0.0, 0.0); 4]);
        let mut bad_output = Array1::from_vec(vec![Complex64::new(0.0, 0.0); 3]);
        assert!(matches!(
            plan.forward_into(&good, &mut bad_output),
            Err(CztError::LengthMismatch)
        ));
    }

    #[test]
    fn forward_into_matches_allocating_fast_path() {
        let input = Array1::from_vec(vec![
            Complex64::new(0.25, 0.5),
            Complex64::new(-0.75, 1.0),
            Complex64::new(1.25, -0.25),
            Complex64::new(0.5, 0.125),
            Complex64::new(-0.375, -0.75),
        ]);
        let plan = CztPlan::new(
            input.len(),
            7,
            Complex64::from_polar(1.0, 0.125),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / 11.0),
        )
        .expect("valid plan");

        let expected = plan.forward(&input).expect("allocating fast path");
        let mut actual = Array1::<Complex64>::zeros(plan.output_len());
        plan.forward_into(&input, &mut actual)
            .expect("caller-owned fast path");

        for (lhs, rhs) in expected.iter().zip(actual.iter()) {
            assert_relative_eq!(lhs.re, rhs.re, epsilon = 1.0e-12);
            assert_relative_eq!(lhs.im, rhs.im, epsilon = 1.0e-12);
        }
    }
}
