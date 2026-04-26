//! 1D Chirp Z-Transform Plan

use crate::application::execution::kernel::bluestein::{
    czt_bluestein_forward, czt_bluestein_forward_into,
};
use crate::application::execution::kernel::direct::czt_direct_forward;
use crate::domain::contracts::error::CztError;
use apollo_fft::application::plan::FftPlan1D;
use apollo_fft::f16;
use apollo_fft::types::{PrecisionProfile, Shape1D};
use ndarray::Array1;
use num_complex::{Complex32, Complex64};

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

    /// Forward CZT for `Complex64`, `Complex32`, or mixed two-lane `f16` storage.
    ///
    /// `Complex64` uses the native high-accuracy path. `Complex32` and mixed
    /// `[f16; 2]` storage convert through the owner kernel and quantize once
    /// into the caller-owned output.
    pub fn forward_typed_into<T: CztStorage>(
        &self,
        input: &Array1<T>,
        output: &mut Array1<T>,
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        T::forward_into(self, input, output, profile)
    }

    /// In-place forward CZT.
    pub fn forward_inplace(&self, data: &mut Array1<Complex64>) -> Result<(), CztError> {
        let transformed = self.forward(data)?;
        *data = transformed;
        Ok(())
    }
}

/// Complex storage accepted by typed CZT paths.
pub trait CztStorage: Copy + Send + Sync + 'static {
    /// Required precision profile for this storage type.
    const PROFILE: PrecisionProfile;

    /// Convert storage into the owner `Complex64` arithmetic path.
    fn to_complex64(self) -> Complex64;

    /// Convert owner arithmetic result back to storage.
    fn from_complex64(value: Complex64) -> Self;

    /// Execute forward transform into caller-owned storage.
    fn forward_into(
        plan: &CztPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        validate_profile(profile, Self::PROFILE)?;
        if input.len() != plan.input_len() || output.len() != plan.output_len() {
            return Err(CztError::LengthMismatch);
        }
        let input64 = Array1::from_iter(input.iter().copied().map(Self::to_complex64));
        let mut output64 = Array1::zeros(plan.output_len());
        plan.forward_into(&input64, &mut output64)?;
        for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
            *slot = Self::from_complex64(value);
        }
        Ok(())
    }
}

impl CztStorage for Complex64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_complex64(self) -> Complex64 {
        self
    }

    fn from_complex64(value: Complex64) -> Self {
        value
    }

    fn forward_into(
        plan: &CztPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_into(input, output)
    }
}

impl CztStorage for Complex32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_complex64(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }

    fn from_complex64(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }
}

impl CztStorage for [f16; 2] {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_complex64(self) -> Complex64 {
        Complex64::new(f64::from(self[0].to_f32()), f64::from(self[1].to_f32()))
    }

    fn from_complex64(value: Complex64) -> Self {
        [
            f16::from_f32(value.re as f32),
            f16::from_f32(value.im as f32),
        ]
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> Result<(), CztError> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(CztError::PrecisionMismatch)
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

    #[test]
    fn typed_paths_support_complex64_complex32_and_mixed_f16_storage() {
        let input64 = Array1::from_vec(vec![
            Complex64::new(0.25, 0.5),
            Complex64::new(-0.75, 1.0),
            Complex64::new(1.25, -0.25),
            Complex64::new(0.5, 0.125),
            Complex64::new(-0.375, -0.75),
        ]);
        let plan = CztPlan::new(
            input64.len(),
            7,
            Complex64::from_polar(1.0, 0.125),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / 11.0),
        )
        .expect("valid plan");
        let expected = plan.forward(&input64).expect("reference");

        let mut out64 = Array1::<Complex64>::zeros(plan.output_len());
        plan.forward_typed_into(&input64, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
            .expect("complex64 typed");
        for (actual, expected) in out64.iter().zip(expected.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }

        let input32 = input64.mapv(|value| Complex32::new(value.re as f32, value.im as f32));
        let mut out32 = Array1::<Complex32>::zeros(plan.output_len());
        plan.forward_typed_into(&input32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
            .expect("complex32 typed");
        for (actual, expected) in out32.iter().zip(expected.iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
            assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
        }

        let input16 = input64.mapv(|value| {
            [
                f16::from_f32(value.re as f32),
                f16::from_f32(value.im as f32),
            ]
        });
        let mut out16 = Array1::from_elem(plan.output_len(), [f16::from_f32(0.0); 2]);
        plan.forward_typed_into(
            &input16,
            &mut out16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("mixed f16 typed");
        for (actual, expected) in out16.iter().zip(expected.iter()) {
            let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound);
            assert!((f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound);
        }
    }

    #[test]
    fn typed_path_rejects_profile_storage_mismatch() {
        let plan = CztPlan::new(
            4,
            4,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / 8.0),
        )
        .expect("valid plan");
        let input = Array1::from_vec(vec![Complex32::new(1.0, 0.0); 4]);
        let mut output = Array1::<Complex32>::zeros(4);
        assert!(matches!(
            plan.forward_typed_into(&input, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
            Err(CztError::PrecisionMismatch)
        ));
    }
}
