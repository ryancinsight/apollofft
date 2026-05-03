//! 1D FFT plan.

use crate::application::execution::kernel::radix2::{
    build_forward_twiddle_table_32, build_forward_twiddle_table_64, build_inverse_twiddle_table_32,
    build_inverse_twiddle_table_64, build_real_fwd_post_twiddles_64,
    forward_inplace_32_with_twiddles, forward_inplace_64_with_twiddles, forward_real_inplace_64,
    inverse_inplace_32_with_twiddles, inverse_inplace_64_with_twiddles, inverse_real_inplace_64,
};
use crate::application::execution::kernel::{fft_forward_32, fft_inverse_32};
use crate::application::execution::plan::fft::real_storage::RealFftData;
use crate::domain::metadata::precision::PrecisionProfile;
use crate::domain::metadata::shape::Shape1D;
use half::f16;
use ndarray::{Array1, Zip};
use num_complex::Complex32;
use num_complex::Complex64;
use std::sync::Mutex;

/// Reusable 1D FFT plan.
///
/// The plan owns the transform length and precision contract. Forward and
/// inverse transforms use Apollo's in-repo auto-selected FFT kernel, with
/// radix-2 execution for power-of-two lengths and Bluestein execution for
/// non-power-of-two lengths. The inverse path uses FFTW-compatible
/// normalization.
///
/// # Theorem
///
/// Let `F` denote the forward discrete Fourier transform and `F⁻¹` denote the
/// inverse path with `1 / n` normalization. For every signal `x ∈ ℂⁿ`,
/// `F⁻¹(F(x)) = x` in exact arithmetic. The implementation is linear because
/// both passes are matrix-vector products against the DFT basis. The inverse
/// normalization matches the standard FFTW convention used throughout Apollo.
///
/// # Proof sketch
///
/// The forward kernel evaluates the Vandermonde form
/// `X_k = Σ_n x_n exp(-2πi kn / N)`.
/// The inverse kernel evaluates
/// `x_n = (1/N) Σ_k X_k exp(2πi kn / N)`.
/// These are exact inverses over the complex field. Floating-point roundoff is
/// bounded by the chosen precision profile and the input length.
///
/// # Complexity
///
/// Power-of-two lengths use `O(N log N)` time and `O(N)` additional plan state
/// (for the precomputed twiddle table). Non-power-of-two lengths use Bluestein
/// convolution with `O(M log M)` time and `O(M)` plan state, where `M` is the
/// next convolution length selected by the Bluestein plan.
///
/// # Precomputed twiddle tables (power-of-two optimization)
///
/// For power-of-two `N`, the plan precomputes a contiguous per-stage forward and
/// inverse twiddle table of total size `N-1` entries each. Stage s (group length
/// `len = 2^s`) occupies `len/2` contiguous entries in the table; the butterfly
/// loop reads them sequentially with no stride. This eliminates both:
/// 1. The per-call `Vec<Complex64>` allocation (N/2 entries) that the naïve path
///    allocates on every `forward_inplace_64` call.
/// 2. The strided twiddle access `T[j * stride]` with `stride = N/len`, which at
///    early stages (len=2, stride=N/2) reads elements separated by N/2 entries,
///    causing L1-cache misses for N ≥ 256.
///
/// The contiguous layout stores stage twiddles sequentially, so the butterfly
/// inner loop reads `cache_line_size / sizeof(Complex64)` twiddles per cache line
/// regardless of N. This is the same principle used by FFTW's split-radix planner.
///
/// # Bluestein scratch reuse
///
/// For non-power-of-two lengths the plan caches a `Mutex<Vec<Complex64>>`
/// scratch buffer of length `M` alongside the Bluestein plan. On each call the
/// mutex is locked, the scratch is reused for the convolution, and the lock is
/// released. This eliminates the per-call `Vec` allocation that would otherwise
/// occur at O(M log M) work boundary — the critical section overhead is O(1).
///
/// # Failure modes
///
/// - zero-length shapes are rejected by `Shape1D::new`
/// - caller-supplied buffers must match the plan length
/// - non-contiguous ndarray buffers are rejected by `expect` on slice access
pub struct FftPlan1D {
    n: usize,
    precision: PrecisionProfile,
    bluestein_plan: Option<crate::application::execution::kernel::bluestein::BluesteinPlan64>,
    /// Scratch buffer for Bluestein convolution, reused across calls via Mutex.
    bluestein_scratch: Option<Mutex<Vec<Complex64>>>,
    /// Precomputed contiguous per-stage forward twiddle table for power-of-two N.
    /// Layout: stage s occupies entries [base..base+half] where half = 2^(s-1).
    /// Total length = N-1. `None` for non-power-of-two (Bluestein handles those).
    twiddle_fwd_64: Option<Vec<Complex64>>,
    /// Precomputed contiguous per-stage inverse twiddle table for power-of-two N.
    twiddle_inv_64: Option<Vec<Complex64>>,
    /// Precomputed f32 forward twiddle table (used in LOW_PRECISION_F32 mode).
    twiddle_fwd_32: Option<Vec<Complex32>>,
    /// Precomputed f32 inverse twiddle table (used in LOW_PRECISION_F32 mode).
    twiddle_inv_32: Option<Vec<Complex32>>,
    /// Post-processing twiddles for the real-input half-spectrum FFT.
    /// Entry k = exp(-2πi·k/N) for k = 0..=N/2. Length N/2+1.
    /// `Some` iff `N` is a power of two ≥ 4; `None` otherwise.
    real_fwd_post_twiddles: Option<Vec<Complex64>>,
    /// Pre-allocated scratch buffer for the inverse real FFT (iRFFT) trick.
    /// Length N/2 Complex64. Reused via Mutex. Avoids the per-call N-element
    /// allocation of the naive `input.to_owned()` path in `inverse_complex_to_real`.
    /// `Some` iff `N` is a power of two ≥ 4; `None` otherwise.
    real_inv_scratch: Option<Mutex<Vec<Complex64>>>,
}

impl std::fmt::Debug for FftPlan1D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan1D")
            .field("n", &self.n)
            .field("precision", &self.precision)
            .finish()
    }
}

impl FftPlan1D {
    /// Create a new 1D plan.
    #[must_use]
    pub fn new(shape: Shape1D) -> Self {
        Self::with_precision(shape, PrecisionProfile::HIGH_ACCURACY_F64)
    }

    /// Create a new 1D plan with an explicit precision profile.
    #[must_use]
    pub fn with_precision(shape: Shape1D, precision: PrecisionProfile) -> Self {
        let bluestein_plan = if shape.n.is_power_of_two() || shape.n <= 1 {
            None
        } else {
            Some(crate::application::execution::kernel::bluestein::BluesteinPlan64::new(shape.n))
        };
        // Pre-allocate the scratch buffer for Bluestein convolution so subsequent
        // calls reuse it instead of allocating a Vec on every invocation.
        let bluestein_scratch = bluestein_plan
            .as_ref()
            .map(|bp| Mutex::new(vec![Complex64::new(0.0, 0.0); bp.m()]));
        // Precompute contiguous per-stage twiddle tables for power-of-two sizes.
        // For non-power-of-two (Bluestein), these are None; Bluestein has its own tables.
        let is_pow2 = shape.n.is_power_of_two() && shape.n > 1;
        let twiddle_fwd_64 = if is_pow2 {
            Some(build_forward_twiddle_table_64(shape.n))
        } else {
            None
        };
        let twiddle_inv_64 = if is_pow2 {
            Some(build_inverse_twiddle_table_64(shape.n))
        } else {
            None
        };
        let twiddle_fwd_32 = if is_pow2 {
            Some(build_forward_twiddle_table_32(shape.n))
        } else {
            None
        };
        let twiddle_inv_32 = if is_pow2 {
            Some(build_inverse_twiddle_table_32(shape.n))
        } else {
            None
        };
        Self {
            n: shape.n,
            precision,
            bluestein_plan,
            bluestein_scratch,
            twiddle_fwd_64,
            twiddle_inv_64,
            twiddle_fwd_32,
            twiddle_inv_32,
            real_fwd_post_twiddles: if shape.n >= 4 && shape.n.is_power_of_two() {
                Some(build_real_fwd_post_twiddles_64(shape.n))
            } else {
                None
            },
            real_inv_scratch: if shape.n >= 4 && shape.n.is_power_of_two() {
                Some(Mutex::new(vec![Complex64::new(0.0, 0.0); shape.n >> 1]))
            } else {
                None
            },
        }
    }

    /// Return the plan length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Return the validated shape owned by this plan.
    #[must_use]
    pub fn shape(&self) -> Shape1D {
        Shape1D { n: self.n }
    }

    /// Return the precision profile used by this plan.
    #[must_use]
    pub fn precision_profile(&self) -> PrecisionProfile {
        self.precision
    }

    /// Whether the plan length is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Forward transform of a real signal.
    #[must_use]
    pub fn forward(&self, input: &Array1<f64>) -> Array1<Complex64> {
        self.forward_real_to_complex(input)
    }

    /// Inverse transform returning a real signal.
    #[must_use]
    pub fn inverse(&self, input: &Array1<Complex64>) -> Array1<f64> {
        self.inverse_complex_to_real(input)
    }

    /// Forward transform of a real signal using generic storage dispatch.
    #[must_use]
    pub fn forward_typed<T: RealFftData>(&self, input: &Array1<T>) -> Array1<T::Spectrum> {
        T::forward_1d(self, input)
    }

    /// Inverse transform of a complex spectrum using generic storage dispatch.
    #[must_use]
    pub fn inverse_typed<T: RealFftData>(&self, input: &Array1<T::Spectrum>) -> Array1<T> {
        T::inverse_1d(self, input)
    }

    /// Forward transform of a real signal.
    #[must_use]
    pub fn forward_real_to_complex(&self, input: &Array1<f64>) -> Array1<Complex64> {
        assert_eq!(input.len(), self.n, "forward input length mismatch");
        // For power-of-two N ≥ 4: use the half-length complex packing trick.
        // Pack x into N/2 complex samples, run N/2-point FFT (using the first
        // N/2-1 entries of the N-point twiddle table), then unpack via conjugate
        // symmetry. This halves the FFT work relative to a full N-point complex FFT.
        if let (Some(fft_tw), Some(post_tw)) = (&self.twiddle_fwd_64, &self.real_fwd_post_twiddles)
        {
            let input_slice = input.as_slice().expect("input must be contiguous");
            let mut output = Array1::<Complex64>::zeros(self.n);
            forward_real_inplace_64(
                input_slice,
                output.as_slice_mut().expect("output must be contiguous"),
                fft_tw,
                post_tw,
            );
            output
        } else {
            let mut output = input.mapv(|value| Complex64::new(value, 0.0));
            self.forward_complex_inplace(&mut output);
            output
        }
    }

    /// Forward transform of a real signal into a caller-supplied complex buffer.
    pub fn forward_real_to_complex_into(
        &self,
        input: &Array1<f64>,
        output: &mut Array1<Complex64>,
    ) {
        assert_eq!(input.len(), self.n, "forward input length mismatch");
        assert_eq!(output.len(), self.n, "forward output length mismatch");
        if let (Some(fft_tw), Some(post_tw)) = (&self.twiddle_fwd_64, &self.real_fwd_post_twiddles)
        {
            let input_slice = input.as_slice().expect("input must be contiguous");
            forward_real_inplace_64(
                input_slice,
                output.as_slice_mut().expect("output must be contiguous"),
                fft_tw,
                post_tw,
            );
        } else {
            Zip::from(&mut *output).and(input).for_each(|out, &value| {
                *out = Complex64::new(value, 0.0);
            });
            self.forward_complex_inplace(output);
        }
    }

    /// Compatibility alias for `forward_real_to_complex_into`.
    pub fn forward_into(&self, input: &Array1<f64>, output: &mut Array1<Complex64>) {
        self.forward_real_to_complex_into(input, output);
    }

    /// Inverse transform returning a real-valued signal.
    #[must_use]
    pub fn inverse_complex_to_real(&self, input: &Array1<Complex64>) -> Array1<f64> {
        let mut output = Array1::<f64>::zeros(self.n);
        self.inverse_complex_to_real_with_workspace(input, &mut output);
        output
    }

    /// Inverse transform into caller-owned real and scratch buffers.
    pub fn inverse_complex_to_real_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<f64>,
        scratch: &mut Array1<Complex64>,
    ) {
        assert_eq!(input.len(), self.n, "inverse input length mismatch");
        assert_eq!(output.len(), self.n, "inverse output length mismatch");
        assert_eq!(scratch.len(), self.n, "inverse scratch length mismatch");
        // Fast path: iRFFT trick. The caller-supplied scratch is N elements but
        // we only need M=N/2; use the lower half to avoid a lock when the caller
        // already owns a buffer.
        if let (Some(inv_tw), Some(post_tw)) = (&self.twiddle_inv_64, &self.real_fwd_post_twiddles)
        {
            let input_slice = input.as_slice().expect("input must be contiguous");
            let output_slice = output.as_slice_mut().expect("output must be contiguous");
            let scratch_slice = scratch.as_slice_mut().expect("scratch must be contiguous");
            let m = self.n >> 1;
            inverse_real_inplace_64(
                input_slice,
                output_slice,
                &mut scratch_slice[..m],
                inv_tw,
                post_tw,
            );
        } else {
            scratch.assign(input);
            self.inverse_complex_inplace(scratch);
            Zip::from(output)
                .and(scratch.view())
                .for_each(|out, value| {
                    *out = value.re;
                });
        }
    }

    fn inverse_complex_to_real_with_workspace(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<f64>,
    ) {
        assert_eq!(input.len(), self.n, "inverse input length mismatch");
        assert_eq!(output.len(), self.n, "inverse output length mismatch");
        // Fast path: iRFFT half-spectrum trick for PoT N ≥ 4.
        // Uses the M-point inverse FFT (M = N/2) instead of full N-point,
        // halving the FFT work and eliminating the per-call N-element allocation.
        if let (Some(inv_tw), Some(post_tw), Some(scratch_mu)) = (
            &self.twiddle_inv_64,
            &self.real_fwd_post_twiddles,
            &self.real_inv_scratch,
        ) {
            let input_slice = input.as_slice().expect("input must be contiguous");
            let output_slice = output.as_slice_mut().expect("output must be contiguous");
            let mut scratch = scratch_mu.lock().expect("real_inv_scratch mutex poisoned");
            inverse_real_inplace_64(input_slice, output_slice, &mut scratch, inv_tw, post_tw);
        } else {
            let mut spectrum = input.to_owned();
            self.inverse_complex_inplace(&mut spectrum);
            Zip::from(output)
                .and(spectrum.view())
                .for_each(|out, value| {
                    *out = value.re;
                });
        }
    }

    /// Forward transform of a complex signal in-place using a slice.
    pub fn forward_complex_slice_inplace(&self, data: &mut [Complex64]) {
        assert_eq!(data.len(), self.n, "complex forward length mismatch");
        if let (Some(plan), Some(scratch_mu)) = (&self.bluestein_plan, &self.bluestein_scratch) {
            let mut scratch = scratch_mu.lock().expect("bluestein scratch mutex poisoned");
            plan.forward_with_scratch(data, &mut scratch);
        } else if let Some(twiddles) = &self.twiddle_fwd_64 {
            forward_inplace_64_with_twiddles(data, twiddles);
        } else {
            crate::application::execution::kernel::radix2::forward_inplace_64(data);
        }
    }

    /// Inverse transform of a complex signal in-place with normalization using a slice.
    pub fn inverse_complex_slice_inplace(&self, data: &mut [Complex64]) {
        assert_eq!(data.len(), self.n, "complex inverse length mismatch");
        if let (Some(plan), Some(scratch_mu)) = (&self.bluestein_plan, &self.bluestein_scratch) {
            let mut scratch = scratch_mu.lock().expect("bluestein scratch mutex poisoned");
            plan.inverse_unnorm_with_scratch(data, &mut scratch);
            let scale = 1.0 / self.n as f64;
            for x in data.iter_mut() {
                *x *= scale;
            }
        } else if let Some(twiddles) = &self.twiddle_inv_64 {
            inverse_inplace_64_with_twiddles(data, twiddles);
        } else {
            crate::application::execution::kernel::radix2::inverse_inplace_64(data);
        }
    }

    /// Forward transform of a complex signal in-place.
    pub fn forward_complex_inplace(&self, data: &mut Array1<Complex64>) {
        self.forward_complex_slice_inplace(data.as_slice_mut().expect("Array must be contiguous"));
    }

    /// Inverse transform of a complex signal in-place with normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array1<Complex64>) {
        self.inverse_complex_slice_inplace(data.as_slice_mut().expect("Array must be contiguous"));
    }

    /// Compatibility alias for `inverse_complex_to_real_into`.
    pub fn inverse_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<f64>,
        scratch: &mut Array1<Complex64>,
    ) {
        self.inverse_complex_to_real_into(input, output, scratch);
    }

    /// Forward transform of a complex signal (allocating).
    #[must_use]
    pub fn forward_complex(&self, input: &Array1<Complex64>) -> Array1<Complex64> {
        let mut output = input.clone();
        self.forward_complex_inplace(&mut output);
        output
    }

    /// Inverse transform of a complex signal (allocating).
    #[must_use]
    pub fn inverse_complex(&self, input: &Array1<Complex64>) -> Array1<Complex64> {
        let mut output = input.clone();
        self.inverse_complex_inplace(&mut output);
        output
    }

    /// Forward transform of a real signal stored as `f32`.
    #[must_use]
    pub(crate) fn forward_f32(&self, input: &Array1<f32>) -> Array1<Complex32> {
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            let mut output = input.mapv(|value| Complex32::new(value, 0.0));
            let slice = output.as_slice_mut().expect("Array must be contiguous");
            if let Some(twiddles) = &self.twiddle_fwd_32 {
                forward_inplace_32_with_twiddles(slice, twiddles);
            } else {
                fft_forward_32(slice);
            }
            output
        } else {
            let promoted = input.mapv(f64::from);
            self.forward_real_to_complex(&promoted)
                .mapv(|value| Complex32::new(value.re as f32, value.im as f32))
        }
    }

    /// Inverse transform of an `f32`-storage complex spectrum.
    #[must_use]
    pub(crate) fn inverse_f32(&self, input: &Array1<Complex32>) -> Array1<f32> {
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            let mut output = input.clone();
            let slice = output.as_slice_mut().expect("Array must be contiguous");
            if let Some(twiddles) = &self.twiddle_inv_32 {
                inverse_inplace_32_with_twiddles(slice, twiddles);
            } else {
                fft_inverse_32(slice);
            }
            output.mapv(|value| value.re)
        } else {
            let promoted =
                input.mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im)));
            self.inverse_complex_to_real(&promoted)
                .mapv(|value| value as f32)
        }
    }

    /// Forward transform of a real signal stored as `f16`.
    #[must_use]
    pub(crate) fn forward_f16(&self, input: &Array1<f16>) -> Array1<Complex32> {
        if self.precision == PrecisionProfile::MIXED_PRECISION_F16_F32 {
            let complex_input = input.mapv(|value| Complex32::new(value.to_f32(), 0.0));
            let mut output = complex_input;
            fft_forward_32(output.as_slice_mut().expect("Array must be contiguous"));
            output
        } else {
            let promoted = input.mapv(|value| f64::from(value.to_f32()));
            self.forward_real_to_complex(&promoted)
                .mapv(|value| Complex32::new(value.re as f32, value.im as f32))
        }
    }

    /// Inverse transform of a complex spectrum to `f16` storage.
    #[must_use]
    pub(crate) fn inverse_f16(&self, input: &Array1<Complex32>) -> Array1<f16> {
        if self.precision == PrecisionProfile::MIXED_PRECISION_F16_F32 {
            let mut output = input.clone();
            fft_inverse_32(output.as_slice_mut().expect("Array must be contiguous"));
            output.mapv(|value| f16::from_f32(value.re))
        } else {
            let promoted =
                input.mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im)));
            self.inverse_complex_to_real(&promoted)
                .mapv(|value| f16::from_f32(value as f32))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use proptest::prelude::*;
    use proptest::proptest;

    fn energy_time_domain(signal: &Array1<f64>) -> f64 {
        signal.iter().map(|value| value * value).sum()
    }

    fn energy_frequency_domain(spectrum: &Array1<Complex64>) -> f64 {
        spectrum.iter().map(|value| value.norm_sqr()).sum()
    }

    #[test]
    fn forward_is_linear() {
        let shape = Shape1D::new(16).expect("shape");
        let plan = FftPlan1D::new(shape);
        let lhs = Array1::from_iter((0..16).map(|i| (i as f64 * 0.2).sin()));
        let rhs = Array1::from_iter((0..16).map(|i| (i as f64 * 0.3).cos()));
        let alpha = 1.75;
        let beta = -0.25;
        let combined = &lhs * alpha + &rhs * beta;

        let combined_hat = plan.forward_real_to_complex(&combined);
        let lhs_hat = plan.forward_real_to_complex(&lhs);
        let rhs_hat = plan.forward_real_to_complex(&rhs);

        for (actual, expected) in combined_hat.iter().zip(
            lhs_hat
                .iter()
                .zip(rhs_hat.iter())
                .map(|(x, y)| *x * alpha + *y * beta),
        ) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn parseval_identity_holds() {
        let shape = Shape1D::new(32).expect("shape");
        let plan = FftPlan1D::new(shape);
        let signal = Array1::from_iter((0..32).map(|i| {
            let x = i as f64;
            (0.17 * x).sin() + 0.5 * (0.41 * x).cos()
        }));

        let spectrum = plan.forward_real_to_complex(&signal);
        let time_energy = energy_time_domain(&signal);
        let spectral_energy = energy_frequency_domain(&spectrum) / shape.n as f64;

        assert_relative_eq!(time_energy, spectral_energy, epsilon = 1.0e-10);
    }

    #[test]
    fn caller_owned_paths_match_allocating_paths() {
        let shape = Shape1D::new(24).expect("shape");
        let plan = FftPlan1D::new(shape);
        let signal = Array1::from_iter((0..24).map(|i| (i as f64 * 0.1).sin()));

        let expected_hat = plan.forward_real_to_complex(&signal);
        let mut actual_hat = Array1::<Complex64>::zeros(shape.n);
        plan.forward_real_to_complex_into(&signal, &mut actual_hat);
        for (expected, actual) in expected_hat.iter().zip(actual_hat.iter()) {
            assert_relative_eq!(expected.re, actual.re, epsilon = 1.0e-12);
            assert_relative_eq!(expected.im, actual.im, epsilon = 1.0e-12);
        }

        let expected_signal = plan.inverse_complex_to_real(&expected_hat);
        let mut actual_signal = Array1::<f64>::zeros(shape.n);
        let mut scratch = Array1::<Complex64>::zeros(shape.n);
        plan.inverse_complex_to_real_into(&expected_hat, &mut actual_signal, &mut scratch);
        for (expected, actual) in expected_signal.iter().zip(actual_signal.iter()) {
            assert_relative_eq!(expected, actual, epsilon = 1.0e-12);
        }
    }

    #[test]
    #[should_panic(expected = "forward output length mismatch")]
    fn forward_rejects_mismatched_output_length() {
        let shape = Shape1D::new(8).expect("shape");
        let plan = FftPlan1D::new(shape);
        let signal = Array1::from_elem(shape.n, 1.0);
        let mut output = Array1::<Complex64>::zeros(shape.n - 1);
        plan.forward_real_to_complex_into(&signal, &mut output);
    }

    #[test]
    #[should_panic(expected = "inverse scratch length mismatch")]
    fn inverse_rejects_mismatched_scratch_length() {
        let shape = Shape1D::new(8).expect("shape");
        let plan = FftPlan1D::new(shape);
        let spectrum = Array1::<Complex64>::zeros(shape.n);
        let mut output = Array1::<f64>::zeros(shape.n);
        let mut scratch = Array1::<Complex64>::zeros(shape.n - 1);
        plan.inverse_complex_to_real_into(&spectrum, &mut output, &mut scratch);
    }

    #[test]
    fn typed_precision_profiles_remain_bounded_for_quantized_inputs() {
        let shape = Shape1D::new(32).expect("shape");
        let reference = Array1::from_iter((0..32).map(|i| {
            let x = i as f64;
            (0.09 * x).sin() + 0.25 * (0.71 * x).cos()
        }));

        let high_plan = FftPlan1D::with_precision(shape, PrecisionProfile::HIGH_ACCURACY_F64);
        let low_plan = FftPlan1D::with_precision(shape, PrecisionProfile::LOW_PRECISION_F32);
        let mixed_plan =
            FftPlan1D::with_precision(shape, PrecisionProfile::MIXED_PRECISION_F16_F32);

        let mixed_input = reference.mapv(|value| f16::from_f32(value as f32));
        let quantized_reference = mixed_input.mapv(|value| f64::from(value.to_f32()));
        let low_input = mixed_input.mapv(|value| value.to_f32());

        let low_recovered: Array1<f32> =
            low_plan.inverse_typed(&low_plan.forward_typed(&low_input));
        let low_roundtrip = low_recovered.mapv(f64::from);
        let mixed_recovered: Array1<f16> =
            mixed_plan.inverse_typed(&mixed_plan.forward_typed(&mixed_input));
        let mixed_roundtrip: Array1<f64> = mixed_recovered.mapv(|value| f64::from(value.to_f32()));
        let high_roundtrip = high_plan.inverse(&high_plan.forward(&reference));

        let low_error: f64 = low_roundtrip
            .iter()
            .zip(quantized_reference.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .sum();
        let mixed_error: f64 = mixed_roundtrip
            .iter()
            .zip(quantized_reference.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .sum();
        let high_error: f64 = high_roundtrip
            .iter()
            .zip(reference.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .sum();

        assert!(high_error <= 1.0e-12);
        assert!(low_error <= 1.0e-4);
        assert!(mixed_error <= low_error + 1.0e-6);
    }

    proptest! {
        #[test]
        fn roundtrip_holds_for_random_real_signals(
            signal in (1usize..33).prop_flat_map(|len| {
                prop::collection::vec(-10.0f64..10.0f64, len)
            })
        ) {
            let shape = Shape1D::new(signal.len()).expect("shape");
            let plan = FftPlan1D::new(shape);
            let input = Array1::from_vec(signal);
            let recovered = plan.inverse_complex_to_real(&plan.forward_real_to_complex(&input));
            for (expected, actual) in input.iter().zip(recovered.iter()) {
                prop_assert!((expected - actual).abs() < 1.0e-9);
            }
        }
    }
}
