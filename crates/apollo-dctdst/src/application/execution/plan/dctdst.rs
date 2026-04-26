//! Reusable DCT/DST plan metadata surface.

use crate::domain::contracts::error::{DctDstError, DctDstResult};
use crate::domain::metadata::kind::{RealTransformConfig, RealTransformKind};
use crate::infrastructure::kernel::direct::{dct2, dct3, dst2, dst3};
use crate::infrastructure::kernel::fast::{
    dct2_fast, dct3_fast, dst2_fast, dst3_fast, FAST_THRESHOLD,
};
use apollo_fft::{f16, PrecisionProfile};

/// Reusable DCT/DST plan.
///
/// The plan owns a validated real-to-real transform length and kind.
///
/// # Theorem
///
/// The DCT-II/DCT-III pair and DST-II/DST-III pair are biorthogonal under the
/// unnormalized conventions implemented in this crate:
///
/// ```text
/// DCT-III(DCT-II(x)) = (N / 2) x
/// DST-III(DST-II(x)) = (N / 2) x
/// ```
///
/// Therefore multiplying the paired inverse by `2 / N` recovers the input in
/// exact arithmetic.
///
/// # Proof sketch
///
/// The cosine and sine basis functions used by the type-II/type-III pairs are
/// orthogonal over the half-sample shifted grid. The cross terms vanish by
/// finite trigonometric sum identities, and the diagonal terms evaluate to
/// `N / 2` under Apollo's unnormalized convention.
///
/// # Complexity
///
/// O(N log N) for N ≥ 16 (2N-point FFT fast path); O(N²) for N < 16 (direct
/// analytical kernel). Both paths use O(1) auxiliary storage for caller-owned
/// `*_into` paths (the fast path allocates a 2N complex buffer internally for
/// the FFT work area).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DctDstPlan {
    config: RealTransformConfig,
}

impl DctDstPlan {
    /// Create a validated DCT/DST plan.
    pub fn new(len: usize, kind: RealTransformKind) -> DctDstResult<Self> {
        Ok(Self {
            config: RealTransformConfig::new(len, kind)?,
        })
    }

    /// Return the validated configuration.
    #[must_use]
    pub const fn config(self) -> RealTransformConfig {
        self.config
    }

    /// Return transform length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.config.len()
    }

    /// Return true when transform length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.config.is_empty()
    }

    /// Return transform kind.
    #[must_use]
    pub const fn kind(self) -> RealTransformKind {
        self.config.kind()
    }

    /// Execute the forward transform and return allocated coefficients.
    ///
    /// Returns `LengthMismatch` when the input slice length differs from the
    /// plan length.
    ///
    /// # Complexity
    ///
    /// O(N log N) for N ≥ 16 (2N-point FFT fast path); O(N²) for N < 16 (direct
    /// analytical kernel).
    pub fn forward(&self, signal: &[f64]) -> DctDstResult<Vec<f64>> {
        let mut output = vec![0.0_f64; self.len()];
        self.forward_into(signal, &mut output)?;
        Ok(output)
    }

    /// Execute the forward transform into a caller-supplied buffer.
    ///
    /// Returns `LengthMismatch` when either slice length differs from the plan
    /// length.
    ///
    /// Dispatches to the O(N log N) FFT fast path for N ≥ 16, and to the
    /// direct O(N²) analytical kernel for N < 16.
    ///
    /// # Complexity
    ///
    /// O(N log N) for N ≥ 16 (2N-point FFT fast path); O(N²) for N < 16 (direct
    /// analytical kernel).
    pub fn forward_into(&self, signal: &[f64], output: &mut [f64]) -> DctDstResult<()> {
        if signal.len() != self.len() || output.len() != self.len() {
            return Err(DctDstError::LengthMismatch);
        }

        let n = self.len();
        if n >= FAST_THRESHOLD {
            match self.kind() {
                RealTransformKind::DctII => dct2_fast(signal, output),
                RealTransformKind::DctIII => dct3_fast(signal, output),
                RealTransformKind::DstII => dst2_fast(signal, output),
                RealTransformKind::DstIII => dst3_fast(signal, output),
            }
        } else {
            match self.kind() {
                RealTransformKind::DctII => dct2(signal, output),
                RealTransformKind::DctIII => dct3(signal, output),
                RealTransformKind::DstII => dst2(signal, output),
                RealTransformKind::DstIII => dst3(signal, output),
            }
        }

        Ok(())
    }

    /// Compute the inverse of the given forward transform.
    ///
    /// DCT-III is the inverse of DCT-II (up to a 2/N scaling factor).
    /// DST-III is the inverse of DST-II (up to a 2/N scaling factor).
    /// The result is scaled by 2/N to recover the original signal.
    ///
    /// Returns `LengthMismatch` when the input slice length differs from the
    /// plan length.
    ///
    /// # Complexity
    ///
    /// O(N log N) for N ≥ 16 (2N-point FFT fast path); O(N²) for N < 16 (direct
    /// analytical kernel).
    pub fn inverse(&self, signal: &[f64]) -> DctDstResult<Vec<f64>> {
        let mut output = vec![0.0_f64; self.len()];
        self.inverse_into(signal, &mut output)?;
        Ok(output)
    }

    /// Compute the inverse of the configured transform into caller-owned output.
    ///
    /// DCT-III is the inverse of DCT-II (up to a 2/N scaling factor).
    /// DST-III is the inverse of DST-II (up to a 2/N scaling factor).
    /// The result is scaled by 2/N to recover the original signal.
    ///
    /// Returns `LengthMismatch` when either slice length differs from the plan
    /// length.
    ///
    /// Dispatches to the O(N log N) FFT fast path for N ≥ 16, and to the
    /// direct O(N²) analytical kernel for N < 16.
    ///
    /// # Complexity
    ///
    /// O(N log N) for N ≥ 16 (2N-point FFT fast path); O(N²) for N < 16 (direct
    /// analytical kernel).
    pub fn inverse_into(&self, signal: &[f64], output: &mut [f64]) -> DctDstResult<()> {
        if signal.len() != self.len() {
            return Err(DctDstError::LengthMismatch);
        }
        if output.len() != self.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let n = self.len();
        let mut raw = vec![0.0_f64; n];

        if n >= FAST_THRESHOLD {
            match self.kind() {
                RealTransformKind::DctII => dct3_fast(signal, &mut raw),
                RealTransformKind::DstII => dst3_fast(signal, &mut raw),
                RealTransformKind::DctIII => dct2_fast(signal, &mut raw),
                RealTransformKind::DstIII => dst2_fast(signal, &mut raw),
            }
        } else {
            match self.kind() {
                RealTransformKind::DctII => dct3(signal, &mut raw),
                RealTransformKind::DstII => dst3(signal, &mut raw),
                RealTransformKind::DctIII => dct2(signal, &mut raw),
                RealTransformKind::DstIII => dst2(signal, &mut raw),
            }
        }

        let scale = 2.0 / n as f64;
        for (slot, value) in output.iter_mut().zip(raw.into_iter()) {
            *slot = value * scale;
        }
        Ok(())
    }

    /// Execute the forward transform for `f64`, `f32`, or mixed `f16` storage.
    ///
    /// Lower storage profiles reuse the crate's authoritative `f64` transform
    /// and quantize once into the caller-owned output slice. This avoids
    /// precision-specific algorithm forks and preserves the DCT/DST theorem
    /// surface.
    pub fn forward_typed_into<T: RealTransformStorage>(
        &self,
        signal: &[T],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        T::forward_into(self, signal, output, profile)
    }

    /// Execute the inverse transform for `f64`, `f32`, or mixed `f16` storage.
    pub fn inverse_typed_into<T: RealTransformStorage>(
        &self,
        signal: &[T],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        T::inverse_into(self, signal, output, profile)
    }
}

/// Real storage accepted by typed DCT/DST paths.
pub trait RealTransformStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage to owner arithmetic.
    fn to_f64(self) -> f64;
    /// Convert owner arithmetic to storage.
    fn from_f64(value: f64) -> Self;

    /// Execute forward transform into caller-owned storage.
    fn forward_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if signal.len() != plan.len() || output.len() != plan.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let input64: Vec<f64> = signal.iter().map(|value| value.to_f64()).collect();
        let mut output64 = vec![0.0_f64; plan.len()];
        plan.forward_into(&input64, &mut output64)?;
        for (slot, value) in output.iter_mut().zip(output64.into_iter()) {
            *slot = Self::from_f64(value);
        }
        Ok(())
    }

    /// Execute inverse transform into caller-owned storage.
    fn inverse_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if signal.len() != plan.len() || output.len() != plan.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let input64: Vec<f64> = signal.iter().map(|value| value.to_f64()).collect();
        let mut output64 = vec![0.0_f64; plan.len()];
        plan.inverse_into(&input64, &mut output64)?;
        for (slot, value) in output.iter_mut().zip(output64.into_iter()) {
            *slot = Self::from_f64(value);
        }
        Ok(())
    }
}

impl RealTransformStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn forward_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_into(signal, output)
    }

    fn inverse_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.inverse_into(signal, output)
    }
}

impl RealTransformStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl RealTransformStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> DctDstResult<()> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(DctDstError::PrecisionMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn typed_paths_support_f64_f32_and_mixed_f16_storage() {
        let plan = DctDstPlan::new(8, RealTransformKind::DctII).expect("valid plan");
        let signal64 = [1.0_f64, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
        let expected = plan.forward(&signal64).expect("forward");

        let mut out64 = [0.0_f64; 8];
        plan.forward_typed_into(&signal64, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
            .expect("typed f64 forward");
        for (actual, expected) in out64.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }

        let signal32 = signal64.map(|value| value as f32);
        let mut out32 = [0.0_f32; 8];
        plan.forward_typed_into(&signal32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed f32 forward");
        for (actual, expected) in out32.iter().zip(expected.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-5);
        }

        let signal16 = signal64.map(|value| f16::from_f32(value as f32));
        let mut out16 = [f16::from_f32(0.0); 8];
        plan.forward_typed_into(
            &signal16,
            &mut out16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed mixed f16 forward");
        for (actual, expected) in out16.iter().zip(expected.iter()) {
            assert!((f64::from(actual.to_f32()) - *expected).abs() < 2.0e-3);
        }
    }

    #[test]
    fn typed_path_rejects_profile_storage_mismatch() {
        let plan = DctDstPlan::new(4, RealTransformKind::DstII).expect("valid plan");
        let signal = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        assert!(matches!(
            plan.forward_typed_into(&signal, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
            Err(DctDstError::PrecisionMismatch)
        ));
    }
}
