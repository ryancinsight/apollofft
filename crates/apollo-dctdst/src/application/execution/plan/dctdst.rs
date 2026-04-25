//! Reusable DCT/DST plan metadata surface.

use crate::domain::contracts::error::{DctDstError, DctDstResult};
use crate::domain::metadata::kind::{RealTransformConfig, RealTransformKind};
use crate::infrastructure::kernel::direct::{dct2, dct3, dst2, dst3};
use crate::infrastructure::kernel::fast::{
    dct2_fast, dct3_fast, dst2_fast, dst3_fast, FAST_THRESHOLD,
};

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
}
