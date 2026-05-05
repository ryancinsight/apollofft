//! Shared arithmetic traits and normalization primitive for all radix kernels.
//!
//! ## Purpose
//!
//! This module provides two pieces of shared infrastructure used across the
//! radix-2, -4, -8, -16, -32, and -64 kernel families:
//!
//! 1. [`WinogradComplex`] — a trait supplying the arithmetic operations required
//!    by every radix butterfly: the zero-multiplication ±i rotations
//!    (`rot_pos_i` / `rot_neg_i`) and the standard field operations inherited
//!    via supertraits.  Implemented for [`Complex64`] and [`Complex32`].
//!
//! 2. [`normalize_inplace`] — the single authoritative `1/N` scale pass applied
//!    after every unnormalized inverse transform (SSOT for normalization).
//!
//! ## Design notes
//!
//! - No algorithm body lives here; only shared contracts.
//! - Both exports are `pub(crate)` — nothing escapes the kernel module.
//! - All methods are `#[inline(always)]` so monomorphization produces zero
//!   overhead relative to hand-written concrete specializations.
//!
//! ## References
//!
//! Winograd, S. (1978). On computing the discrete Fourier transform.
//! *Mathematics of Computation*, 32(141), 175–199.

use num_complex::{Complex32, Complex64};

// ── WinogradComplex trait ─────────────────────────────────────────────────────

/// Arithmetic interface required by the Winograd short-DFT butterfly kernels.
///
/// Implemented for [`Complex64`] and [`Complex32`]. Every method is
/// `#[inline(always)]` so monomorphization produces zero-overhead
/// specializations for each precision.
///
/// # Design
///
/// The trait exposes the two zero-multiplication ±i rotations central to all
/// radix Cooley-Tukey butterflies:
///
/// | Method | Operation | Cost |
/// |--------|-----------|------|
/// | `rot_pos_i` | `(re, im) → (−im, re)` | 1 swap + 1 negate |
/// | `rot_neg_i` | `(re, im) → (im, −re)` | 1 swap + 1 negate |
///
/// Field arithmetic (`+`, `-`) is inherited from supertrait bounds.
///
/// # Correctness
///
/// For `z = re + i·im`:
/// - `z · i   = −im + i·re` → `rot_pos_i`
/// - `z · (−i) = im − i·re` → `rot_neg_i`
///
/// Both use zero multiplications — the compiler emits a field swap and a
/// conditional negate.
pub(crate) trait WinogradComplex:
    std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + Copy + Sized + Send + Sync + 'static
{
    /// Rotate by +i: `(re, im) → (−im, re)`.  Zero multiplications.
    fn rot_pos_i(self) -> Self;
    /// Rotate by −i: `(re, im) → (im, −re)`.  Zero multiplications.
    fn rot_neg_i(self) -> Self;
}

impl WinogradComplex for Complex64 {
    #[inline(always)]
    fn rot_pos_i(self) -> Self {
        Complex64::new(-self.im, self.re)
    }
    #[inline(always)]
    fn rot_neg_i(self) -> Self {
        Complex64::new(self.im, -self.re)
    }
}

impl WinogradComplex for Complex32 {
    #[inline(always)]
    fn rot_pos_i(self) -> Self {
        Complex32::new(-self.im, self.re)
    }
    #[inline(always)]
    fn rot_neg_i(self) -> Self {
        Complex32::new(self.im, -self.re)
    }
}

// ── Shared normalization primitive ────────────────────────────────────────────

/// Scale every element of `data` in-place by `scale`.
///
/// ## SSOT role
///
/// This is the single authoritative implementation of the `1/N` normalization
/// pass applied after unnormalized inverse transforms. All callers in the kernel
/// hierarchy — `bluestein`, `kernel_api`, and `radix2` — delegate here so that
/// the loop, its bounds, and its vectorization contract live in one place.
///
/// ## Zero-cost
///
/// Monomorphizes to a plain scalar-multiply loop. LLVM auto-vectorizes for
/// `Complex64` (256-bit AVX, 2 elements/cycle) and `Complex32` (256-bit AVX,
/// 4 elements/cycle) with `-C target-feature=+avx`.
///
/// ## Correctness
///
/// For `T = Complex64` and `S = f64`, `v *= scale` is
/// `Complex64::mul_assign(v, scale)`, which multiplies both real and imaginary
/// components by `scale`, preserving the complex number field semantics.
#[inline]
pub(crate) fn normalize_inplace<T, S>(data: &mut [T], scale: S)
where
    T: std::ops::MulAssign<S>,
    S: Copy,
{
    for v in data.iter_mut() {
        *v *= scale;
    }
}
