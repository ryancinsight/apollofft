//! SSOT for Cooley-Tukey twiddle-factor table construction.
//!
//! ## Algorithm
//!
//! For an N-point power-of-two transform, the per-stage twiddle table is a
//! contiguous vector of length N − 1. Stage s (group length `len = 2^s`) stores
//! `half = len / 2` consecutive entries beginning at base `2^(s−1) − 1`:
//!
//! ```text
//! entry[base + j] = exp(i · sign · 2π · j / len),   j = 0..half
//! ```
//!
//! Twiddles are computed at f64 precision and demoted on store for f32 and f16.
//!
//! ## DRY invariant
//!
//! `build_twiddle_table<C>` is the single authoritative implementation.
//! All public typed wrappers (`build_twiddle_table_64`, `build_twiddle_table_32`,
//! `build_twiddle_table_f16`) are trivial one-line delegations that may not
//! contain any computation. Supporting additional precision targets requires
//! only a new `impl TwiddleOutput for <new type>` plus a one-line wrapper.
//!
//! ## References
//!
//! - Cooley, J.W. & Tukey, J.W. (1965). An algorithm for the machine calculation
//!   of complex Fourier series. *Mathematics of Computation*, 19(90), 297–301.

use super::radix2_f16::Cf16;
use num_complex::{Complex32, Complex64};

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Conversion from a twiddle angle to a stored complex element.
///
/// Implementing this trait for a type `C` adds support for building a
/// Cooley-Tukey twiddle table stored as `Vec<C>`.
///
/// # Invariant
///
/// `from_angle(a)` must return `exp(i · a)` rounded to the precision of `C`.
/// The angle `a` is computed in f64. Precision loss on conversion is acceptable
/// and documented in the caller's precisions model.
pub(crate) trait TwiddleOutput: Copy + Sized {
    /// Convert radian angle `a` to the stored complex element `exp(i·a)`.
    fn from_angle(a: f64) -> Self;
}

impl TwiddleOutput for Complex64 {
    #[inline]
    fn from_angle(a: f64) -> Self {
        Complex64::new(a.cos(), a.sin())
    }
}

impl TwiddleOutput for Complex32 {
    #[inline]
    fn from_angle(a: f64) -> Self {
        Complex32::new(a.cos() as f32, a.sin() as f32)
    }
}

impl TwiddleOutput for Cf16 {
    #[inline]
    fn from_angle(a: f64) -> Self {
        Cf16::from_f32_pair(a.cos() as f32, a.sin() as f32)
    }
}

// ── Generic builder ───────────────────────────────────────────────────────────

/// Build a contiguous per-stage Cooley-Tukey twiddle table for a power-of-two
/// transform of length `n`.
///
/// ## Parameters
///
/// - `n`: transform length (must be a power of two).
/// - `sign`: `−1.0` for the forward transform, `+1.0` for the inverse.
///
/// ## Output
///
/// A `Vec<C>` of length `n − 1`. Stage `s` occupies indices
/// `2^(s−1) − 1 .. 2^s − 1` (half = `2^(s−1)` entries). The entry at
/// relative position `j` within that stage is `C::from_angle(sign · 2π · j / len)`.
///
/// ## Preconditions
///
/// `n` must be a power of two. Violated preconditions trigger a debug assertion
/// and unspecified output in release mode.
pub(crate) fn build_twiddle_table<C: TwiddleOutput>(n: usize, sign: f64) -> Vec<C> {
    debug_assert!(n.is_power_of_two());
    if n <= 1 {
        return Vec::new();
    }
    let log_n = n.trailing_zeros() as usize;
    let mut table = Vec::with_capacity(n - 1);
    let mut len = 2usize;
    for _ in 0..log_n {
        let half = len >> 1;
        for j in 0..half {
            let a = sign * std::f64::consts::TAU * j as f64 / len as f64;
            table.push(C::from_angle(a));
        }
        len <<= 1;
    }
    table
}
