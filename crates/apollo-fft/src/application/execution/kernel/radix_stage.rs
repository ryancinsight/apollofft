//! Shared normalization primitive for radix and Bluestein kernel modules.
//!
//! ## Contents
//!
//! - `normalize_inplace`: SSOT 1/N scale pass, used by inverse paths.

/// Scale every element of `data` in-place by `scale`.
///
/// ## SSOT role
///
/// Single authoritative 1/N normalization pass used by all inverse transform paths
/// (`bluestein`, `radix_composite`). Loop bounds and vectorization
/// contract live here; all callers delegate.
///
/// ## Zero-cost
///
/// Monomorphizes to a scalar-multiply loop. LLVM auto-vectorizes for `Complex64`
/// (256-bit AVX, 2 elements/cycle) and `Complex32` (256-bit AVX, 4 elements/cycle)
/// with `-C target-feature=+avx`.
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
