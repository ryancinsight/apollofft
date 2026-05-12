//! Mixed-radix strategy facade.
//!
//! Routes in-place FFTs to the best available kernel for the given length:
//!
//! ## Dispatch hierarchy (f64 / f32)
//!
//! | Input length       | Kernel selected |
//! |--------------------|------------------|
//! | Power of two ≥ 2   | **Stockham autosort** — out-of-place ping-pong FFT between `data` and a thread-local scratch buffer; no bit-reversal permutation. AVX/FMA SIMD codelets for N = 4, 8, 64, 4096 (f32); N = 64 (f64). Falls back to generic `transform<F>` loop for other PoT sizes. |
//! | 2/3/5/7-smooth     | **Composite mixed-radix** — Cooley-Tukey DIT with digit-reversal permutation. |
//! | Other non-PoT      | **Bluestein chirp-Z** — pads to next PoT and runs Stockham internally. |
//!
//! ## Dispatch hierarchy (f16)
//!
//! f16 is a storage-only precision. All PoT sizes are promoted to f32, run
//! through the Stockham f32 kernel, and demoted back to f16 via
//! `run_f16_via_f32`. The bit-reversal radix2/radix4 kernels are **not** used
//! on the PoT path for any precision.
//!
//! ## SSOT principle
//!
//! Each precision × operation combination delegates to one of the above three
//! authoritative algorithm implementations. No algorithm body is duplicated
//! across precision variants.
//!
//! ## Precision strategy
//!
//! | Variant | Element type | Notes |
//! |---------|--------------|-------|
//! | `_64`   | `Complex64`  | Native f64; Stockham for PoT. |
//! | `_32`   | `Complex32`  | Native f32; Stockham for PoT. |
//! | `_f16`  | `Cf16`       | Promotes to f32 for all arithmetic; Stockham f32 for PoT. |

#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::uninit_vec)]

use super::f16_bridge::run_f16_via_f32;
use super::radix2_f16::Cf16;
use super::radix_shape::{factorize_composite, should_use_bluestein_instead_of_composite};
use super::{bluestein, radix2, radix2_f16, radix_composite, stockham};
use num_complex::{Complex32, Complex64};
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// ── Global backing caches (cross-thread sharing, written once per size) ───────

static TWIDDLE_FWD_64_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex64]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_INV_64_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex64]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_FWD_32_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex32]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_INV_32_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex32]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_FWD_F16_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Cf16]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_INV_F16_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Cf16]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static COMPOSITE_RADIX_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Option<Arc<[usize]>>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

// ── Thread-local fast-path caches (zero locking on the hot path after warmup) ─
//
// On a cache hit the hot path is:
//   TW_FWD_64.with(|c| c.borrow().get(&n).cloned()) — no atomic, no lock.
// On a miss, the global RwLock is consulted or the table is built once, then
// the Arc is stored into the thread-local HashMap for all future calls.

thread_local! {
    static TL_FWD_64: RefCell<HashMap<usize, Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_INV_64: RefCell<HashMap<usize, Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_FWD_32: RefCell<HashMap<usize, Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_INV_32: RefCell<HashMap<usize, Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_FWD_F16: RefCell<HashMap<usize, Arc<[Cf16]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_INV_F16: RefCell<HashMap<usize, Arc<[Cf16]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_COMPOSITE_RADIX: RefCell<HashMap<usize, Option<Arc<[usize]>>>> =
        RefCell::new(HashMap::with_capacity(8));
    // Single-size Stockham ping-pong scratch (one per thread; grows but never shrinks).
    static TL_STOCKHAM_SCRATCH_64: RefCell<Vec<Complex64>> =
        const { RefCell::new(Vec::new()) };
    static TL_STOCKHAM_SCRATCH_32: RefCell<Vec<Complex32>> =
        const { RefCell::new(Vec::new()) };
}

#[inline]
fn with_stockham_scratch_64<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
    TL_STOCKHAM_SCRATCH_64.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        if scratch.len() < n {
            // Grow without zero-init: Stockham kernel overwrites before reading.
            let cur = scratch.len();
            scratch.reserve(n.saturating_sub(cur));
            unsafe { scratch.set_len(n) };
        }
        f(&mut scratch[..n])
    })
}

#[inline]
fn with_stockham_scratch_32<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
    TL_STOCKHAM_SCRATCH_32.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        if scratch.len() < n {
            let cur = scratch.len();
            scratch.reserve(n.saturating_sub(cur));
            unsafe { scratch.set_len(n) };
        }
        f(&mut scratch[..n])
    })
}

/// Retrieves the Arc from the thread-local map or falls back to global.
/// `build_fn` is called at most once per (thread, size) pair.
#[inline]
fn tl_cached<T: Clone>(
    tl: &'static std::thread::LocalKey<RefCell<HashMap<usize, Arc<[T]>>>>,
    global: &'static std::sync::LazyLock<RwLock<HashMap<usize, Arc<[T]>>>>,
    n: usize,
    build_fn: impl FnOnce(usize) -> Vec<T>,
) -> Arc<[T]> {
    // Fast path: hit in this thread's HashMap — no lock or atomic.
    if let Some(tw) = tl.with(|c| c.borrow().get(&n).cloned()) {
        return tw;
    }
    // Slow path (once per thread per size): read or write the global cache.
    // NOTE: The read guard MUST be dropped (via let binding with `;`) before
    // calling global.write() to avoid a same-thread read→write deadlock with
    // parking_lot::RwLock, which does not allow upgrading a read lock on the
    // same thread.
    let tw = {
        let maybe_cached = global.read().get(&n).cloned(); // ReadGuard drops here at `;`
        if let Some(tw) = maybe_cached {
            tw
        } else {
            let new_tw: Arc<[T]> = Arc::from(build_fn(n));
            global
                .write()
                .entry(n)
                .or_insert_with(|| Arc::clone(&new_tw))
                .clone()
        }
    };
    tl.with(|c| c.borrow_mut().insert(n, Arc::clone(&tw)));
    tw
}

#[inline]
pub(crate) fn cached_twiddle_fwd_64(n: usize) -> Arc<[Complex64]> {
    tl_cached(
        &TL_FWD_64,
        &TWIDDLE_FWD_64_CACHE,
        n,
        radix2::build_forward_twiddle_table_64,
    )
}

#[inline]
pub(crate) fn cached_twiddle_inv_64(n: usize) -> Arc<[Complex64]> {
    tl_cached(
        &TL_INV_64,
        &TWIDDLE_INV_64_CACHE,
        n,
        radix2::build_inverse_twiddle_table_64,
    )
}

#[inline]
pub(crate) fn cached_twiddle_fwd_32(n: usize) -> Arc<[Complex32]> {
    tl_cached(
        &TL_FWD_32,
        &TWIDDLE_FWD_32_CACHE,
        n,
        radix2::build_forward_twiddle_table_32,
    )
}

#[inline]
pub(crate) fn cached_twiddle_inv_32(n: usize) -> Arc<[Complex32]> {
    tl_cached(
        &TL_INV_32,
        &TWIDDLE_INV_32_CACHE,
        n,
        radix2::build_inverse_twiddle_table_32,
    )
}

#[inline]
pub(crate) fn cached_twiddle_fwd_f16(n: usize) -> Arc<[Cf16]> {
    tl_cached(
        &TL_FWD_F16,
        &TWIDDLE_FWD_F16_CACHE,
        n,
        radix2_f16::build_forward_twiddle_table_f16,
    )
}

#[inline]
pub(crate) fn cached_twiddle_inv_f16(n: usize) -> Arc<[Cf16]> {
    tl_cached(
        &TL_INV_F16,
        &TWIDDLE_INV_F16_CACHE,
        n,
        radix2_f16::build_inverse_twiddle_table_f16,
    )
}

// ── SSOT dispatch macro ───────────────────────────────────────────────────────

/// Emit the dispatch body for a `_with_twiddles` function that is already
/// inside `if data.len().is_power_of_two()`.
///
/// Parameters:
/// - `$data`      — the `&mut [T]` expression
/// - `$twiddles`  — the `Option<&[T]>` expression
/// - `cache`      — thread-local twiddle-cache lookup (`fn(usize) -> Arc<[T]>`)
/// - `r8`, `r4`, `r2` — `with_twiddles_fn` paths (single function each)
///
/// When `$twiddles` is `None`, the cache is consulted (O(1) HashMap lookup
/// on the hot path after the first per-thread per-size warm-up, no locks).
/// The previous `_no` (twiddle-rebuilding) variants are no longer needed.
// ── f64 ───────────────────────────────────────────────────────────────────────

/// In-place forward FFT (unnormalized, f64) with optional precomputed twiddles.
#[inline]
pub fn forward_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: Option<&[Complex64]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        if let Some(tw) = twiddles {
            with_stockham_scratch_64(data.len(), |scratch| {
                <f64 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw);
            });
        } else {
            let tw = cached_twiddle_fwd_64(data.len());
            with_stockham_scratch_64(data.len(), |scratch| {
                <f64 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw.as_ref());
            });
        }
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                radix_composite::forward_inplace_with_radices(data, &radices);
                return;
            }
        }
        bluestein::forward_inplace_64(data);
    }
}

#[inline]
fn cached_composite_radices(n: usize) -> Option<Arc<[usize]>> {
    if let Some(radices) = TL_COMPOSITE_RADIX.with(|c| c.borrow().get(&n).cloned()) {
        return radices;
    }

    // Slow path: factorize and populate global + thread-local cache.
    // new_radices is cloned at most once: once for the global cache insert
    // (if a race requires it), zero times otherwise.
    let radices = {
        let maybe_cached = COMPOSITE_RADIX_CACHE.read().get(&n).cloned();
        if let Some(radices) = maybe_cached {
            radices
        } else {
            let new_radices = factorize_composite(n).map(|rad| Arc::from(rad.into_boxed_slice()));
            // or_insert_with needs an owned value; clone the Arc (cheap refcount bump).
            // Returns &Arc<[usize]>; clone once more to get owned Arc back.
            let inserted = COMPOSITE_RADIX_CACHE
                .write()
                .entry(n)
                .or_insert_with(|| match &new_radices {
                    Some(a) => Some(Arc::clone(a)),
                    None => None,
                })
                .clone();
            inserted
        }
    };

    TL_COMPOSITE_RADIX.with(|c| c.borrow_mut().insert(n, radices.clone()));
    radices
}

/// In-place inverse FFT (unnormalized, f64) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_unnorm_64_with_twiddles(
    data: &mut [Complex64],
    twiddles: Option<&[Complex64]>,
) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        if let Some(tw) = twiddles {
            with_stockham_scratch_64(data.len(), |scratch| {
                <f64 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw);
            });
        } else {
            let tw = cached_twiddle_inv_64(data.len());
            with_stockham_scratch_64(data.len(), |scratch| {
                <f64 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw.as_ref());
            });
        }
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                radix_composite::inverse_inplace_unnorm_with_radices(data, &radices);
                return;
            }
        }
        bluestein::inverse_inplace_unnorm_64(data);
    }
}

/// In-place inverse FFT normalized by 1/N (f64) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: Option<&[Complex64]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        inverse_inplace_unnorm_64_with_twiddles(data, twiddles);
        let scale = 1.0 / data.len() as f64;
        data.iter_mut().for_each(|value| *value *= scale);
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                radix_composite::inverse_inplace_with_radices(data, &radices);
                return;
            }
        }
        bluestein::inverse_inplace_64(data);
    }
}

/// In-place forward FFT (unnormalized, f64).
pub fn forward_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_fwd_64(data.len());
        forward_inplace_64_with_twiddles(data, Some(tw.as_ref()));
    } else {
        forward_inplace_64_with_twiddles(data, None);
    }
}

/// In-place inverse FFT (unnormalized, f64).
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_64(data.len());
        inverse_inplace_unnorm_64_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_unnorm_64_with_twiddles(data, None);
    }
}

/// In-place inverse FFT normalized by 1/N (f64).
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_64(data.len());
        inverse_inplace_64_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_64_with_twiddles(data, None);
    }
}

// ── f32 ───────────────────────────────────────────────────────────────────────

/// In-place forward FFT (unnormalized, f32) with optional precomputed twiddles.
#[inline]
pub fn forward_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: Option<&[Complex32]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        if let Some(tw) = twiddles {
            with_stockham_scratch_32(data.len(), |scratch| {
                <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw);
            });
        } else {
            let tw = cached_twiddle_fwd_32(data.len());
            with_stockham_scratch_32(data.len(), |scratch| {
                <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw.as_ref());
            });
        }
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                radix_composite::forward_inplace_with_radices(data, &radices);
                return;
            }
        }
        bluestein::forward_inplace_32(data);
    }
}

/// In-place inverse FFT (unnormalized, f32) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_unnorm_32_with_twiddles(
    data: &mut [Complex32],
    twiddles: Option<&[Complex32]>,
) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        if let Some(tw) = twiddles {
            with_stockham_scratch_32(data.len(), |scratch| {
                <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw);
            });
        } else {
            let tw = cached_twiddle_inv_32(data.len());
            with_stockham_scratch_32(data.len(), |scratch| {
                <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw.as_ref());
            });
        }
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                radix_composite::inverse_inplace_unnorm_with_radices(data, &radices);
                return;
            }
        }
        bluestein::inverse_inplace_unnorm_32(data);
    }
}

/// In-place inverse FFT normalized by 1/N (f32) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: Option<&[Complex32]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        inverse_inplace_unnorm_32_with_twiddles(data, twiddles);
        let scale = 1.0 / data.len() as f32;
        data.iter_mut().for_each(|value| *value *= scale);
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                radix_composite::inverse_inplace_with_radices(data, &radices);
                return;
            }
        }
        bluestein::inverse_inplace_32(data);
    }
}

/// In-place forward FFT (unnormalized, f32).
#[inline]
pub fn forward_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_fwd_32(data.len());
        with_stockham_scratch_32(data.len(), |scratch| {
            <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw.as_ref());
        });
    } else {
        forward_inplace_32_with_twiddles(data, None);
    }
}

/// In-place inverse FFT (unnormalized, f32).
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_32(data.len());
        inverse_inplace_unnorm_32_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_unnorm_32_with_twiddles(data, None);
    }
}

/// In-place inverse FFT normalized by 1/N (f32).
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_32(data.len());
        inverse_inplace_32_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_32_with_twiddles(data, None);
    }
}

// ── f16 ───────────────────────────────────────────────────────────────────────

/// In-place forward FFT (unnormalized, f16 storage) with optional precomputed twiddles.
///
/// ## Dispatch
///
/// All power-of-two sizes promote to f32, run the Stockham f32 autosort kernel
/// (no bit-reversal), and demote back to f16. Non-PoT 2/3/5/7-smooth sizes use
/// the composite mixed-radix path via `run_f16_via_f32`. Other lengths use
/// Bluestein-f32.
///
/// The `twiddles` parameter is accepted for API uniformity but is unused for
/// PoT sizes: the f32 Stockham kernel builds and caches its own f32 twiddle
/// table. Callers may pass `None`.
#[inline]
pub fn forward_inplace_f16_with_twiddles(data: &mut [Cf16], _twiddles: Option<&[Cf16]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        // Promote f16 → f32, run Stockham (no bit-reversal), demote f32 → f16.
        let n = data.len();
        let tw = cached_twiddle_fwd_32(n);
        run_f16_via_f32(data, |buf| {
            with_stockham_scratch_32(n, |scratch| {
                <f32 as stockham::StockhamKernel>::forward_with_scratch(buf, scratch, tw.as_ref());
            });
        });
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                run_f16_via_f32(data, |buf| {
                    radix_composite::forward_inplace_with_radices(buf, &radices)
                });
                return;
            }
        }
        run_f16_via_f32(data, bluestein::forward_inplace_32);
    }
}

/// In-place inverse FFT (unnormalized, f16 storage) with optional precomputed twiddles.
///
/// PoT sizes: promote f16→f32, run Stockham f32 (inverse twiddles, no 1/N scale),
/// demote f32→f16. See `forward_inplace_f16_with_twiddles` for the dispatch rationale.
#[inline]
pub fn inverse_inplace_unnorm_f16_with_twiddles(data: &mut [Cf16], _twiddles: Option<&[Cf16]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let n = data.len();
        let tw = cached_twiddle_inv_32(n);
        run_f16_via_f32(data, |buf| {
            with_stockham_scratch_32(n, |scratch| {
                <f32 as stockham::StockhamKernel>::forward_with_scratch(buf, scratch, tw.as_ref());
            });
        });
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                run_f16_via_f32(data, |buf| {
                    radix_composite::inverse_inplace_unnorm_with_radices(buf, &radices)
                });
                return;
            }
        }
        run_f16_via_f32(data, bluestein::inverse_inplace_unnorm_32);
    }
}

/// In-place inverse FFT normalized by 1/N (f16 storage) with optional precomputed twiddles.
///
/// PoT sizes: promote f16→f32, run Stockham f32 (inverse twiddles), apply 1/N scale,
/// demote f32→f16. See `forward_inplace_f16_with_twiddles` for the dispatch rationale.
#[inline]
pub fn inverse_inplace_f16_with_twiddles(data: &mut [Cf16], _twiddles: Option<&[Cf16]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let n = data.len();
        let tw = cached_twiddle_inv_32(n);
        run_f16_via_f32(data, |buf| {
            with_stockham_scratch_32(n, |scratch| {
                <f32 as stockham::StockhamKernel>::forward_with_scratch(buf, scratch, tw.as_ref());
            });
            let scale = 1.0f32 / n as f32;
            buf.iter_mut().for_each(|v| *v *= scale);
        });
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                run_f16_via_f32(data, |buf| {
                    radix_composite::inverse_inplace_with_radices(buf, &radices)
                });
                return;
            }
        }
        run_f16_via_f32(data, bluestein::inverse_inplace_32);
    }
}

/// In-place forward FFT (unnormalized, f16 storage).
pub fn forward_inplace_f16(data: &mut [Cf16]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_fwd_f16(data.len());
        forward_inplace_f16_with_twiddles(data, Some(tw.as_ref()));
    } else {
        forward_inplace_f16_with_twiddles(data, None);
    }
}

/// In-place inverse FFT (unnormalized, f16 storage).
pub fn inverse_inplace_unnorm_f16(data: &mut [Cf16]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_f16(data.len());
        inverse_inplace_unnorm_f16_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_unnorm_f16_with_twiddles(data, None);
    }
}

/// In-place inverse FFT normalized by 1/N (f16 storage).
pub fn inverse_inplace_f16(data: &mut [Cf16]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_f16(data.len());
        inverse_inplace_f16_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_f16_with_twiddles(data, None);
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::max_abs_err_64;
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward_64, dft_inverse_64};

    #[test]
    fn mixed_forward_n32_matches_direct() {
        let n = 32usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.17).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "mixed-radix forward mismatch err={err:.2e}");
    }

    #[test]
    fn mixed_inverse_unnorm_n32_matches_direct() {
        let n = 32usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.19).cos(), (k as f64 * 0.07).sin()))
            .collect();
        let mut got = input.clone();
        inverse_inplace_unnorm_64(&mut got);
        let expected = dft_inverse_64(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect::<Vec<_>>();
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "mixed-radix inverse mismatch err={err:.2e}");
    }

    #[test]
    fn mixed_f32_stockham_forward_inverse_roundtrip_n256() {
        let n = 256usize;
        let input: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new((k as f32 * 0.013).sin(), (k as f32 * 0.017).cos()))
            .collect();
        let mut got = input.clone();

        forward_inplace_32(&mut got);
        inverse_inplace_32(&mut got);

        let err = got
            .iter()
            .zip(input.iter())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0f32, f32::max);
        assert!(
            err < 1.0e-4,
            "f32 Stockham roundtrip mismatch err={err:.2e}"
        );
    }

    #[test]
    fn mixed_f32_stockham_forward_inverse_roundtrip_n512() {
        let n = 512usize;
        let input: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new((k as f32 * 0.011).sin(), (k as f32 * 0.019).cos()))
            .collect();
        let mut got = input.clone();

        forward_inplace_32(&mut got);
        inverse_inplace_32(&mut got);

        let err = got
            .iter()
            .zip(input.iter())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0f32, f32::max);
        assert!(
            err < 1.0e-4,
            "f32 Stockham N=512 roundtrip mismatch err={err:.2e}"
        );
    }

    #[test]
    fn mixed_f32_stockham_forward_inverse_roundtrip_n4096() {
        let n = 4096usize;
        let input: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new((k as f32 * 0.007).sin(), (k as f32 * 0.011).cos()))
            .collect();
        let mut got = input.clone();

        forward_inplace_32(&mut got);
        inverse_inplace_32(&mut got);

        let err = got
            .iter()
            .zip(input.iter())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0f32, f32::max);
        let tolerance = 8.0 * n as f32 * f32::EPSILON;
        assert!(
            err < tolerance,
            "f32 Stockham N=4096 roundtrip mismatch err={err:.2e} tolerance={tolerance:.2e}"
        );
    }

    #[test]
    fn mixed_f64_stockham_forward_inverse_roundtrip_n256() {
        let n = 256usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.013).sin(), (k as f64 * 0.017).cos()))
            .collect();
        let mut got = input.clone();

        forward_inplace_64(&mut got);
        inverse_inplace_64(&mut got);

        let err = max_abs_err_64(&got, &input);
        assert!(
            err < 1.0e-10,
            "f64 Stockham roundtrip mismatch err={err:.2e}"
        );
    }

    #[test]
    fn mixed_f64_stockham_forward_inverse_roundtrip_n512() {
        let n = 512usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.011).sin(), (k as f64 * 0.019).cos()))
            .collect();
        let mut got = input.clone();

        forward_inplace_64(&mut got);
        inverse_inplace_64(&mut got);

        let err = max_abs_err_64(&got, &input);
        assert!(
            err < 1.0e-10,
            "f64 Stockham N=512 roundtrip mismatch err={err:.2e}"
        );
    }
}
