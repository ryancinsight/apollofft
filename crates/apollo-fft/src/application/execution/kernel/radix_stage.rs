//! Generic Winograd Cooley-Tukey stage loop shared by radix-8, -16, -32, and -64 kernels.
//!
//! ## Invariant
//!
//! All radix-K DIT kernels (K ∈ {8, 16, 32, 64}) share identical outer
//! stage-loop structure:
//!
//!   1. Digit-reverse permutation in base K.
//!   2. Stage loop: for m = 1, RADIX, RADIX², … < N:
//!      - For each non-overlapping chunk of length `m * RADIX`:
//!        - For each j in `0..m`: apply twiddle by recurrence, call inner DFT.
//!
//! The only per-radix variation is the inner DFT closure (`dft`) and the
//! scalar element type. This module provides a single zero-cost generic driver
//! so all four radices share one authoritative algorithm body.
//!
//! ## References
//!
//! - Winograd, S. (1978). On computing the discrete Fourier transform.
//!   *Mathematics of Computation*, 32(141), 175–199.

use super::radix_permute::digit_reverse_permute_pow2_radix;
use super::radix_shape::stage_twiddle_slice;
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;

// ── Scalar trait ─────────────────────────────────────────────────────────────

/// Arithmetic interface required by the Winograd Cooley-Tukey stage loop.
///
/// Implemented for [`Complex64`] and [`Complex32`]. Every method is marked
/// `#[inline]` so monomorphization eliminates all trait overhead.
///
/// # Invariants
///
/// - `zero()` returns the additive identity `0 + 0i`.
/// - `twiddle_step(sign, j, len)` computes `e^(i · sign · 2π · j / len)`.
/// - `apply_twiddle(val, tw)` computes the complex product `val × tw`.
pub(crate) trait WinogradScalar: Copy + Send + Sync + 'static {
    /// Compute the stage twiddle factor `e^(i · sign · 2π · j / len)`.
    ///
    /// `sign = -1.0` for the forward transform, `+1.0` for inverse.
    fn twiddle_step(sign: f64, j: usize, len: usize) -> Self;

    /// Complex multiply: `val × tw`.
    fn apply_twiddle(val: Self, tw: Self) -> Self;

    /// Additive identity: `0 + 0i`.
    fn zero() -> Self;
}

impl WinogradScalar for Complex64 {
    #[inline]
    fn twiddle_step(sign: f64, j: usize, len: usize) -> Self {
        let a = sign * std::f64::consts::TAU * j as f64 / len as f64;
        Complex64::new(a.cos(), a.sin())
    }

    #[inline]
    fn apply_twiddle(val: Self, tw: Self) -> Self {
        val * tw
    }

    #[inline]
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
}

impl WinogradScalar for Complex32 {
    #[inline]
    fn twiddle_step(sign: f64, j: usize, len: usize) -> Self {
        let a = sign * std::f64::consts::TAU * j as f64 / len as f64;
        Complex32::new(a.cos() as f32, a.sin() as f32)
    }

    #[inline]
    fn apply_twiddle(val: Self, tw: Self) -> Self {
        val * tw
    }

    #[inline]
    fn zero() -> Self {
        Complex32::new(0.0, 0.0)
    }
}

// ── WinogradComplex trait ─────────────────────────────────────────────────────

/// Arithmetic interface required by the Winograd short-DFT kernels (DFT-2 through DFT-64).
///
/// Implemented for [`Complex64`] and [`Complex32`]. Every method is `#[inline(always)]`
/// so monomorphization produces zero-overhead specializations for each precision.
///
/// # Design
///
/// The trait exposes exactly the operations needed by the DFT kernels:
/// - `+`, `-` via supertrait bounds
/// - `rot_pos_i` / `rot_neg_i` for the ±i twiddle (free swap+negate, zero multiplications)
/// - `from_f64_pair` to construct twiddle constants from f64 literals (cast to element precision)
/// - `cmul` for the complex multiply step in each butterfly combine stage
pub(crate) trait WinogradComplex:
    std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + Copy + Sized + Send + Sync + 'static
{
    /// Rotate by +i: `(re, im) → (−im, re)`.  Zero multiplications.
    fn rot_pos_i(self) -> Self;
    /// Rotate by −i: `(re, im) → (im, −re)`.  Zero multiplications.
    fn rot_neg_i(self) -> Self;
    /// Construct from an `(re, im)` pair of `f64`, cast to element precision.
    fn from_f64_pair(re: f64, im: f64) -> Self;
    /// Complex multiply: `self × rhs`.
    fn cmul(self, rhs: Self) -> Self;
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
    #[inline(always)]
    fn from_f64_pair(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
    #[inline(always)]
    fn cmul(self, rhs: Self) -> Self {
        self * rhs
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
    #[inline(always)]
    fn from_f64_pair(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }
    #[inline(always)]
    fn cmul(self, rhs: Self) -> Self {
        self * rhs
    }
}

// ── Generic stage driver ──────────────────────────────────────────────────────

/// In-place radix-`RADIX` Cooley-Tukey DIT FFT using a Winograd inner DFT.
///
/// ## Algorithm
///
/// 1. Digit-reverse permutation (base `RADIX`).
/// 2. Stage loop with `m = 1, RADIX, RADIX², …` while `m < data.len()`:
///    - `len = m × RADIX`.
///    - For each non-overlapping chunk of `len` elements:
///      - For each `j` in `0..m`:
///        - Compute step twiddle from `twiddles[j]` or on the fly.
///        - Fill `buf[RADIX]` by twiddle-recurrence gather from stride-`m` elements.
///        - Call `dft(&mut buf, inverse)`.
///        - Scatter `buf` back at stride `m`.
///
/// ## Parallelism
///
/// When `parallel_threshold` is `Some(t)` and `data.len() ≥ t` with more than
/// one chunk in the current stage, chunks are dispatched via Rayon
/// `par_chunks_exact_mut`. For radix-8 and radix-16 (smaller per-stage work)
/// pass `None`; for radix-32 and radix-64 pass
/// `Some(RADIX_PARALLEL_CHUNK_THRESHOLD)`.
///
/// ## Invariants
///
/// - `data.len()` is a power of two whose `log₂` is divisible by `log₂(RADIX)`.
/// - `RADIX` is a power of two in `{8, 16, 32, 64}`.
/// - `dft` implements the Winograd short DFT for exactly `RADIX` elements.
///
/// ## Mathematical justification
///
/// The twiddle recurrence `tw_p = step × tw_{p-1}` avoids `O(RADIX)` calls to
/// `cos`/`sin` per `j`, preserving `O(N log_RADIX N)` work per stage.

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
/// Monomorphizes to a plain scalar-multiply loop. LLVM auto-vectorizes
/// for `Complex64` (256-bit AVX, 2 elements/cycle) and `Complex32`
/// (256-bit AVX, 4 elements/cycle) with `-C target-feature=+avx`.
///
/// ## Correctness
///
/// For `T = Complex64` and `S = f64`, the call `v *= scale` is
/// `Complex64::mul_assign(v, scale)` which multiplies both components by
/// `scale`, preserving the complex number semantics.
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

pub(crate) fn radix_winograd_inplace<const RADIX: usize, C, Dft>(
    data: &mut [C],
    inverse: bool,
    twiddles: Option<&[C]>,
    dft: &Dft,
    parallel_threshold: Option<usize>,
) where
    C: WinogradScalar,
    Dft: Fn(&mut [C; RADIX], bool) + Send + Sync,
{
    debug_assert!(data.len().is_power_of_two());
    if data.len() <= 1 {
        return;
    }
    digit_reverse_permute_pow2_radix::<RADIX, _>(data);
    let sign: f64 = if inverse { 1.0 } else { -1.0 };
    let mut m = 1usize;
    while m < data.len() {
        let len = m * RADIX;
        let half = len >> 1;
        let stage_twiddles = stage_twiddle_slice(twiddles, half);
        let use_parallel = parallel_threshold
            .map(|thr| data.len() >= thr && data.len() / len > 1)
            .unwrap_or(false);
        if use_parallel {
            data.par_chunks_exact_mut(len).for_each(|chunk| {
                process_chunk::<RADIX, C, Dft>(chunk, m, len, inverse, stage_twiddles, sign, dft);
            });
        } else {
            for chunk in data.chunks_exact_mut(len) {
                process_chunk::<RADIX, C, Dft>(chunk, m, len, inverse, stage_twiddles, sign, dft);
            }
        }
        m = len;
    }
}

// ── Inner per-chunk processor ─────────────────────────────────────────────────

#[inline(always)]
fn process_chunk<const RADIX: usize, C, Dft>(
    chunk: &mut [C],
    m: usize,
    len: usize,
    inverse: bool,
    stage_twiddles: Option<&[C]>,
    sign: f64,
    dft: &Dft,
) where
    C: WinogradScalar,
    Dft: Fn(&mut [C; RADIX], bool),
{
    for j in 0..m {
        let step = match stage_twiddles {
            Some(st) => st[j],
            None => C::twiddle_step(sign, j, len),
        };
        let mut buf = [C::zero(); RADIX];
        buf[0] = chunk[j];
        let mut tw = step;
        for p in 1..RADIX {
            buf[p] = C::apply_twiddle(chunk[j + p * m], tw);
            tw = C::apply_twiddle(tw, step);
        }
        dft(&mut buf, inverse);
        for p in 0..RADIX {
            chunk[j + p * m] = buf[p];
        }
    }
}
