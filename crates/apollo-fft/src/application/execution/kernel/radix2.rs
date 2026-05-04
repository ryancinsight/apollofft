//! Iterative Cooley-Tukey radix-2 DIT FFT kernel.
//!
//! ## Mathematical contract
//!
//! For N = 2^k, the forward transform computes:
//!
//! `X[k] = Σ_{n=0}^{N-1} x[n] · exp(-2πi · k·n / N)` (unnormalized)
//!
//! The unnormalized inverse computes:
//!
//! `y[n] = Σ_{k=0}^{N-1} X[k] · exp(+2πi · k·n / N)` (unnormalized, no 1/N factor)
//!
//! The normalized inverse divides the unnormalized result by N.
//!
//! ## Algorithm
//!
//! Theorem (Cooley-Tukey): The N-point DFT with N = 2^k decomposes into two
//! (N/2)-point DFTs via even/odd index splitting. Applied recursively this
//! yields O(N log₂ N) arithmetic operations. The iterative DIT form applies
//! a bit-reversal permutation once, then executes log₂ N butterfly stages
//! in-place, each stage of half-length butterflies using a precomputed twiddle.
//!
//! ## Twiddle table construction
//!
//! Theorem (Unified Twiddle Table): A single N/2-entry table
//! T[j] = exp(-2πi·j/N), j = 0…N/2-1, suffices for all log₂ N stages.
//!
//! Proof: In stage s the butterfly half-length is `half = 2^(s-1)`. The twiddle
//! for butterfly position j within each group of `len = 2·half` elements is
//! W_{len}^j = exp(-2πi·j/len). Substituting len = N/stride where stride = N/len
//! gives W_{len}^j = exp(-2πi·j·stride/N) = T[j·stride]. Hence every stage's
//! twiddle sequence is a stride-sub-sample of T with stride = N/len, which equals
//! N >> len.trailing_zeros(). The sub-sample reads T[j·stride] for j = 0…half-1,
//! accessing exactly the same values as the per-stage allocation without any
//! heap allocation. □
//!
//! Corollary: The full twiddle table is computed once before the butterfly loop
//! (O(N/2) trigonometric calls, amortized over all stages), eliminating all k
//! per-stage Vec allocations for N = 2^k.
//!
//! ## Energy invariance (Parseval / Plancherel)
//!
//! Theorem (Parseval): For any x ∈ ℂⁿ with DFT X computed by this kernel,
//! `Σ_n |x[n]|² = (1/N) · Σ_k |X[k]|²`.
//!
//! Proof sketch: The N×N DFT matrix F satisfies F·F* = N·I (unitary up to √N).
//! Therefore ‖Fx‖² = xᴴF*Fx = N·‖x‖², giving ‖X‖² = N·‖x‖². Dividing by N
//! yields the normalized Parseval identity. This kernel computes the unnormalized
//! Fx, so callers must divide the spectral energy sum by N to recover time-domain
//! energy — consistent with the `energy_frequency_domain / N` normalization in
//! the plan-level `parseval_identity_holds` test. □
//!
//! ## Failure modes
//!
//! - Empty slice: returns immediately (N=0).
//! - N=1: returns immediately (trivial transform).
//! - N not a power of 2: triggers `debug_assert!` in debug builds.

use num_complex::{Complex32, Complex64};

// ── SIMD butterfly helpers ────────────────────────────────────────────────────
//
// AVX + FMA butterfly for the general stages (len ≥ 32).
//
// All intrinsics are from AVX (256-bit load/store/unpack/permute) and FMA3
// (VFMADDSUBPD/PS). These features are present on all CPUs supporting FMA
// (Intel Haswell+, AMD Piledriver+). With `RUSTFLAGS=-C target-cpu=native`
// the `cfg(target_feature)` guards resolve to constant-true at compile time
// and the scalar fall-back is dead-code-eliminated.
//
// ## Correctness proof for `butterfly_f64`
//
// Let u = lo[j], v = hi[j], w = twiddle[j].
// Complex product:
//   tw = w · v = (w.re·v.re − w.im·v.im) + i(w.re·v.im + w.im·v.re)
//
// SIMD implementation (2 Complex64 per register):
//   ac      = [w[j].re, w[j].re, w[j+1].re, w[j+1].re]   (_mm256_unpacklo_pd)
//   bd      = [w[j].im, w[j].im, w[j+1].im, w[j+1].im]   (_mm256_unpackhi_pd)
//   v_perm  = [v[j].im, v[j].re, v[j+1].im, v[j+1].re]   (_mm256_permute_pd, imm=5)
//   tw_vec  = _mm256_fmaddsub_pd(ac, v, bd * v_perm)
//           = [ac·v.re − bd·v.im, ac·v.im + bd·v.re, …]   (alternating sub/add)
//           = [tw.re, tw.im, …]                            ✓
//   lo[j..] ← u + tw_vec
//   hi[j..] ← u − tw_vec                                  ✓
//
// Analogous proof holds for `butterfly_f32` (4 Complex32 per register) using
// VMOVSLDUP/VMOVSHDUP for twiddle broadcast and VPERMILPS(0xB1) for re/im swap.

/// Apply the Cooley–Tukey butterfly to `lo[1..]`, `hi[1..]`, `tw[1..]`.
///
/// `lo` and `hi` each have `half` elements; `tw` has `half` elements.
/// Element at index 0 (unity twiddle) must be handled by the caller.
/// Dispatches to `butterfly_f64_avx_fma` when the build target has AVX+FMA
/// (static cfg check, zero runtime overhead with `target-cpu=native`).
#[inline]
fn butterfly_f64(lo: &mut [Complex64], hi: &mut [Complex64], tw: &[Complex64]) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx",
        target_feature = "fma"
    ))]
    {
        // SAFETY: cfg guard ensures AVX and FMA are available at compile time.
        unsafe { return butterfly_f64_avx_fma(lo, hi, tw); }
    }
    #[allow(unreachable_code)]
    butterfly_f64_scalar(lo, hi, tw);
}

/// Apply the scaled Cooley–Tukey butterfly to `lo[1..]`, `hi[1..]`, `tw[1..]`.
///
/// Same as `butterfly_f64` but each output element is multiplied by `scale`
/// (used to fuse the 1/N normalization into the final inverse stage).
#[inline]
fn butterfly_f64_scaled(lo: &mut [Complex64], hi: &mut [Complex64], tw: &[Complex64], scale: f64) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx",
        target_feature = "fma"
    ))]
    {
        unsafe { return butterfly_f64_scaled_avx_fma(lo, hi, tw, scale); }
    }
    #[allow(unreachable_code)]
    butterfly_f64_scaled_scalar(lo, hi, tw, scale);
}

/// Apply the Cooley–Tukey butterfly to `lo`, `hi`, `tw` (all same length).
///
/// Dispatches to AVX+FMA for f32 when the build target supports it.
#[inline]
fn butterfly_f32(lo: &mut [Complex32], hi: &mut [Complex32], tw: &[Complex32]) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx",
        target_feature = "fma"
    ))]
    {
        unsafe { return butterfly_f32_avx_fma(lo, hi, tw); }
    }
    #[allow(unreachable_code)]
    butterfly_f32_scalar(lo, hi, tw);
}

/// Scaled f32 butterfly (fuse 1/N into final inverse stage).
#[inline]
fn butterfly_f32_scaled(lo: &mut [Complex32], hi: &mut [Complex32], tw: &[Complex32], scale: f32) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx",
        target_feature = "fma"
    ))]
    {
        unsafe { return butterfly_f32_scaled_avx_fma(lo, hi, tw, scale); }
    }
    #[allow(unreachable_code)]
    butterfly_f32_scaled_scalar(lo, hi, tw, scale);
}

// ── scalar fall-backs ─────────────────────────────────────────────────────────

#[inline]
fn butterfly_f64_scalar(lo: &mut [Complex64], hi: &mut [Complex64], tw: &[Complex64]) {
    for k in 0..lo.len() {
        let u = lo[k];
        let v = hi[k];
        let w = tw[k];
        let tr = w.re * v.re - w.im * v.im;
        let ti = w.re * v.im + w.im * v.re;
        lo[k] = Complex64::new(u.re + tr, u.im + ti);
        hi[k] = Complex64::new(u.re - tr, u.im - ti);
    }
}

#[inline]
fn butterfly_f64_scaled_scalar(
    lo: &mut [Complex64],
    hi: &mut [Complex64],
    tw: &[Complex64],
    scale: f64,
) {
    for k in 0..lo.len() {
        let u = lo[k];
        let v = hi[k];
        let w = tw[k];
        let tr = w.re * v.re - w.im * v.im;
        let ti = w.re * v.im + w.im * v.re;
        lo[k] = Complex64::new((u.re + tr) * scale, (u.im + ti) * scale);
        hi[k] = Complex64::new((u.re - tr) * scale, (u.im - ti) * scale);
    }
}

#[inline]
fn butterfly_f32_scalar(lo: &mut [Complex32], hi: &mut [Complex32], tw: &[Complex32]) {
    for k in 0..lo.len() {
        let u = lo[k];
        let v = hi[k];
        let w = tw[k];
        let tr = w.re * v.re - w.im * v.im;
        let ti = w.re * v.im + w.im * v.re;
        lo[k] = Complex32::new(u.re + tr, u.im + ti);
        hi[k] = Complex32::new(u.re - tr, u.im - ti);
    }
}

#[inline]
fn butterfly_f32_scaled_scalar(
    lo: &mut [Complex32],
    hi: &mut [Complex32],
    tw: &[Complex32],
    scale: f32,
) {
    for k in 0..lo.len() {
        let u = lo[k];
        let v = hi[k];
        let w = tw[k];
        let tr = w.re * v.re - w.im * v.im;
        let ti = w.re * v.im + w.im * v.re;
        lo[k] = Complex32::new((u.re + tr) * scale, (u.im + ti) * scale);
        hi[k] = Complex32::new((u.re - tr) * scale, (u.im - ti) * scale);
    }
}

// ── AVX + FMA implementations ─────────────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[target_feature(enable = "avx,fma")]
/// # Safety
/// Caller must guarantee AVX and FMA are available (enforced by cfg guard).
unsafe fn butterfly_f64_avx_fma(lo: &mut [Complex64], hi: &mut [Complex64], tw: &[Complex64]) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd,
        _mm256_storeu_pd, _mm256_sub_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    };
    debug_assert_eq!(lo.len(), hi.len());
    debug_assert_eq!(lo.len(), tw.len());
    let count = lo.len();
    let lo_f = lo.as_mut_ptr() as *mut f64;
    let hi_f = hi.as_mut_ptr() as *mut f64;
    let tw_f = tw.as_ptr() as *const f64;
    let batches = count / 2;
    for b in 0..batches {
        let f = b * 4; // f64 offset: 2 Complex64 = 4 f64
        let u = _mm256_loadu_pd(lo_f.add(f));
        let v = _mm256_loadu_pd(hi_f.add(f));
        let w = _mm256_loadu_pd(tw_f.add(f));
        // Broadcast re and im parts of each twiddle:
        // unpacklo_pd([a,b,c,d],[a,b,c,d]) = [a,a,c,c]  → re parts
        // unpackhi_pd([a,b,c,d],[a,b,c,d]) = [b,b,d,d]  → im parts
        let ac = _mm256_unpacklo_pd(w, w);
        let bd = _mm256_unpackhi_pd(w, w);
        // Swap re/im in each Complex64: permute_pd(imm=5=0b0101) swaps pairs
        // [v.re, v.im, v.re, v.im] → [v.im, v.re, v.im, v.re]
        let v_perm = _mm256_permute_pd(v, 5);
        // fmaddsub(a,b,c) = [a[0]*b[0]-c[0], a[1]*b[1]+c[1], …]
        // = [ac*v.re - bd*v.im, ac*v.im + bd*v.re, …] = [tw.re, tw.im, …]
        let tw_vec = _mm256_fmaddsub_pd(ac, v, _mm256_mul_pd(bd, v_perm));
        _mm256_storeu_pd(lo_f.add(f), _mm256_add_pd(u, tw_vec));
        _mm256_storeu_pd(hi_f.add(f), _mm256_sub_pd(u, tw_vec));
    }
    // Scalar tail when count is odd (e.g. half=17 → 1 tail element).
    let tail = batches * 2;
    butterfly_f64_scalar(&mut lo[tail..], &mut hi[tail..], &tw[tail..]);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[target_feature(enable = "avx,fma")]
/// # Safety
/// Caller must guarantee AVX and FMA are available (enforced by cfg guard).
unsafe fn butterfly_f64_scaled_avx_fma(
    lo: &mut [Complex64],
    hi: &mut [Complex64],
    tw: &[Complex64],
    scale: f64,
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_set1_pd,
        _mm256_storeu_pd, _mm256_sub_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    };
    // _mm256_fmadd_pd computes a*b+c; here we want (u±tw)*scale which is
    // _mm256_mul_pd(_mm256_add_pd(u, tw_vec), scale_v).
    // Using VFMADD: scale*(u+tw) = fmadd(tw, scale, u*scale) — requires u*scale first.
    // Simpler: load scale broadcast, multiply after add/sub.
    let scale_v = _mm256_set1_pd(scale);
    let count = lo.len();
    let lo_f = lo.as_mut_ptr() as *mut f64;
    let hi_f = hi.as_mut_ptr() as *mut f64;
    let tw_f = tw.as_ptr() as *const f64;
    let batches = count / 2;
    for b in 0..batches {
        let f = b * 4;
        let u = _mm256_loadu_pd(lo_f.add(f));
        let v = _mm256_loadu_pd(hi_f.add(f));
        let w = _mm256_loadu_pd(tw_f.add(f));
        let ac = _mm256_unpacklo_pd(w, w);
        let bd = _mm256_unpackhi_pd(w, w);
        let v_perm = _mm256_permute_pd(v, 5);
        let tw_vec = _mm256_fmaddsub_pd(ac, v, _mm256_mul_pd(bd, v_perm));
        // Fuse scale: (u + tw) * scale and (u - tw) * scale
        // Using FMA: fmadd(tw_vec, scale_v, u*scale_v) = (u+tw)*scale → but needs u*scale
        // Simplest: mul(add(u,tw), scale) — 2 instructions per output
        use std::arch::x86_64::_mm256_add_pd;
        let sum = _mm256_add_pd(u, tw_vec);
        let dif = _mm256_sub_pd(u, tw_vec);
        _mm256_storeu_pd(lo_f.add(f), _mm256_mul_pd(sum, scale_v));
        _mm256_storeu_pd(hi_f.add(f), _mm256_mul_pd(dif, scale_v));
    }
    let tail = batches * 2;
    butterfly_f64_scaled_scalar(&mut lo[tail..], &mut hi[tail..], &tw[tail..], scale);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[target_feature(enable = "avx,fma")]
/// # Safety
/// Caller must guarantee AVX and FMA are available (enforced by cfg guard).
unsafe fn butterfly_f32_avx_fma(lo: &mut [Complex32], hi: &mut [Complex32], tw: &[Complex32]) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_fmaddsub_ps, _mm256_loadu_ps, _mm256_movehdup_ps,
        _mm256_moveldup_ps, _mm256_mul_ps, _mm256_permute_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };
    debug_assert_eq!(lo.len(), hi.len());
    debug_assert_eq!(lo.len(), tw.len());
    let count = lo.len();
    let lo_f = lo.as_mut_ptr() as *mut f32;
    let hi_f = hi.as_mut_ptr() as *mut f32;
    let tw_f = tw.as_ptr() as *const f32;
    let batches = count / 4; // 4 Complex32 = 8 f32 per 256-bit register
    for b in 0..batches {
        let f = b * 8; // f32 offset
        let u = _mm256_loadu_ps(lo_f.add(f));
        let v = _mm256_loadu_ps(hi_f.add(f));
        let w = _mm256_loadu_ps(tw_f.add(f));
        // moveldup duplicates even floats (re parts): [w0.re,w0.re,w1.re,w1.re,…]
        let w_re = _mm256_moveldup_ps(w);
        // movehdup duplicates odd floats (im parts): [w0.im,w0.im,w1.im,w1.im,…]
        let w_im = _mm256_movehdup_ps(w);
        // Swap re/im in each Complex32 pair: 0xB1 = 0b10110001 swaps adjacent f32 pairs
        // [v0.re, v0.im, v1.re, v1.im, …] → [v0.im, v0.re, v1.im, v1.re, …]
        let v_swap = _mm256_permute_ps(v, 0xB1);
        // fmaddsub(a,b,c)[2k]   = a[2k]*b[2k] - c[2k]   = w.re*v.re - w.im*v.im = tw.re
        // fmaddsub(a,b,c)[2k+1] = a[2k+1]*b[2k+1]+c[2k+1] = w.re*v.im + w.im*v.re = tw.im
        let tw_vec = _mm256_fmaddsub_ps(w_re, v, _mm256_mul_ps(w_im, v_swap));
        _mm256_storeu_ps(lo_f.add(f), _mm256_add_ps(u, tw_vec));
        _mm256_storeu_ps(hi_f.add(f), _mm256_sub_ps(u, tw_vec));
    }
    let tail = batches * 4;
    butterfly_f32_scalar(&mut lo[tail..], &mut hi[tail..], &tw[tail..]);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[target_feature(enable = "avx,fma")]
/// # Safety
/// Caller must guarantee AVX and FMA are available (enforced by cfg guard).
unsafe fn butterfly_f32_scaled_avx_fma(
    lo: &mut [Complex32],
    hi: &mut [Complex32],
    tw: &[Complex32],
    scale: f32,
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_fmaddsub_ps, _mm256_loadu_ps, _mm256_movehdup_ps,
        _mm256_moveldup_ps, _mm256_mul_ps, _mm256_permute_ps, _mm256_set1_ps, _mm256_storeu_ps,
        _mm256_sub_ps,
    };
    let scale_v = _mm256_set1_ps(scale);
    let count = lo.len();
    let lo_f = lo.as_mut_ptr() as *mut f32;
    let hi_f = hi.as_mut_ptr() as *mut f32;
    let tw_f = tw.as_ptr() as *const f32;
    let batches = count / 4;
    for b in 0..batches {
        let f = b * 8;
        let u = _mm256_loadu_ps(lo_f.add(f));
        let v = _mm256_loadu_ps(hi_f.add(f));
        let w = _mm256_loadu_ps(tw_f.add(f));
        let w_re = _mm256_moveldup_ps(w);
        let w_im = _mm256_movehdup_ps(w);
        let v_swap = _mm256_permute_ps(v, 0xB1);
        let tw_vec = _mm256_fmaddsub_ps(w_re, v, _mm256_mul_ps(w_im, v_swap));
        _mm256_storeu_ps(lo_f.add(f), _mm256_mul_ps(_mm256_add_ps(u, tw_vec), scale_v));
        _mm256_storeu_ps(hi_f.add(f), _mm256_mul_ps(_mm256_sub_ps(u, tw_vec), scale_v));
    }
    let tail = batches * 4;
    butterfly_f32_scaled_scalar(&mut lo[tail..], &mut hi[tail..], &tw[tail..], scale);
}

// ── twiddle table helpers ─────────────────────────────────────────────────────

/// Build a contiguous per-stage forward twiddle table for an N-point DFT.
///
/// # Layout
///
/// The table stores twiddles for log₂N butterfly stages in order from stage 1
/// (len=2) through stage K (len=N). Stage s with group length `len = 2^s` requires
/// `half = len/2` entries `W_{len}^j = exp(-2πi·j/len)` for j = 0…half-1.
///
/// These are laid out contiguously: `[stage1_entries | stage2_entries | … | stageK_entries]`.
/// Stage s has `2^(s-1)` entries. Total length = Σ 2^(s-1) for s=1..K = N-1.
///
/// # Cache benefit
///
/// The original strided-table approach reads `T_fwd[j * stride]` where
/// `stride = N / len = N / 2^s`. For stage 1 (len=2, stride=N/2), each of the N/2
/// independent butterfly groups reads a different element strided by N/2, causing
/// L1/L2 cache misses for N ≥ 256. The contiguous layout ensures each stage reads
/// a sequential sub-slice, maximising cache-line utilization.
///
/// # Correctness
///
/// `W_{len}^j = exp(-2πi·j/len)`. With `len = 2^s` and `j < 2^(s-1)`, the entry at
/// position `base + j` (where `base = 2^(s-1) - 1`) is `exp(-2πi·j / 2^s)`.
pub fn build_forward_twiddle_table_64(n: usize) -> Vec<Complex64> {
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
            let a = -std::f64::consts::TAU * j as f64 / len as f64;
            table.push(Complex64::new(a.cos(), a.sin()));
        }
        len <<= 1;
    }
    table
}

/// Build a contiguous per-stage inverse twiddle table (positive exponent sign).
pub fn build_inverse_twiddle_table_64(n: usize) -> Vec<Complex64> {
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
            let a = std::f64::consts::TAU * j as f64 / len as f64;
            table.push(Complex64::new(a.cos(), a.sin()));
        }
        len <<= 1;
    }
    table
}

/// Build a contiguous per-stage forward twiddle table for f32.
///
/// Twiddles computed in f64 for accuracy, then cast to f32.
pub fn build_forward_twiddle_table_32(n: usize) -> Vec<Complex32> {
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
            let a = -std::f64::consts::TAU * j as f64 / len as f64;
            table.push(Complex32::new(a.cos() as f32, a.sin() as f32));
        }
        len <<= 1;
    }
    table
}

/// Build post-processing twiddle table for real-input forward FFT.
///
/// Returns N/2 + 1 entries: `post[k] = exp(-2πi·k/N)` for k = 0..=N/2.
///
/// These are the standard DFT twiddle factors W_N^k. During the unpack step of
/// `forward_real_inplace_64`, each complex bin `X[k]` for k = 1..N/2-1 requires
/// multiplication by W_N^k to separate the even and odd sub-DFTs. The DC (k=0)
/// and Nyquist (k=N/2) bins use simple real sums and require no multiplication.
pub fn build_real_fwd_post_twiddles_64(n: usize) -> Vec<Complex64> {
    debug_assert!(n.is_power_of_two() && n >= 4);
    let m = n >> 1;
    (0..=m)
        .map(|k| {
            let a = -std::f64::consts::TAU * k as f64 / n as f64;
            Complex64::new(a.cos(), a.sin())
        })
        .collect()
}

/// Real-input forward FFT via half-length complex packing.
///
/// # Algorithm
///
/// Theorem (split-radix real DFT): For real x[n] of length N = 2M, define
/// the complex signal z[k] = x[2k] + i·x[2k+1] for k = 0..M-1. Let
/// Z = FFT_M(z), A[l] = DFT of even sub-sequence, B[l] = DFT of odd sub-sequence.
/// Then Z[l] = A[l] + i·B[l] and Z[M-l]* = A[l] - i·B[l], giving
/// A[l] = (Z[l] + Z[M-l]*)/2 and B[l] = -i·(Z[l] - Z[M-l]*)/2.
/// The N-point DFT is X[l] = A[l] + W_N^l·B[l] for l = 0..N-1, where W_N^l = exp(-2πi·l/N).
/// Together: X[l] = (Z[l] + Z[M-l]*)/2 - i·W_N^l·(Z[l] - Z[M-l]*)/2.
/// Conjugate symmetry X[N-l] = X[l]* (since x is real) yields all N bins.
///
/// # Complexity
///
/// O((N/2)·log₂(N/2)) arithmetic operations for the inner FFT plus O(N) for
/// packing and unpacking — approximately half the work of a full N-point complex FFT.
///
/// # Twiddle reuse
///
/// The N/2-point FFT uses `fft_twiddles[0..N/2-1]`. By the contiguous per-stage
/// layout invariant, stages 1..log₂(N/2) of the N-point table occupy exactly
/// the first N/2-1 entries. The post-processing twiddles W_N^k are distinct from
/// the FFT twiddles and are supplied separately as `post_twiddles`.
///
/// # In-place unpack correctness
///
/// After the inner FFT, `output[0..M]` holds Z[0..M-1]. The unpack writes
/// X[l] to `output[l]` and X[N-l] = X[l]* to `output[N-l]`. For each pair
/// (l, M-l) with l < M/2, both Z[l] and Z[M-l] are read before either is
/// overwritten — since `output[M-l]` (index > M/2) has not been touched by
/// any earlier pair (which only wrote to indices ≤ pair_l < M/2).
///
/// # Preconditions
///
/// - `n = input.len() = output.len()`, a power of two, ≥ 4.
/// - `fft_twiddles.len()` ≥ n/2 - 1.
/// - `post_twiddles.len()` = n/2 + 1.
pub fn forward_real_inplace_64(
    input: &[f64],
    output: &mut [Complex64],
    fft_twiddles: &[Complex64],
    post_twiddles: &[Complex64],
) {
    let n = input.len();
    debug_assert!(
        n.is_power_of_two() && n >= 4,
        "real FFT requires PoT length >= 4"
    );
    let m = n >> 1;
    debug_assert_eq!(output.len(), n);
    debug_assert!(fft_twiddles.len() >= m - 1);
    debug_assert_eq!(post_twiddles.len(), m + 1);

    // Pack: z[k] = x[2k] + i·x[2k+1] into output[0..m]
    for k in 0..m {
        output[k] = Complex64::new(input[2 * k], input[2 * k + 1]);
    }

    // N/2-point forward FFT using the first m-1 twiddle entries.
    // Correct by the contiguous-layout invariant: stages 1..log₂M occupy
    // positions 0..M-1 of the N-point forward twiddle table.
    forward_inplace_64_with_twiddles(&mut output[..m], &fft_twiddles[..m - 1]);

    // In-place unpack. Save Z[0] before overwriting output[0].
    let z0 = output[0];

    // Process symmetric pairs l, m-l for l = 1..ceil(m/2).
    // For l < m/2: Z[l] is at output[l] and Z[m-l] is at output[m-l] (m-l > m/2,
    // so not yet written), avoiding read-after-write aliasing within the loop.
    //
    // Twiddle symmetry: post_twiddles[m-l] = exp(-2πi·(N/2-l)/N)
    //   = exp(-πi)·exp(2πi·l/N) = -conj(post_twiddles[l]).
    // One twiddle read per pair halves post-twiddle cache pressure for large N
    // (N=65536 saves 256 KB of twiddle reads in this loop alone).
    let pair_end = (m + 1) / 2;
    for l in 1..pair_end {
        let ml = m - l;
        let zl = output[l]; // Z[l]   — not yet overwritten
        let zml = output[ml]; // Z[m-l] — not yet overwritten (ml ≥ pair_end > l)
        let a = (zl + zml.conj()) * 0.5;
        let b = (zl - zml.conj()) * Complex64::new(0.0, -0.5);
        let a2 = (zml + zl.conj()) * 0.5;
        let b2 = (zml - zl.conj()) * Complex64::new(0.0, -0.5);
        let wl = post_twiddles[l]; // post_twiddles[ml] = -conj(wl) by symmetry above
        let xl = a + wl * b; // X[l]
        let xml = a2 - wl.conj() * b2; // X[m-l] = a2 + (-conj(wl))·b2
        output[l] = xl;
        output[ml] = xml;
        output[n - l] = xl.conj(); // conjugate symmetry: X[N-l] = X[l]*
        output[n - ml] = xml.conj(); // X[N-(m-l)] = X[m-l]*
    }

    // Middle element at l = m/2 (exists iff m is even; for all PoT N ≥ 4, m = N/2 is even).
    // post_twiddles[m/2] = exp(-2πi·(N/4)/N) = exp(-πi/2) = -i.
    // Analytically: a = zmid.re, b = zmid.im (both real); -i·zmid.im = -i·b.
    // xmid = zmid.re + (-i·zmid.im) = zmid.re - i·zmid.im = conj(zmid). No multiply needed.
    if m % 2 == 0 {
        let mid = m / 2;
        let zmid = output[mid];
        output[mid] = zmid.conj(); // X[m/2]  = conj(Z[m/2])
        output[n - mid] = zmid; // X[3m/2] = conj(X[m/2])* = Z[m/2]
    }

    // DC bin: X[0] = Z[0].re + Z[0].im  (W_N^0 = 1, A[0] = Z[0].re, B[0] = Z[0].im)
    output[0] = Complex64::new(z0.re + z0.im, 0.0);
    // Nyquist bin: X[m] = Z[0].re - Z[0].im  (W_N^m = -1)
    output[m] = Complex64::new(z0.re - z0.im, 0.0);
}

/// Inverse real FFT via half-length complex packing (conjugate of the forward trick).
///
/// # Algorithm
///
/// Given a full N-point Hermitian spectrum X (X[N-k] = X[k]* for all k), recover
/// the real signal x of length N. The algorithm inverts `forward_real_inplace_64`:
///
/// **Pre-process** (solve for Z from X using the forward unpack formula):
///
/// From the forward unpack identity, letting a = Z[k] and b = conj(Z[M-k]):
///
///   X[k] = a*(1-i·W_k)/2 + b*(1+i·W_k)/2
///   conj(X[M-k]) = a*(1+i·W_k)/2 + b*(1-i·W_k)/2
///
/// where W_k = post_twiddles[k] = exp(-2πi·k/N) and M = N/2.
///
/// Adding and subtracting:
///   a + b = X[k] + conj(X[M-k])
///   a - b = i·conj(W_k)·(X[k] - conj(X[M-k]))
///
/// Solving: Z[k] = (X[k] + conj(X[M-k]))/2 + i·conj(W_k)·(X[k] - conj(X[M-k]))/2
///
/// Note: i·conj(W_k) = i·(W_k.re - i·W_k.im) = (W_k.im, W_k.re) as Complex64.
///
/// **Special cases:**
/// - k = 0: W_0 = 1, X[0] and X[M] are real, so Z[0] = (X[0]+X[M])/2 + i·(X[0]-X[M])/2.
///
/// **M-point normalized IFFT:** `inverse_inplace_64_with_twiddles` on scratch[0..M]
/// divides by M, giving z[k] = x[2k] + i·x[2k+1].
///
/// **Unpack:** x[2k] = z[k].re, x[2k+1] = z[k].im.
///
/// # Twiddle reuse
///
/// `fft_twiddles` is the N-point inverse table. Stages 1..log₂M occupy
/// positions 0..M-1, identical to the M-point inverse table.
/// `post_twiddles` is the same table as `build_real_fwd_post_twiddles_64(N)`,
/// reused from the plan's forward field.
///
/// # Complexity
///
/// O((N/2)·log₂(N/2)) for the inner IFFT plus O(N) pre-process and unpack.
/// Eliminates the N-point complex IFFT and the per-call N-element allocation of
/// the naive path.
///
/// # Memory
///
/// Requires a caller-supplied scratch of M = N/2 Complex64 entries (plan-owned
/// via `real_inv_scratch`).
///
/// # Preconditions
///
/// - `input.len() = output.len() = N`, power of two, ≥ 4.
/// - `scratch.len()` = N/2.
/// - `fft_twiddles.len()` ≥ N/2 - 1.
/// - `post_twiddles.len()` = N/2 + 1.
/// - `input` has Hermitian symmetry: input[N-k] = input[k]* for all k.
pub fn inverse_real_inplace_64(
    input: &[Complex64],
    output: &mut [f64],
    scratch: &mut [Complex64],
    fft_twiddles: &[Complex64],
    post_twiddles: &[Complex64],
) {
    let n = input.len();
    debug_assert!(
        n.is_power_of_two() && n >= 4,
        "iRFFT requires PoT length >= 4"
    );
    let m = n >> 1;
    debug_assert_eq!(output.len(), n);
    debug_assert_eq!(scratch.len(), m);
    debug_assert!(fft_twiddles.len() >= m - 1);
    debug_assert_eq!(post_twiddles.len(), m + 1);

    // k = 0: W_0 = 1, conj(W_0) = 1, i*conj(W_0) = i.
    // X[0] and X[M] are real (Hermitian spectrum at DC and Nyquist).
    scratch[0] = Complex64::new(
        (input[0].re + input[m].re) * 0.5,
        (input[0].re - input[m].re) * 0.5,
    );

    // Process pairs (k, m-k) for k = 1..m/2 with one twiddle read per pair.
    // Twiddle symmetry: post_twiddles[m-k] = -conj(post_twiddles[k]).
    // Let wk = post_twiddles[k]; wmk = -conj(wk).
    //   i·conj(wk)  = (wk.im, +wk.re)  [since i·(wk.re - i·wk.im) = wk.im + i·wk.re]
    //   i·conj(wmk) = i·(-wk)           = (wk.im, -wk.re)
    // Both factors share wk.im and differ only in sign of wk.re.
    // One twiddle read per pair halves post-twiddle bandwidth for large N.
    let half_m = m / 2; // m = N/2 is always even for PoT N ≥ 4
    for k in 1..half_m {
        let mk = m - k;
        let xk = input[k];
        let xmk = input[mk];
        let xmk_conj = xmk.conj();
        let xk_conj = xk.conj();
        let sum_k = xk + xmk_conj;
        let diff_k = xk - xmk_conj;
        let sum_mk = xmk + xk_conj;
        let diff_mk = xmk - xk_conj;
        let wk = post_twiddles[k];
        let i_conj_wk = Complex64::new(wk.im, wk.re);
        let i_conj_wmk = Complex64::new(wk.im, -wk.re);
        scratch[k] = (sum_k + i_conj_wk * diff_k) * 0.5;
        scratch[mk] = (sum_mk + i_conj_wmk * diff_mk) * 0.5;
    }
    // k = half_m (Nyquist, self-paired: m - half_m = half_m).
    // post_twiddles[half_m] = -i; i·conj(-i) = -1.
    // Closed form: scratch[half_m] = (xk + xk.conj() + (-1)·(xk - xk.conj())) * 0.5
    //   = (2·xk.conj()) * 0.5 = xk.conj(). No twiddle read or multiply.
    scratch[half_m] = input[half_m].conj();

    // M-point normalized IFFT on scratch[0..M].
    // `inverse_inplace_64_with_twiddles` applies the 1/M scale, giving z[k] = x[2k]+i*x[2k+1].
    // Correctness: stages 1..log₂M occupy the first M-1 positions of the N-point inverse
    // twiddle table (same contiguous-layout invariant as the forward path).
    inverse_inplace_64_with_twiddles(scratch, &fft_twiddles[..m - 1]);

    // Unpack: z[k] = x[2k] + i*x[2k+1]
    for k in 0..m {
        output[2 * k] = scratch[k].re;
        output[2 * k + 1] = scratch[k].im;
    }
}

/// Build a contiguous per-stage inverse twiddle table for f32.
pub fn build_inverse_twiddle_table_32(n: usize) -> Vec<Complex32> {
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
            let a = std::f64::consts::TAU * j as f64 / len as f64;
            table.push(Complex32::new(a.cos() as f32, a.sin() as f32));
        }
        len <<= 1;
    }
    table
}

// ── private helpers ───────────────────────────────────────────────────────────────────────────

/// O(N) bit-reversal permutation for Complex64 data.
///
/// Uses the iterative XOR / binary-counter-in-reverse technique: maintains the
/// variable `j` such that after iteration `i`, `j = bit_reverse(i, log_n)`.
/// Each bit of `j` is flipped by inspecting the carry from incrementing the
/// reversed index from MSB toward LSB. Amortized cost ≈ 2 operations per
/// element (geometric series), replacing the prior O(N log N) path.
fn bit_reverse_permutation_64(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

/// O(N) bit-reversal permutation for Complex32 data.
fn bit_reverse_permutation_32(data: &mut [Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

// ── public API ────────────────────────────────────────────────────────────────────────────────

/// Iterative radix-2 DIT forward FFT (unnormalized, f64).
///
/// `N` must be a power of 2.
///
/// ## Twiddle accuracy
///
/// Twiddles are evaluated directly from the N/2 unified table (see module doc,
/// Unified Twiddle Table theorem), bounding phase error to O(ε_mach) per stage.
pub fn forward_inplace_64(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    // Use on-stack table for small N to avoid heap allocation; precomputed contiguous
    // layout otherwise. The fallback unified-stride path is preserved for correctness
    // when called outside a plan context.
    let table = build_forward_twiddle_table_64(n);
    forward_inplace_64_with_twiddles(data, &table);
}

/// Forward FFT using a precomputed contiguous per-stage twiddle table.
///
/// ## Twiddle layout
///
/// `twiddles` must be the output of `build_forward_twiddle_table_64(n)`.
/// Stage s (group length `len = 2^s`) reads `half = len/2` entries from
/// `twiddles[base..]` where `base = 2^(s-1) - 1` (= total entries for stages 1..s-1).
/// Sequential access eliminates strided cache misses at all stage boundaries.
///
/// ## Correctness
///
/// Identical to `forward_inplace_64`: both apply bit-reversal then log₂N
/// Cooley-Tukey butterfly stages with the same twiddle values `W_{len}^j`.
/// The layout change is a pure memory-access optimization, not an algorithmic change.
#[inline]
pub fn forward_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    bit_reverse_permutation_64(data);
    // Stage 1 (len=2): W_2^0 = 1+0i for every butterfly — multiply is a
    // no-op. Skip the twiddle access entirely; butterfly reduces to pure
    // add/sub, eliminating N/2 complex multiplications.
    for chunk in data.chunks_exact_mut(2) {
        let u = chunk[0];
        let v = chunk[1];
        chunk[0] = u + v;
        chunk[1] = u - v;
    }
    // Stage 2 (len=4): W_4^1 = exp(-2πi/4) = (0,-1) = -i.
    // Proof: -i·(a+ib) = b-ia → tr=v.im, ti=-v.re. Zero complex multiplications.
    if n >= 4 {
        for chunk in data.chunks_exact_mut(4) {
            let (lo, hi) = chunk.split_at_mut(2);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            let u = lo[1];
            let v = hi[1];
            lo[1] = Complex64::new(u.re + v.im, u.im - v.re);
            hi[1] = Complex64::new(u.re - v.im, u.im + v.re);
        }
    }
    // Stage 3 (len=8): W_8^j for j=0..3. Compile-time constants eliminate twiddle table reads.
    // Proof: W_8=exp(-2πi/8). W_8^1=(C,-C), W_8^2=(0,-1)=-i, W_8^3=(-C,-C), C=1/√2.
    // j=1: tr=C(v.re+v.im), ti=C(v.im-v.re). j=2: zero-multiply. j=3: tr=C(v.im-v.re), ti=-C(v.re+v.im).
    if n >= 8 {
        const C: f64 = std::f64::consts::FRAC_1_SQRT_2;
        for chunk in data.chunks_exact_mut(8) {
            let (lo, hi) = chunk.split_at_mut(4);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            {
                let u = lo[1];
                let v = hi[1];
                let tr = C * (v.re + v.im);
                let ti = C * (v.im - v.re);
                lo[1] = Complex64::new(u.re + tr, u.im + ti);
                hi[1] = Complex64::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[2];
                let v = hi[2];
                lo[2] = Complex64::new(u.re + v.im, u.im - v.re);
                hi[2] = Complex64::new(u.re - v.im, u.im + v.re);
            }
            {
                let u = lo[3];
                let v = hi[3];
                let tr = C * (v.im - v.re);
                let ti = -C * (v.re + v.im);
                lo[3] = Complex64::new(u.re + tr, u.im + ti);
                hi[3] = Complex64::new(u.re - tr, u.im - ti);
            }
        }
    }
    // Stage 4 (len=16): W_16^j for j=0..7. Compile-time constants from 16th roots of unity.
    // Proof: W_16^j = exp(-2πi·j/16). Savings over general twiddle-table path:
    //   j=0: 1+0i (no-op, already handled); j=4: (0,−1)=−i (0 real mults, swap+negate only);
    //   j=2=(C,−C), j=6=(−C,−C) where C=1/√2 (2 real mults each vs 4 general);
    //   j=1=(C8,−S8), j=3=(S8,−C8), j=5=(−S8,−C8), j=7=(−C8,−S8) (4 real mults each).
    // Net: 8 real multiplications per chunk eliminated; for N=16384 (1024 chunks): 8192 mults.
    if n >= 16 {
        const C: f64 = std::f64::consts::FRAC_1_SQRT_2; // cos(π/4) = 1/√2
        const C8: f64 = 0.9238795325112867_f64; // cos(π/8)
        const S8: f64 = 0.3826834323650898_f64; // sin(π/8)
        for chunk in data.chunks_exact_mut(16) {
            let (lo, hi) = chunk.split_at_mut(8);
            // j=0: W_16^0 = 1+0i
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            // j=1: W_16^1 = (C8,−S8) → tr=C8·v.re+S8·v.im, ti=C8·v.im−S8·v.re
            {
                let u = lo[1];
                let v = hi[1];
                let tr = C8 * v.re + S8 * v.im;
                let ti = C8 * v.im - S8 * v.re;
                lo[1] = Complex64::new(u.re + tr, u.im + ti);
                hi[1] = Complex64::new(u.re - tr, u.im - ti);
            }
            // j=2: W_16^2 = (C,−C) → tr=C(v.re+v.im), ti=C(v.im−v.re)
            {
                let u = lo[2];
                let v = hi[2];
                let tr = C * (v.re + v.im);
                let ti = C * (v.im - v.re);
                lo[2] = Complex64::new(u.re + tr, u.im + ti);
                hi[2] = Complex64::new(u.re - tr, u.im - ti);
            }
            // j=3: W_16^3 = (S8,−C8) → tr=S8·v.re+C8·v.im, ti=S8·v.im−C8·v.re
            {
                let u = lo[3];
                let v = hi[3];
                let tr = S8 * v.re + C8 * v.im;
                let ti = S8 * v.im - C8 * v.re;
                lo[3] = Complex64::new(u.re + tr, u.im + ti);
                hi[3] = Complex64::new(u.re - tr, u.im - ti);
            }
            // j=4: W_16^4 = (0,−1) = −i → tr=v.im, ti=−v.re (zero multiplications)
            {
                let u = lo[4];
                let v = hi[4];
                lo[4] = Complex64::new(u.re + v.im, u.im - v.re);
                hi[4] = Complex64::new(u.re - v.im, u.im + v.re);
            }
            // j=5: W_16^5 = (−S8,−C8) → tr=−S8·v.re+C8·v.im, ti=−S8·v.im−C8·v.re
            {
                let u = lo[5];
                let v = hi[5];
                let tr = -S8 * v.re + C8 * v.im;
                let ti = -S8 * v.im - C8 * v.re;
                lo[5] = Complex64::new(u.re + tr, u.im + ti);
                hi[5] = Complex64::new(u.re - tr, u.im - ti);
            }
            // j=6: W_16^6 = (−C,−C) → tr=C(v.im−v.re), ti=−C(v.re+v.im)
            {
                let u = lo[6];
                let v = hi[6];
                let tr = C * (v.im - v.re);
                let ti = -C * (v.re + v.im);
                lo[6] = Complex64::new(u.re + tr, u.im + ti);
                hi[6] = Complex64::new(u.re - tr, u.im - ti);
            }
            // j=7: W_16^7 = (−C8,−S8) → tr=−C8·v.re+S8·v.im, ti=−C8·v.im−S8·v.re
            {
                let u = lo[7];
                let v = hi[7];
                let tr = -C8 * v.re + S8 * v.im;
                let ti = -C8 * v.im - S8 * v.re;
                lo[7] = Complex64::new(u.re + tr, u.im + ti);
                hi[7] = Complex64::new(u.re - tr, u.im - ti);
            }
        }
    }
    // General stages: len = 32, 64, …, n. base=15 (stages 1-4: 1+2+4+8=15 entries consumed).
    // split_at_mut exposes non-aliasing of lo/hi to LLVM, enabling
    // autovectorization of the j-loop across butterfly pairs.
    let mut len = 32usize;
    let mut base = 15usize;
    while len <= n {
        let half = len >> 1;
        let stage_twiddles = &twiddles[base..base + half];
        for chunk in data.chunks_exact_mut(len) {
            let (lo, hi) = chunk.split_at_mut(half);
            // j=0: W_L^0 = 1+0i for every stage — no multiply.
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            butterfly_f64(&mut lo[1..], &mut hi[1..], &stage_twiddles[1..]);
        }
        base += half;
        len <<= 1;
    }
}

/// Inverse FFT using a precomputed contiguous per-stage inverse twiddle table.
///
/// `twiddles` must be the output of `build_inverse_twiddle_table_64(n)`.
/// Result is unnormalized (no 1/N factor).
#[inline]
pub fn inverse_inplace_unnorm_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    bit_reverse_permutation_64(data);
    // Stage 1 (len=2): W_2^0 = 1+0i — no multiply.
    for chunk in data.chunks_exact_mut(2) {
        let u = chunk[0];
        let v = chunk[1];
        chunk[0] = u + v;
        chunk[1] = u - v;
    }
    // Stage 2 (len=4): W_4^1 = exp(+2πi/4) = (0,+1) = +i.
    // Proof: +i·(a+ib) = -b+ia → tr=-v.im, ti=v.re. Zero complex multiplications.
    if n >= 4 {
        for chunk in data.chunks_exact_mut(4) {
            let (lo, hi) = chunk.split_at_mut(2);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            let u = lo[1];
            let v = hi[1];
            lo[1] = Complex64::new(u.re - v.im, u.im + v.re);
            hi[1] = Complex64::new(u.re + v.im, u.im - v.re);
        }
    }
    // Stage 3 (len=8) inverse: W_8^j=exp(+2πi·j/8). Compile-time constants; zero twiddle reads.
    // W_8^1=(C,+C), W_8^2=(0,+1)=+i, W_8^3=(-C,+C), C=1/√2.
    // j=1: tr=C(v.re-v.im), ti=C(v.re+v.im). j=2: zero-multiply. j=3: tr=-C(v.re+v.im), ti=C(v.re-v.im).
    if n >= 8 {
        const C: f64 = std::f64::consts::FRAC_1_SQRT_2;
        for chunk in data.chunks_exact_mut(8) {
            let (lo, hi) = chunk.split_at_mut(4);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            {
                let u = lo[1];
                let v = hi[1];
                let tr = C * (v.re - v.im);
                let ti = C * (v.re + v.im);
                lo[1] = Complex64::new(u.re + tr, u.im + ti);
                hi[1] = Complex64::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[2];
                let v = hi[2];
                lo[2] = Complex64::new(u.re - v.im, u.im + v.re);
                hi[2] = Complex64::new(u.re + v.im, u.im - v.re);
            }
            {
                let u = lo[3];
                let v = hi[3];
                let tr = -C * (v.re + v.im);
                let ti = C * (v.re - v.im);
                lo[3] = Complex64::new(u.re + tr, u.im + ti);
                hi[3] = Complex64::new(u.re - tr, u.im - ti);
            }
        }
    }
    // Stage 4 (len=16): inverse W_16^j = exp(+2πi·j/16) constants.
    // Savings: j=4=(0,+1)=+i (0 mults); j=2=(C,+C), j=6=(−C,+C) (2 mults each);
    // j=1=(C8,+S8), j=3=(S8,+C8), j=5=(−S8,+C8), j=7=(−C8,+S8) (4 mults each).
    // Net: 8 real multiplications per chunk eliminated.
    if n >= 16 {
        const C: f64 = std::f64::consts::FRAC_1_SQRT_2;
        const C8: f64 = 0.9238795325112867_f64; // cos(π/8)
        const S8: f64 = 0.3826834323650898_f64; // sin(π/8)
        for chunk in data.chunks_exact_mut(16) {
            let (lo, hi) = chunk.split_at_mut(8);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            // j=1: (C8,+S8) → tr=C8·v.re−S8·v.im, ti=C8·v.im+S8·v.re
            {
                let u = lo[1];
                let v = hi[1];
                let tr = C8 * v.re - S8 * v.im;
                let ti = C8 * v.im + S8 * v.re;
                lo[1] = Complex64::new(u.re + tr, u.im + ti);
                hi[1] = Complex64::new(u.re - tr, u.im - ti);
            }
            // j=2: (C,+C) → tr=C(v.re−v.im), ti=C(v.re+v.im)
            {
                let u = lo[2];
                let v = hi[2];
                let tr = C * (v.re - v.im);
                let ti = C * (v.re + v.im);
                lo[2] = Complex64::new(u.re + tr, u.im + ti);
                hi[2] = Complex64::new(u.re - tr, u.im - ti);
            }
            // j=3: (S8,+C8) → tr=S8·v.re−C8·v.im, ti=S8·v.im+C8·v.re
            {
                let u = lo[3];
                let v = hi[3];
                let tr = S8 * v.re - C8 * v.im;
                let ti = S8 * v.im + C8 * v.re;
                lo[3] = Complex64::new(u.re + tr, u.im + ti);
                hi[3] = Complex64::new(u.re - tr, u.im - ti);
            }
            // j=4: (0,+1)=+i → tr=−v.im, ti=v.re (zero multiplications)
            {
                let u = lo[4];
                let v = hi[4];
                lo[4] = Complex64::new(u.re - v.im, u.im + v.re);
                hi[4] = Complex64::new(u.re + v.im, u.im - v.re);
            }
            // j=5: (−S8,+C8) → tr=−S8·v.re−C8·v.im, ti=C8·v.re−S8·v.im
            {
                let u = lo[5];
                let v = hi[5];
                let tr = -S8 * v.re - C8 * v.im;
                let ti = C8 * v.re - S8 * v.im;
                lo[5] = Complex64::new(u.re + tr, u.im + ti);
                hi[5] = Complex64::new(u.re - tr, u.im - ti);
            }
            // j=6: (−C,+C) → tr=−C(v.re+v.im), ti=C(v.re−v.im)
            {
                let u = lo[6];
                let v = hi[6];
                let tr = -C * (v.re + v.im);
                let ti = C * (v.re - v.im);
                lo[6] = Complex64::new(u.re + tr, u.im + ti);
                hi[6] = Complex64::new(u.re - tr, u.im - ti);
            }
            // j=7: (−C8,+S8) → tr=−C8·v.re−S8·v.im, ti=S8·v.re−C8·v.im
            {
                let u = lo[7];
                let v = hi[7];
                let tr = -C8 * v.re - S8 * v.im;
                let ti = S8 * v.re - C8 * v.im;
                lo[7] = Complex64::new(u.re + tr, u.im + ti);
                hi[7] = Complex64::new(u.re - tr, u.im - ti);
            }
        }
    }
    // General stages: len = 32, 64, …, n. base=15 (stages 1-4: 1+2+4+8=15 entries consumed).
    let mut len = 32usize;
    let mut base = 15usize;
    while len <= n {
        let half = len >> 1;
        let stage_twiddles = &twiddles[base..base + half];
        for chunk in data.chunks_exact_mut(len) {
            let (lo, hi) = chunk.split_at_mut(half);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            butterfly_f64(&mut lo[1..], &mut hi[1..], &stage_twiddles[1..]);
        }
        base += half;
        len <<= 1;
    }
}

/// Normalized inverse FFT using a precomputed contiguous per-stage twiddle table.
///
/// `twiddles` must be the output of `build_inverse_twiddle_table_64(n)`.
#[inline]
pub fn inverse_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    // Inlined butterfly loop with scale fused into the final stage.
    // Eliminates a separate O(N) normalization pass (one full read+write saved).
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    bit_reverse_permutation_64(data);
    let scale = 1.0 / n as f64;
    if n == 2 {
        // Single stage: fuse scale directly (no twiddle multiply; W_2^0 = 1).
        let u = data[0];
        let v = data[1];
        data[0] = (u + v) * scale;
        data[1] = (u - v) * scale;
        return;
    }
    // Stage 1 (len=2): W_2^0 = 1+0i — no multiply, purely additive.
    for chunk in data.chunks_exact_mut(2) {
        let u = chunk[0];
        let v = chunk[1];
        chunk[0] = u + v;
        chunk[1] = u - v;
    }
    if n == 4 {
        // Stage 2 is the final stage. W_4^1 = +i → tr=-v.im, ti=v.re. Fuse scale.
        // Proof: (0,+1)·(a+ib) = -b+ia. Zero complex multiplications.
        let (lo, hi) = data.split_at_mut(2);
        {
            let u = lo[0];
            let v = hi[0];
            lo[0] = (u + v) * scale;
            hi[0] = (u - v) * scale;
        }
        let u = lo[1];
        let v = hi[1];
        lo[1] = Complex64::new((u.re - v.im) * scale, (u.im + v.re) * scale);
        hi[1] = Complex64::new((u.re + v.im) * scale, (u.im - v.re) * scale);
        return;
    }
    // Stage 2 (len=4) as intermediate stage: W_4^1 = +i → no multiply.
    for chunk in data.chunks_exact_mut(4) {
        let (lo, hi) = chunk.split_at_mut(2);
        {
            let u = lo[0];
            let v = hi[0];
            lo[0] = u + v;
            hi[0] = u - v;
        }
        let u = lo[1];
        let v = hi[1];
        lo[1] = Complex64::new(u.re - v.im, u.im + v.re);
        hi[1] = Complex64::new(u.re + v.im, u.im - v.re);
    }
    if n == 8 {
        // Stage 3 is the final stage. W_8^1=(C,+C), W_8^2=+i, W_8^3=(-C,+C). Fuse 1/N scale.
        // Proof: W_8=exp(+2πi/8). C=1/√2. Compile-time constants; zero twiddle table reads.
        const C: f64 = std::f64::consts::FRAC_1_SQRT_2;
        let (lo, hi) = data.split_at_mut(4);
        {
            let u = lo[0];
            let v = hi[0];
            lo[0] = (u + v) * scale;
            hi[0] = (u - v) * scale;
        }
        {
            let u = lo[1];
            let v = hi[1];
            let tr = C * (v.re - v.im);
            let ti = C * (v.re + v.im);
            lo[1] = Complex64::new((u.re + tr) * scale, (u.im + ti) * scale);
            hi[1] = Complex64::new((u.re - tr) * scale, (u.im - ti) * scale);
        }
        {
            let u = lo[2];
            let v = hi[2];
            lo[2] = Complex64::new((u.re - v.im) * scale, (u.im + v.re) * scale);
            hi[2] = Complex64::new((u.re + v.im) * scale, (u.im - v.re) * scale);
        }
        {
            let u = lo[3];
            let v = hi[3];
            let tr = -C * (v.re + v.im);
            let ti = C * (v.re - v.im);
            lo[3] = Complex64::new((u.re + tr) * scale, (u.im + ti) * scale);
            hi[3] = Complex64::new((u.re - tr) * scale, (u.im - ti) * scale);
        }
        return;
    }
    // Stage 3 (len=8) as intermediate. n > 8 here (n >= 16). W_8^1=(C,+C), W_8^2=+i, W_8^3=(-C,+C).
    {
        const C: f64 = std::f64::consts::FRAC_1_SQRT_2;
        for chunk in data.chunks_exact_mut(8) {
            let (lo, hi) = chunk.split_at_mut(4);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            {
                let u = lo[1];
                let v = hi[1];
                let tr = C * (v.re - v.im);
                let ti = C * (v.re + v.im);
                lo[1] = Complex64::new(u.re + tr, u.im + ti);
                hi[1] = Complex64::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[2];
                let v = hi[2];
                lo[2] = Complex64::new(u.re - v.im, u.im + v.re);
                hi[2] = Complex64::new(u.re + v.im, u.im - v.re);
            }
            {
                let u = lo[3];
                let v = hi[3];
                let tr = -C * (v.re + v.im);
                let ti = C * (v.re - v.im);
                lo[3] = Complex64::new(u.re + tr, u.im + ti);
                hi[3] = Complex64::new(u.re - tr, u.im - ti);
            }
        }
    }
    // Stage 4 (len=16): inverse W_16^j constants (positive exponent), same as unnorm inverse.
    // Guard: only when n >= 32; for n=16 stage 4 is handled by the scale-fused final stage.
    if n >= 32 {
        const C: f64 = std::f64::consts::FRAC_1_SQRT_2;
        const C8: f64 = 0.9238795325112867_f64; // cos(π/8)
        const S8: f64 = 0.3826834323650898_f64; // sin(π/8)
        for chunk in data.chunks_exact_mut(16) {
            let (lo, hi) = chunk.split_at_mut(8);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            {
                let u = lo[1];
                let v = hi[1];
                let tr = C8 * v.re - S8 * v.im;
                let ti = C8 * v.im + S8 * v.re;
                lo[1] = Complex64::new(u.re + tr, u.im + ti);
                hi[1] = Complex64::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[2];
                let v = hi[2];
                let tr = C * (v.re - v.im);
                let ti = C * (v.re + v.im);
                lo[2] = Complex64::new(u.re + tr, u.im + ti);
                hi[2] = Complex64::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[3];
                let v = hi[3];
                let tr = S8 * v.re - C8 * v.im;
                let ti = S8 * v.im + C8 * v.re;
                lo[3] = Complex64::new(u.re + tr, u.im + ti);
                hi[3] = Complex64::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[4];
                let v = hi[4];
                lo[4] = Complex64::new(u.re - v.im, u.im + v.re);
                hi[4] = Complex64::new(u.re + v.im, u.im - v.re);
            }
            {
                let u = lo[5];
                let v = hi[5];
                let tr = -S8 * v.re - C8 * v.im;
                let ti = C8 * v.re - S8 * v.im;
                lo[5] = Complex64::new(u.re + tr, u.im + ti);
                hi[5] = Complex64::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[6];
                let v = hi[6];
                let tr = -C * (v.re + v.im);
                let ti = C * (v.re - v.im);
                lo[6] = Complex64::new(u.re + tr, u.im + ti);
                hi[6] = Complex64::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[7];
                let v = hi[7];
                let tr = -C8 * v.re - S8 * v.im;
                let ti = S8 * v.re - C8 * v.im;
                lo[7] = Complex64::new(u.re + tr, u.im + ti);
                hi[7] = Complex64::new(u.re - tr, u.im - ti);
            }
        }
    }
    // General intermediate stages. base/len start past stage 4 if specialized;
    // for n=16 (stage 4 = final stage with scale fusion), use base=7/len=16.
    let mut len = if n >= 32 { 32usize } else { 16usize };
    let mut base = if n >= 32 { 15usize } else { 7usize };
    while len < n {
        let half = len >> 1;
        let stage_twiddles = &twiddles[base..base + half];
        for chunk in data.chunks_exact_mut(len) {
            let (lo, hi) = chunk.split_at_mut(half);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            butterfly_f64(&mut lo[1..], &mut hi[1..], &stage_twiddles[1..]);
        }
        base += half;
        len <<= 1;
    }
    // Final stage (len=n): fuse 1/N scale. j=0: W_N^0 = 1, no multiply.
    let half = n >> 1;
    let stage_twiddles = &twiddles[base..base + half];
    let (lo, hi) = data.split_at_mut(half);
    {
        let u = lo[0];
        let v = hi[0];
        lo[0] = (u + v) * scale;
        hi[0] = (u - v) * scale;
    }
    butterfly_f64_scaled(&mut lo[1..], &mut hi[1..], &stage_twiddles[1..], scale);
}

/// Forward FFT (f32) using a precomputed contiguous per-stage twiddle table.
///
/// `twiddles` must be the output of `build_forward_twiddle_table_32(n)`.
#[inline]
pub fn forward_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    bit_reverse_permutation_32(data);
    // Stage 1 (len=2): W_2^0 = 1+0i — no multiply.
    for chunk in data.chunks_exact_mut(2) {
        let u = chunk[0];
        let v = chunk[1];
        chunk[0] = u + v;
        chunk[1] = u - v;
    }
    // Stage 2 (len=4): W_4^1 = exp(-2πi/4) = (0,-1) = -i.
    // Proof: -i·(a+ib) = b-ia → tr=v.im, ti=-v.re. Zero complex multiplications.
    if n >= 4 {
        for chunk in data.chunks_exact_mut(4) {
            let (lo, hi) = chunk.split_at_mut(2);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            let u = lo[1];
            let v = hi[1];
            lo[1] = Complex32::new(u.re + v.im, u.im - v.re);
            hi[1] = Complex32::new(u.re - v.im, u.im + v.re);
        }
    }
    // Stage 3 (len=8) f32: W_8^1=(C,-C), W_8^2=-i, W_8^3=(-C,-C), C=1/√2. Compile-time constants.
    if n >= 8 {
        const C: f32 = std::f32::consts::FRAC_1_SQRT_2;
        for chunk in data.chunks_exact_mut(8) {
            let (lo, hi) = chunk.split_at_mut(4);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            {
                let u = lo[1];
                let v = hi[1];
                let tr = C * (v.re + v.im);
                let ti = C * (v.im - v.re);
                lo[1] = Complex32::new(u.re + tr, u.im + ti);
                hi[1] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[2];
                let v = hi[2];
                lo[2] = Complex32::new(u.re + v.im, u.im - v.re);
                hi[2] = Complex32::new(u.re - v.im, u.im + v.re);
            }
            {
                let u = lo[3];
                let v = hi[3];
                let tr = C * (v.im - v.re);
                let ti = -C * (v.re + v.im);
                lo[3] = Complex32::new(u.re + tr, u.im + ti);
                hi[3] = Complex32::new(u.re - tr, u.im - ti);
            }
        }
    }
    // Stage 4 (len=16) f32: W_16^j for j=0..7. Compile-time constants.
    // Same structure as the f64 stage-4 specialization, adapted to f32.
    if n >= 16 {
        const C: f32 = std::f32::consts::FRAC_1_SQRT_2; // cos(π/4)
        const C8: f32 = 0.9238795_f32; // cos(π/8)
        const S8: f32 = 0.38268343_f32; // sin(π/8)
        for chunk in data.chunks_exact_mut(16) {
            let (lo, hi) = chunk.split_at_mut(8);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            {
                let u = lo[1];
                let v = hi[1];
                let tr = C8 * v.re + S8 * v.im;
                let ti = C8 * v.im - S8 * v.re;
                lo[1] = Complex32::new(u.re + tr, u.im + ti);
                hi[1] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[2];
                let v = hi[2];
                let tr = C * (v.re + v.im);
                let ti = C * (v.im - v.re);
                lo[2] = Complex32::new(u.re + tr, u.im + ti);
                hi[2] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[3];
                let v = hi[3];
                let tr = S8 * v.re + C8 * v.im;
                let ti = S8 * v.im - C8 * v.re;
                lo[3] = Complex32::new(u.re + tr, u.im + ti);
                hi[3] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[4];
                let v = hi[4];
                lo[4] = Complex32::new(u.re + v.im, u.im - v.re);
                hi[4] = Complex32::new(u.re - v.im, u.im + v.re);
            }
            {
                let u = lo[5];
                let v = hi[5];
                let tr = -S8 * v.re + C8 * v.im;
                let ti = -S8 * v.im - C8 * v.re;
                lo[5] = Complex32::new(u.re + tr, u.im + ti);
                hi[5] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[6];
                let v = hi[6];
                let tr = C * (v.im - v.re);
                let ti = -C * (v.re + v.im);
                lo[6] = Complex32::new(u.re + tr, u.im + ti);
                hi[6] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[7];
                let v = hi[7];
                let tr = -C8 * v.re + S8 * v.im;
                let ti = -C8 * v.im - S8 * v.re;
                lo[7] = Complex32::new(u.re + tr, u.im + ti);
                hi[7] = Complex32::new(u.re - tr, u.im - ti);
            }
        }
    }
    let mut len = 32usize;
    let mut base = 15usize;
    while len <= n {
        let half = len >> 1;
        let stage_twiddles = &twiddles[base..base + half];
        for chunk in data.chunks_exact_mut(len) {
            let (lo, hi) = chunk.split_at_mut(half);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            butterfly_f32(&mut lo[1..], &mut hi[1..], &stage_twiddles[1..]);
        }
        base += half;
        len <<= 1;
    }
}

/// Inverse FFT (f32, unnormalized) using a precomputed contiguous per-stage twiddle table.
///
/// `twiddles` must be the output of `build_inverse_twiddle_table_32(n)`.
#[inline]
pub fn inverse_inplace_unnorm_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    bit_reverse_permutation_32(data);
    // Stage 1 (len=2): W_2^0 = 1+0i — no multiply.
    for chunk in data.chunks_exact_mut(2) {
        let u = chunk[0];
        let v = chunk[1];
        chunk[0] = u + v;
        chunk[1] = u - v;
    }
    // Stage 2 (len=4): W_4^1 = exp(+2πi/4) = (0,+1) = +i.
    // Proof: +i·(a+ib) = -b+ia → tr=-v.im, ti=v.re. Zero complex multiplications.
    if n >= 4 {
        for chunk in data.chunks_exact_mut(4) {
            let (lo, hi) = chunk.split_at_mut(2);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            let u = lo[1];
            let v = hi[1];
            lo[1] = Complex32::new(u.re - v.im, u.im + v.re);
            hi[1] = Complex32::new(u.re + v.im, u.im - v.re);
        }
    }
    // Stage 3 (len=8) inverse f32: W_8^1=(C,+C), W_8^2=+i, W_8^3=(-C,+C), C=1/√2.
    if n >= 8 {
        const C: f32 = std::f32::consts::FRAC_1_SQRT_2;
        for chunk in data.chunks_exact_mut(8) {
            let (lo, hi) = chunk.split_at_mut(4);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            {
                let u = lo[1];
                let v = hi[1];
                let tr = C * (v.re - v.im);
                let ti = C * (v.re + v.im);
                lo[1] = Complex32::new(u.re + tr, u.im + ti);
                hi[1] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[2];
                let v = hi[2];
                lo[2] = Complex32::new(u.re - v.im, u.im + v.re);
                hi[2] = Complex32::new(u.re + v.im, u.im - v.re);
            }
            {
                let u = lo[3];
                let v = hi[3];
                let tr = -C * (v.re + v.im);
                let ti = C * (v.re - v.im);
                lo[3] = Complex32::new(u.re + tr, u.im + ti);
                hi[3] = Complex32::new(u.re - tr, u.im - ti);
            }
        }
    }
    // Stage 4 (len=16) inverse f32: W_16^j = exp(+2πi·j/16) constants.
    if n >= 16 {
        const C: f32 = std::f32::consts::FRAC_1_SQRT_2;
        const C8: f32 = 0.9238795_f32;
        const S8: f32 = 0.38268343_f32;
        for chunk in data.chunks_exact_mut(16) {
            let (lo, hi) = chunk.split_at_mut(8);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            {
                let u = lo[1];
                let v = hi[1];
                let tr = C8 * v.re - S8 * v.im;
                let ti = C8 * v.im + S8 * v.re;
                lo[1] = Complex32::new(u.re + tr, u.im + ti);
                hi[1] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[2];
                let v = hi[2];
                let tr = C * (v.re - v.im);
                let ti = C * (v.re + v.im);
                lo[2] = Complex32::new(u.re + tr, u.im + ti);
                hi[2] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[3];
                let v = hi[3];
                let tr = S8 * v.re - C8 * v.im;
                let ti = S8 * v.im + C8 * v.re;
                lo[3] = Complex32::new(u.re + tr, u.im + ti);
                hi[3] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[4];
                let v = hi[4];
                lo[4] = Complex32::new(u.re - v.im, u.im + v.re);
                hi[4] = Complex32::new(u.re + v.im, u.im - v.re);
            }
            {
                let u = lo[5];
                let v = hi[5];
                let tr = -S8 * v.re - C8 * v.im;
                let ti = C8 * v.re - S8 * v.im;
                lo[5] = Complex32::new(u.re + tr, u.im + ti);
                hi[5] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[6];
                let v = hi[6];
                let tr = -C * (v.re + v.im);
                let ti = C * (v.re - v.im);
                lo[6] = Complex32::new(u.re + tr, u.im + ti);
                hi[6] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[7];
                let v = hi[7];
                let tr = -C8 * v.re - S8 * v.im;
                let ti = S8 * v.re - C8 * v.im;
                lo[7] = Complex32::new(u.re + tr, u.im + ti);
                hi[7] = Complex32::new(u.re - tr, u.im - ti);
            }
        }
    }
    let mut len = 32usize;
    let mut base = 15usize;
    while len <= n {
        let half = len >> 1;
        let stage_twiddles = &twiddles[base..base + half];
        for chunk in data.chunks_exact_mut(len) {
            let (lo, hi) = chunk.split_at_mut(half);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            butterfly_f32(&mut lo[1..], &mut hi[1..], &stage_twiddles[1..]);
        }
        base += half;
        len <<= 1;
    }
}

/// Normalized inverse FFT using a precomputed contiguous per-stage twiddle table.
///
/// `twiddles` must be the output of `build_inverse_twiddle_table_32(n)`.
#[inline]
pub fn inverse_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    // Inlined butterfly with fused final-stage normalization.
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    bit_reverse_permutation_32(data);
    let scale = 1.0f32 / n as f32;
    if n == 2 {
        let u = data[0];
        let v = data[1];
        data[0] = (u + v) * scale;
        data[1] = (u - v) * scale;
        return;
    }
    for chunk in data.chunks_exact_mut(2) {
        let u = chunk[0];
        let v = chunk[1];
        chunk[0] = u + v;
        chunk[1] = u - v;
    }
    if n == 4 {
        // Stage 2 is the final stage. W_4^1 = +i → tr=-v.im, ti=v.re. Fuse scale.
        // Proof: (0,+1)·(a+ib) = -b+ia. Zero complex multiplications.
        let (lo, hi) = data.split_at_mut(2);
        {
            let u = lo[0];
            let v = hi[0];
            lo[0] = (u + v) * scale;
            hi[0] = (u - v) * scale;
        }
        let u = lo[1];
        let v = hi[1];
        lo[1] = Complex32::new((u.re - v.im) * scale, (u.im + v.re) * scale);
        hi[1] = Complex32::new((u.re + v.im) * scale, (u.im - v.re) * scale);
        return;
    }
    // Stage 2 (len=4) as intermediate stage: W_4^1 = +i → no multiply.
    for chunk in data.chunks_exact_mut(4) {
        let (lo, hi) = chunk.split_at_mut(2);
        {
            let u = lo[0];
            let v = hi[0];
            lo[0] = u + v;
            hi[0] = u - v;
        }
        let u = lo[1];
        let v = hi[1];
        lo[1] = Complex32::new(u.re - v.im, u.im + v.re);
        hi[1] = Complex32::new(u.re + v.im, u.im - v.re);
    }
    if n == 8 {
        // Stage 3 is the final stage. W_8^1=(C,+C), W_8^2=+i, W_8^3=(-C,+C). Fuse 1/N scale.
        const C: f32 = std::f32::consts::FRAC_1_SQRT_2;
        let (lo, hi) = data.split_at_mut(4);
        {
            let u = lo[0];
            let v = hi[0];
            lo[0] = (u + v) * scale;
            hi[0] = (u - v) * scale;
        }
        {
            let u = lo[1];
            let v = hi[1];
            let tr = C * (v.re - v.im);
            let ti = C * (v.re + v.im);
            lo[1] = Complex32::new((u.re + tr) * scale, (u.im + ti) * scale);
            hi[1] = Complex32::new((u.re - tr) * scale, (u.im - ti) * scale);
        }
        {
            let u = lo[2];
            let v = hi[2];
            lo[2] = Complex32::new((u.re - v.im) * scale, (u.im + v.re) * scale);
            hi[2] = Complex32::new((u.re + v.im) * scale, (u.im - v.re) * scale);
        }
        {
            let u = lo[3];
            let v = hi[3];
            let tr = -C * (v.re + v.im);
            let ti = C * (v.re - v.im);
            lo[3] = Complex32::new((u.re + tr) * scale, (u.im + ti) * scale);
            hi[3] = Complex32::new((u.re - tr) * scale, (u.im - ti) * scale);
        }
        return;
    }
    // Stage 3 (len=8) as intermediate. n > 8 here. W_8^1=(C,+C), W_8^2=+i, W_8^3=(-C,+C).
    {
        const C: f32 = std::f32::consts::FRAC_1_SQRT_2;
        for chunk in data.chunks_exact_mut(8) {
            let (lo, hi) = chunk.split_at_mut(4);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            {
                let u = lo[1];
                let v = hi[1];
                let tr = C * (v.re - v.im);
                let ti = C * (v.re + v.im);
                lo[1] = Complex32::new(u.re + tr, u.im + ti);
                hi[1] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[2];
                let v = hi[2];
                lo[2] = Complex32::new(u.re - v.im, u.im + v.re);
                hi[2] = Complex32::new(u.re + v.im, u.im - v.re);
            }
            {
                let u = lo[3];
                let v = hi[3];
                let tr = -C * (v.re + v.im);
                let ti = C * (v.re - v.im);
                lo[3] = Complex32::new(u.re + tr, u.im + ti);
                hi[3] = Complex32::new(u.re - tr, u.im - ti);
            }
        }
    }
    // Stage 4 (len=16) inverse f32 constants.
    // Guard: only when n >= 32; for n=16 this stage is fused with final scaling.
    if n >= 32 {
        const C: f32 = std::f32::consts::FRAC_1_SQRT_2;
        const C8: f32 = 0.9238795_f32;
        const S8: f32 = 0.38268343_f32;
        for chunk in data.chunks_exact_mut(16) {
            let (lo, hi) = chunk.split_at_mut(8);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            {
                let u = lo[1];
                let v = hi[1];
                let tr = C8 * v.re - S8 * v.im;
                let ti = C8 * v.im + S8 * v.re;
                lo[1] = Complex32::new(u.re + tr, u.im + ti);
                hi[1] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[2];
                let v = hi[2];
                let tr = C * (v.re - v.im);
                let ti = C * (v.re + v.im);
                lo[2] = Complex32::new(u.re + tr, u.im + ti);
                hi[2] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[3];
                let v = hi[3];
                let tr = S8 * v.re - C8 * v.im;
                let ti = S8 * v.im + C8 * v.re;
                lo[3] = Complex32::new(u.re + tr, u.im + ti);
                hi[3] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[4];
                let v = hi[4];
                lo[4] = Complex32::new(u.re - v.im, u.im + v.re);
                hi[4] = Complex32::new(u.re + v.im, u.im - v.re);
            }
            {
                let u = lo[5];
                let v = hi[5];
                let tr = -S8 * v.re - C8 * v.im;
                let ti = C8 * v.re - S8 * v.im;
                lo[5] = Complex32::new(u.re + tr, u.im + ti);
                hi[5] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[6];
                let v = hi[6];
                let tr = -C * (v.re + v.im);
                let ti = C * (v.re - v.im);
                lo[6] = Complex32::new(u.re + tr, u.im + ti);
                hi[6] = Complex32::new(u.re - tr, u.im - ti);
            }
            {
                let u = lo[7];
                let v = hi[7];
                let tr = -C8 * v.re - S8 * v.im;
                let ti = S8 * v.re - C8 * v.im;
                lo[7] = Complex32::new(u.re + tr, u.im + ti);
                hi[7] = Complex32::new(u.re - tr, u.im - ti);
            }
        }
    }
    // General intermediate stages. Start after stage 4 when specialized.
    let mut len = if n >= 32 { 32usize } else { 16usize };
    let mut base = if n >= 32 { 15usize } else { 7usize };
    while len < n {
        let half = len >> 1;
        let stage_twiddles = &twiddles[base..base + half];
        for chunk in data.chunks_exact_mut(len) {
            let (lo, hi) = chunk.split_at_mut(half);
            {
                let u = lo[0];
                let v = hi[0];
                lo[0] = u + v;
                hi[0] = u - v;
            }
            butterfly_f32(&mut lo[1..], &mut hi[1..], &stage_twiddles[1..]);
        }
        base += half;
        len <<= 1;
    }
    let half = n >> 1;
    let stage_twiddles = &twiddles[base..base + half];
    let (lo, hi) = data.split_at_mut(half);
    {
        let u = lo[0];
        let v = hi[0];
        lo[0] = (u + v) * scale;
        hi[0] = (u - v) * scale;
    }
    butterfly_f32_scaled(&mut lo[1..], &mut hi[1..], &stage_twiddles[1..], scale);
}

/// Iterative radix-2 DIT inverse FFT (unnormalized, f64).
///
/// Produces `Σ X[k] exp(+2πi·k·n/N)` without the 1/N normalization.
/// `N` must be a power of 2. Delegates to the precomputed-twiddle path for
/// zero per-call heap allocation.
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    let table = build_inverse_twiddle_table_64(n);
    inverse_inplace_unnorm_64_with_twiddles(data, &table);
}

/// Iterative radix-2 DIT inverse FFT (normalized by 1/N, f64).
///
/// `N` must be a power of 2.
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    let table = build_inverse_twiddle_table_64(n);
    // Use fused-final-stage normalization to avoid a separate O(N) scale pass.
    inverse_inplace_64_with_twiddles(data, &table);
}

/// Iterative radix-2 DIT forward FFT (unnormalized, f32).
///
/// Twiddle factors are evaluated in f64 then cast to f32 to minimize phase error.
/// Delegates to the contiguous per-stage twiddle path for cache efficiency.
/// `N` must be a power of 2.
pub fn forward_inplace_32(data: &mut [Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    let table = build_forward_twiddle_table_32(n);
    forward_inplace_32_with_twiddles(data, &table);
}

/// Iterative radix-2 DIT inverse FFT (unnormalized, f32).
///
/// `N` must be a power of 2. Delegates to the contiguous per-stage twiddle path.
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    let table = build_inverse_twiddle_table_32(n);
    inverse_inplace_unnorm_32_with_twiddles(data, &table);
}

/// Iterative radix-2 DIT inverse FFT (normalized by 1/N, f32).
///
/// `N` must be a power of 2.
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    let table = build_inverse_twiddle_table_32(n);
    // Use fused-final-stage normalization to avoid a separate O(N) scale pass.
    inverse_inplace_32_with_twiddles(data, &table);
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_err_64(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0f64, f64::max)
    }

    fn max_abs_err_32(a: &[Complex32], b: &[Complex32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0f32, f32::max)
    }

    /// X[0] = x[0]+x[1] = 3, X[1] = x[0]-x[1] = -1.
    #[test]
    fn two_point_forward_matches_analytical() {
        let mut data = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        forward_inplace_64(&mut data);
        assert!((data[0] - Complex64::new(3.0, 0.0)).norm() < 1e-14);
        assert!((data[1] - Complex64::new(-1.0, 0.0)).norm() < 1e-14);
    }

    /// Normalized inverse of [3,-1] with N=2 → [1, 2].
    #[test]
    fn two_point_inverse_normalized_recovers_input() {
        let mut data = vec![Complex64::new(3.0, 0.0), Complex64::new(-1.0, 0.0)];
        inverse_inplace_64(&mut data);
        assert!((data[0] - Complex64::new(1.0, 0.0)).norm() < 1e-14);
        assert!((data[1] - Complex64::new(2.0, 0.0)).norm() < 1e-14);
    }

    /// Unnormalized inverse should be N times the normalized inverse.
    #[test]
    fn unnorm_inverse_differs_from_norm_by_factor_n() {
        let input = vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(-0.5, 1.0),
            Complex64::new(0.25, -0.25),
            Complex64::new(0.75, 0.0),
        ];
        let n = input.len() as f64;
        let mut unnorm = input.clone();
        inverse_inplace_unnorm_64(&mut unnorm);
        let mut norm = input.clone();
        inverse_inplace_64(&mut norm);
        let err = max_abs_err_64(&unnorm, &norm.iter().map(|x| x * n).collect::<Vec<_>>());
        assert!(err < 1e-13, "unnorm / norm ratio failed: err={err}");
    }

    /// Forward then normalized inverse roundtrip for sizes 2, 4, 8, 16, 32, 1024.
    #[test]
    fn roundtrip_holds_for_power_of_two_sizes() {
        for &n in &[2usize, 4, 8, 16, 32, 1024] {
            let input: Vec<Complex64> = (0..n)
                .map(|k| {
                    let t = k as f64 / n as f64;
                    Complex64::new(
                        (std::f64::consts::TAU * 3.0 * t).sin()
                            + 0.5 * (std::f64::consts::TAU * 7.0 * t).cos(),
                        0.25 * (std::f64::consts::TAU * 2.0 * t).sin(),
                    )
                })
                .collect();
            let mut spectrum = input.clone();
            forward_inplace_64(&mut spectrum);
            inverse_inplace_64(&mut spectrum);
            let err = max_abs_err_64(&spectrum, &input);
            assert!(
                err < 1e-10,
                "roundtrip failed for n={n}: ℓ∞ error={err:.2e}"
            );
        }
    }

    /// N=1 is a no-op.
    #[test]
    fn single_element_is_identity() {
        let mut data = vec![Complex64::new(3.7, -1.2)];
        let original = data[0];
        forward_inplace_64(&mut data);
        assert_eq!(data[0], original);
        inverse_inplace_64(&mut data);
        assert_eq!(data[0], original);
    }

    /// f32 forward matches f64 forward within f32 precision.
    #[test]
    fn forward_32_matches_64_within_f32_precision() {
        let n = 16usize;
        let input64: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.31).sin(), (k as f64 * 0.17).cos()))
            .collect();
        let input32: Vec<Complex32> = input64
            .iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect();

        let mut out64 = input64.clone();
        forward_inplace_64(&mut out64);
        let mut out32 = input32.clone();
        forward_inplace_32(&mut out32);

        let err = max_abs_err_32(
            &out32,
            &out64
                .iter()
                .map(|c| Complex32::new(c.re as f32, c.im as f32))
                .collect::<Vec<_>>(),
        );
        assert!(
            err < 1e-4f32,
            "f32 vs f64 forward error too large: {err:.2e}"
        );
    }

    /// f32 roundtrip.
    #[test]
    fn roundtrip_32_holds_for_n_16() {
        let n = 16usize;
        let input: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new((k as f32 * 0.31).sin(), (k as f32 * 0.17).cos()))
            .collect();
        let mut data = input.clone();
        forward_inplace_32(&mut data);
        inverse_inplace_32(&mut data);
        let err = max_abs_err_32(&data, &input);
        assert!(err < 1e-5f32, "f32 roundtrip error={err:.2e}");
    }
}
