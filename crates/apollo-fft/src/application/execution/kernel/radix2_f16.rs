//! Radix-2 Cooley–Tukey FFT over `Cf16` (f16 complex) working buffers.
//!
//! ## Precision model
//!
//! Storage is `f16` throughout: the working buffer holds `Cf16` (4 bytes/element),
//! twiddle tables are precomputed at f64 precision and rounded to f16. Each
//! butterfly stage converts a batch of `Cf16` values to f32 (4-byte), executes
//! the Cooley–Tukey butterfly in f32 arithmetic, then converts back to f16.
//!
//! Accumulated error per stage: ε_f16/2 ≈ 4.88 × 10⁻⁴ (unit roundoff for f16).
//! For an N-point FFT with log₂N stages the total error is bounded by
//! `γ_N · ‖x‖₁` where `γ_N = N · ε_u · (1 + ε_u)^N / (1 - N · ε_u)`.
//!
//! ## SIMD strategy (x86-64 with AVX + F16C + FMA)
//!
//! Each butterfly batch of 4 pairs (8 `Cf16` values per operand = 128 bits):
//!
//! - `_mm256_cvtph_ps`: expand 8 f16 from a 128-bit lane to 8 f32 in a 256-bit register.
//! - `_mm256_moveldup_ps` / `_mm256_movehdup_ps`: broadcast real / imaginary twiddle lanes.
//! - `_mm256_permute_ps(v, 0xB1)`: swap re/im within each complex pair.
//! - `_mm256_fmaddsub_ps`: fused multiply-add/subtract for complex multiply
//!   `w·v` with alternating subtract (re) / add (im): no explicit sign vector needed.
//! - `_mm256_cvtps_ph`: round 8 f32 back to 8 f16 (nearest-even, IEEE 754 mode 0).
//!
//! Throughput: 4 butterfly pairs per SIMD lane group (128-bit F16 load → 256-bit F32 compute).
//! For large N, stages are additionally split into independent chunk groups and
//! executed with Rayon MIMD (`par_chunks_exact_mut`) while preserving per-stage
//! Cooley–Tukey ordering.
//!
//! ## Memory benefit
//!
//! Working buffer: N × 4 bytes (vs N × 8 for `Complex32`).
//! Twiddle table:  (N − 1) × 4 bytes (vs (N − 1) × 8 for `Complex32`).
//! Cache-fitting threshold doubles vs f32: for N = 8192, the Cf16 working buffer
//! is 32 KiB (vs 64 KiB), fitting in typical L1D caches.

use half::f16;
use rayon::prelude::*;

/// Minimum transform size for enabling stage-level Rayon MIMD chunking.
///
/// Below this threshold, task scheduling overhead can dominate execution time.
const RAYON_THRESHOLD: usize = 1 << 14;

// ── Cf16 ──────────────────────────────────────────────────────────────────────

/// Interleaved complex value stored as two `f16` components.
///
/// `#[repr(C)]` ensures layout `[re: u16, im: u16]` with no padding,
/// enabling safe 128-bit SIMD loads of 4 `Cf16` values (8 f16 = 128 bits).
#[derive(Copy, Clone, Default, Debug, PartialEq)]
#[repr(C)]
pub struct Cf16 {
    /// Real component stored as `f16`.
    pub re: f16,
    /// Imaginary component stored as `f16`.
    pub im: f16,
}

impl Cf16 {
    /// Construct from explicit real and imaginary `f16` parts.
    #[inline(always)]
    pub fn new(re: f16, im: f16) -> Self {
        Self { re, im }
    }

    /// Return the additive identity `0 + 0i`.
    #[inline(always)]
    pub fn zero() -> Self {
        Self { re: f16::ZERO, im: f16::ZERO }
    }

    /// Construct from f32 parts, rounding each to f16.
    #[inline(always)]
    pub fn from_f32_pair(re: f32, im: f32) -> Self {
        Self { re: f16::from_f32(re), im: f16::from_f32(im) }
    }

    /// Expand both components to f32 for arithmetic.
    #[inline(always)]
    pub fn to_f32_pair(self) -> (f32, f32) {
        (self.re.to_f32(), self.im.to_f32())
    }
}

// ── Twiddle tables ─────────────────────────────────────────────────────────────

/// Build contiguous per-stage forward twiddle table stored as `Cf16`.
///
/// Twiddles are computed at f64 precision and rounded to f16 on store.
/// Stage s (group length `len = 2^s`) occupies `half = len/2` entries at
/// base position `2^(s-1) - 1`. Total length = N − 1 entries.
pub fn build_forward_twiddle_table_f16(n: usize) -> Vec<Cf16> {
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
            table.push(Cf16::from_f32_pair(a.cos() as f32, a.sin() as f32));
        }
        len <<= 1;
    }
    table
}

/// Build contiguous per-stage inverse twiddle table stored as `Cf16`.
///
/// Identical layout to the forward table but with positive exponent sign
/// (exp(+2πi·j/len) for each twiddle position j, len).
pub fn build_inverse_twiddle_table_f16(n: usize) -> Vec<Cf16> {
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
            table.push(Cf16::from_f32_pair(a.cos() as f32, a.sin() as f32));
        }
        len <<= 1;
    }
    table
}

// ── Bit-reversal permutation ──────────────────────────────────────────────────

fn bit_reverse_permutation_f16(data: &mut [Cf16]) {
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

// ── Scalar butterfly ──────────────────────────────────────────────────────────

/// Scalar Cooley–Tukey butterfly on `lo[j]` and `hi[j]` with twiddle `tw[j]`.
///
/// Each element: convert f16 → f32, compute u + w·v and u − w·v, convert f32 → f16.
/// The f16 round-trip quantization error per butterfly is bounded by ε_f16/2.
#[inline]
fn butterfly_slice_scalar(lo: &mut [Cf16], hi: &mut [Cf16], tw: &[Cf16]) {
    for ((l, h), w) in lo.iter_mut().zip(hi.iter_mut()).zip(tw.iter()) {
        let (lr, li) = l.to_f32_pair();
        let (hr, hi_im) = h.to_f32_pair();
        let (wr, wi) = w.to_f32_pair();
        // w · h: complex multiply in f32
        let whr = wr * hr - wi * hi_im;
        let whi = wr * hi_im + wi * hr;
        *l = Cf16::from_f32_pair(lr + whr, li + whi);
        *h = Cf16::from_f32_pair(lr - whr, li - whi);
    }
}

#[inline]
fn stage1_chunk_f16(chunk: &mut [Cf16]) {
    let (lr, li) = chunk[0].to_f32_pair();
    let (hr, hi) = chunk[1].to_f32_pair();
    chunk[0] = Cf16::from_f32_pair(lr + hr, li + hi);
    chunk[1] = Cf16::from_f32_pair(lr - hr, li - hi);
}

#[inline]
fn process_stage_chunk_scalar(chunk: &mut [Cf16], stage_tw: &[Cf16]) {
    let half = chunk.len() >> 1;
    let (lo, hi) = chunk.split_at_mut(half);
    // j=0: W_L^0 = 1 (twiddle is unity) — skip the complex multiply.
    {
        let (lr, li) = lo[0].to_f32_pair();
        let (hr, hi_val) = hi[0].to_f32_pair();
        lo[0] = Cf16::from_f32_pair(lr + hr, li + hi_val);
        hi[0] = Cf16::from_f32_pair(lr - hr, li - hi_val);
    }
    // j = 1..half: general butterfly with twiddles stage_tw[1..].
    butterfly_slice_scalar(&mut lo[1..], &mut hi[1..], &stage_tw[1..]);
}

fn run_butterfly_stages_f16_scalar(data: &mut [Cf16], twiddles: &[Cf16]) {
    let n = data.len();
    bit_reverse_permutation_f16(data);

    // Stage 1 (len=2): W_2^0 = 1 for every butterfly — pure add/sub.
    if n >= RAYON_THRESHOLD {
        data.par_chunks_exact_mut(2).for_each(stage1_chunk_f16);
    } else {
        data.chunks_exact_mut(2).for_each(stage1_chunk_f16);
    }

    // General stages: len = 4, 8, 16, …, N.
    // base = 1 after stage 1 (which occupies twiddles[0] = W_2^0 = 1, skipped).
    let mut len = 4usize;
    let mut base = 1usize;
    while len <= n {
        let half = len >> 1;
        let stage_tw = &twiddles[base..base + half];
        let groups = n / len;
        if n >= RAYON_THRESHOLD && groups >= 4 {
            data.par_chunks_exact_mut(len)
                .for_each(|chunk| process_stage_chunk_scalar(chunk, stage_tw));
        } else {
            data.chunks_exact_mut(len)
                .for_each(|chunk| process_stage_chunk_scalar(chunk, stage_tw));
        }
        base += half;
        len <<= 1;
    }
}

// ── AVX + F16C + FMA butterfly (x86_64 only) ─────────────────────────────────

/// Process `lo.len() / 4` butterfly batches using AVX2 F16C FMA SIMD.
///
/// Each batch loads 4 `Cf16` from `lo`, `hi`, and `tw` (4 × 4 bytes = 128 bits),
/// expands to 8 f32 via `_mm256_cvtph_ps`, computes the complex multiply
/// `w·v` using `_mm256_fmaddsub_ps` (fused multiply-addsub on interleaved layout),
/// performs the Cooley–Tukey butterfly, then stores back via `_mm256_cvtps_ph`.
///
/// Remainder elements (len % 4 > 0) fall through to `butterfly_slice_scalar`.
///
/// # Safety
///
/// Caller must ensure the target CPU has `avx`, `f16c`, and `fma` features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c,fma")]
unsafe fn butterfly_slice_avx2(lo: &mut [Cf16], hi: &mut [Cf16], tw: &[Cf16]) {
    use std::arch::x86_64::{
        __m128i, __m256, _mm256_add_ps, _mm256_cvtph_ps, _mm256_cvtps_ph,
        _mm256_fmaddsub_ps, _mm256_moveldup_ps, _mm256_movehdup_ps, _mm256_mul_ps,
        _mm256_permute_ps, _mm256_sub_ps, _mm_loadu_si128, _mm_storeu_si128,
    };

    let batches = lo.len() / 4;
    for b in 0..batches {
        // Load 4 Cf16 (128 bits) per operand and expand to 8 f32 (256 bits).
        let lo_ptr = lo.as_mut_ptr().add(b * 4) as *mut __m128i;
        let hi_ptr = hi.as_mut_ptr().add(b * 4) as *mut __m128i;
        let tw_ptr = tw.as_ptr().add(b * 4) as *const __m128i;

        // u_f32 = [ur0, ui0, ur1, ui1, ur2, ui2, ur3, ui3]
        let u_f32: __m256 = _mm256_cvtph_ps(_mm_loadu_si128(lo_ptr));
        // v_f32 = [vr0, vi0, vr1, vi1, vr2, vi2, vr3, vi3]
        let v_f32: __m256 = _mm256_cvtph_ps(_mm_loadu_si128(hi_ptr));
        // w_f32 = [wr0, wi0, wr1, wi1, wr2, wi2, wr3, wi3]
        let w_f32: __m256 = _mm256_cvtph_ps(_mm_loadu_si128(tw_ptr));

        // Broadcast real and imaginary twiddle components.
        // w_re = [wr0, wr0, wr1, wr1, wr2, wr2, wr3, wr3]
        let w_re = _mm256_moveldup_ps(w_f32);
        // w_im = [wi0, wi0, wi1, wi1, wi2, wi2, wi3, wi3]
        let w_im = _mm256_movehdup_ps(w_f32);
        // v_swap = [vi0, vr0, vi1, vr1, vi2, vr2, vi3, vr3]  (swap re/im in each pair)
        // _MM_SHUFFLE(2,3,0,1) = 0xB1: within each 128-bit lane, permute as [1,0,3,2].
        let v_swap = _mm256_permute_ps(v_f32, 0xB1);

        // Complex multiply w·v using fmaddsub:
        //   result[2k]   = w_re[2k]   * v_f32[2k]   − w_im_mul_vswap[2k]
        //                = wr[k] * vr[k] − wi[k] * vi[k]  = (w·v).re[k]  ✓
        //   result[2k+1] = w_re[2k+1] * v_f32[2k+1] + w_im_mul_vswap[2k+1]
        //                = wr[k] * vi[k] + wi[k] * vr[k]  = (w·v).im[k]  ✓
        let w_im_mul_vswap = _mm256_mul_ps(w_im, v_swap);
        let wv = _mm256_fmaddsub_ps(w_re, v_f32, w_im_mul_vswap);

        // Cooley–Tukey butterfly: u + w·v and u − w·v.
        let sum_f32 = _mm256_add_ps(u_f32, wv);
        let dif_f32 = _mm256_sub_ps(u_f32, wv);

        // Convert f32 → f16 (round-to-nearest-even, mode 0) and store.
        let sum_f16 = _mm256_cvtps_ph(sum_f32, 0);
        let dif_f16 = _mm256_cvtps_ph(dif_f32, 0);
        _mm_storeu_si128(lo_ptr, sum_f16);
        _mm_storeu_si128(hi_ptr, dif_f16);
    }

    // Scalar tail for remainder elements (len % 4 > 0).
    let tail = batches * 4;
    butterfly_slice_scalar(&mut lo[tail..], &mut hi[tail..], &tw[tail..]);
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx,f16c,fma")]
unsafe fn process_stage_chunk_avx2(chunk: &mut [Cf16], stage_tw: &[Cf16]) {
    let half = chunk.len() >> 1;
    let (lo, hi) = chunk.split_at_mut(half);
    // j=0: unity twiddle, no multiply.
    {
        let (lr, li) = lo[0].to_f32_pair();
        let (hr, hi_val) = hi[0].to_f32_pair();
        lo[0] = Cf16::from_f32_pair(lr + hr, li + hi_val);
        hi[0] = Cf16::from_f32_pair(lr - hr, li - hi_val);
    }
    // j=1..half: SIMD butterfly with 4-pair batches, scalar tail.
    butterfly_slice_avx2(&mut lo[1..], &mut hi[1..], &stage_tw[1..]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c,fma")]
unsafe fn run_butterfly_stages_f16_avx2(data: &mut [Cf16], twiddles: &[Cf16]) {
    let n = data.len();
    bit_reverse_permutation_f16(data);

    // Stage 1 (len=2): W_2^0 = 1, scalar (2 elements per chunk, SIMD overhead not worth it).
    if n >= RAYON_THRESHOLD {
        data.par_chunks_exact_mut(2).for_each(stage1_chunk_f16);
    } else {
        data.chunks_exact_mut(2).for_each(stage1_chunk_f16);
    }

    let mut len = 4usize;
    let mut base = 1usize;
    while len <= n {
        let half = len >> 1;
        let stage_tw = &twiddles[base..base + half];
        let groups = n / len;
        if n >= RAYON_THRESHOLD && groups >= 4 {
            data.par_chunks_exact_mut(len).for_each(|chunk| {
                // SAFETY: run_butterfly_stages_f16_avx2 is only entered after runtime
                // feature detection; chunk-local processing preserves non-aliasing.
                unsafe { process_stage_chunk_avx2(chunk, stage_tw) }
            });
        } else {
            data.chunks_exact_mut(len).for_each(|chunk| {
                // SAFETY: same as above; sequential branch keeps semantics identical.
                unsafe { process_stage_chunk_avx2(chunk, stage_tw) }
            });
        }
        base += half;
        len <<= 1;
    }
}

// ── Dispatcher ─────────────────────────────────────────────────────────────────

/// Run all Cooley–Tukey butterfly stages on `data`, dispatching to the AVX +
/// F16C + FMA path when available, falling back to the scalar path otherwise.
///
/// Feature detection runs once per FFT call (O(1)) via a cached CPU flag read.
fn run_butterfly_stages_f16(data: &mut [Cf16], twiddles: &[Cf16]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx")
        && std::is_x86_feature_detected!("f16c")
        && std::is_x86_feature_detected!("fma")
    {
        // SAFETY: feature detection confirms avx, f16c, and fma are available.
        unsafe {
            return run_butterfly_stages_f16_avx2(data, twiddles);
        }
    }
    run_butterfly_stages_f16_scalar(data, twiddles);
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Iterative radix-2 DIT forward FFT on `Cf16` data (unnormalized).
///
/// `N = data.len()` must be a power of two. Modifies `data` in-place.
pub fn forward_inplace_f16(data: &mut [Cf16]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(data.len().is_power_of_two(), "radix-2 f16 requires power-of-2 length");
    let table = build_forward_twiddle_table_f16(data.len());
    run_butterfly_stages_f16(data, &table);
}

/// Forward FFT using a precomputed contiguous per-stage twiddle table.
pub fn forward_inplace_f16_with_twiddles(data: &mut [Cf16], twiddles: &[Cf16]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(data.len().is_power_of_two(), "radix-2 f16 requires power-of-2 length");
    run_butterfly_stages_f16(data, twiddles);
}

/// Iterative radix-2 DIT inverse FFT on `Cf16` data, normalized by 1/N.
pub fn inverse_inplace_f16(data: &mut [Cf16]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 f16 requires power-of-2 length");
    let table = build_inverse_twiddle_table_f16(n);
    run_butterfly_stages_f16(data, &table);
    // 1/N normalization: scalar, O(N). LLVM auto-vectorizes with target-cpu=native.
    let inv_n = 1.0f32 / n as f32;
    for c in data.iter_mut() {
        let (r, i) = c.to_f32_pair();
        *c = Cf16::from_f32_pair(r * inv_n, i * inv_n);
    }
}

/// Iterative radix-2 DIT inverse FFT on `Cf16` data, unnormalized.
pub fn inverse_inplace_unnorm_f16(data: &mut [Cf16]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 f16 requires power-of-2 length");
    let table = build_inverse_twiddle_table_f16(n);
    run_butterfly_stages_f16(data, &table);
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::radix2::{
        build_forward_twiddle_table_64, forward_inplace_64_with_twiddles,
    };
    use num_complex::Complex64;

    /// Reference f64 forward DFT for comparison.
    fn fft64(signal: &[f64]) -> Vec<Complex64> {
        let mut buf: Vec<Complex64> =
            signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let table = build_forward_twiddle_table_64(buf.len());
        forward_inplace_64_with_twiddles(&mut buf, &table);
        buf
    }

    /// Convert Cf16 slice to Complex64 for comparison.
    fn to_c64(v: &[Cf16]) -> Vec<Complex64> {
        v.iter()
            .map(|c| Complex64::new(c.re.to_f32() as f64, c.im.to_f32() as f64))
            .collect()
    }

    /// Forward f16 FFT of a real signal packed as complex (imaginary part = 0).
    fn fft_f16_real(signal: &[f64]) -> Vec<Cf16> {
        let mut buf: Vec<Cf16> = signal
            .iter()
            .map(|&x| Cf16::from_f32_pair(x as f32, 0.0))
            .collect();
        forward_inplace_f16(&mut buf);
        buf
    }

    #[test]
    fn trivial_lengths() {
        // N=0: no-op.
        let mut empty: Vec<Cf16> = Vec::new();
        forward_inplace_f16(&mut empty);

        // N=1: unchanged.
        let mut one = vec![Cf16::from_f32_pair(3.0, 0.0)];
        forward_inplace_f16(&mut one);
        assert!((one[0].re.to_f32() - 3.0f32).abs() < 1e-3);
    }

    #[test]
    fn dc_bin_equals_sum() {
        // Theorem: X[0] = Σ x[n] for any input (W_N^0 = 1 for all N).
        let n = 16;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let expected_dc: f64 = signal.iter().sum();
        let out = fft_f16_real(&signal);
        // f16 has ~3 decimal digits of precision.
        assert!(
            (out[0].re.to_f32() as f64 - expected_dc).abs() < 5e-2,
            "DC bin error: got {}, expected {}",
            out[0].re.to_f32(),
            expected_dc
        );
    }

    #[test]
    fn impulse_flat_spectrum() {
        // Theorem: DFT of δ[0] (impulse at n=0) is the all-ones vector.
        // Proof: X[k] = Σ δ[n] · W_N^{kn} = W_N^0 = 1 for all k.
        let n = 8;
        let mut buf = vec![Cf16::zero(); n];
        buf[0] = Cf16::from_f32_pair(1.0, 0.0);
        forward_inplace_f16(&mut buf);
        for (k, c) in buf.iter().enumerate() {
            assert!(
                (c.re.to_f32() - 1.0f32).abs() < 5e-3,
                "impulse spectrum bin {k} re: got {}, expected 1.0",
                c.re.to_f32()
            );
            assert!(
                c.im.to_f32().abs() < 5e-3,
                "impulse spectrum bin {k} im: got {}, expected 0.0",
                c.im.to_f32()
            );
        }
    }

    #[test]
    fn forward_vs_f64_reference() {
        // Compare f16 FFT output to f64 reference within f16 precision bounds.
        // Tolerance: 5 × ε_f16 × ‖X‖_∞ where ε_f16 ≈ 9.77×10⁻⁴.
        // For N=64, ‖X‖_∞ ≤ N × max|x| = 64, so absolute tolerance ≈ 0.31.
        // We use a relative tolerance of 5e-2 per bin (conservative).
        let n = 64;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.17).sin()).collect();

        let ref64 = fft64(&signal);
        let out_f16 = fft_f16_real(&signal);
        let out64 = to_c64(&out_f16);

        let mut max_rel_err = 0.0f64;
        for k in 0..n {
            let ref_mag = ref64[k].norm();
            if ref_mag > 1e-6 {
                let err = (out64[k] - ref64[k]).norm() / ref_mag;
                max_rel_err = max_rel_err.max(err);
            }
        }
        // Expected: max relative error ≲ log₂(64) × ε_f16 ≈ 6 × 9.77×10⁻⁴ ≈ 5.9×10⁻³.
        // Allow 10× headroom for worst-case alignment and phase accumulation.
        assert!(
            max_rel_err < 6e-2,
            "max relative error vs f64 reference: {max_rel_err:.4e}"
        );
    }

    #[test]
    fn round_trip_within_f16_error_bound() {
        // Theorem: IFFT(FFT(x)) = x up to f16 quantization error accumulated over
        // 2 × log₂N stages. Analytical bound for N=64: ≈ 2 × 6 × ε_u ≈ 5.86×10⁻³.
        let n = 64;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.23 - 1.5).tanh()).collect();
        let mut buf: Vec<Cf16> = signal.iter().map(|&x| Cf16::from_f32_pair(x as f32, 0.0)).collect();

        forward_inplace_f16(&mut buf);
        inverse_inplace_f16(&mut buf);

        let mut max_err = 0.0f64;
        for (orig, rec) in signal.iter().zip(buf.iter()) {
            let err = (rec.re.to_f32() as f64 - orig).abs();
            max_err = max_err.max(err);
        }
        // Use 5e-2 to account for worst-case f16 error accumulation over log₂N stages.
        assert!(max_err < 5e-2, "round-trip max error: {max_err:.4e}");
    }

    #[test]
    fn round_trip_n4_tight_bound() {
        // N=4 has only log₂4 = 2 stages. Analytic bound: 2 × ε_u × N × max|x| ≈ 3.9×10⁻³.
        let signal = [1.0f32, -0.5f32, 0.75f32, -0.25f32];
        let mut buf: Vec<Cf16> = signal.iter().map(|&x| Cf16::from_f32_pair(x, 0.0)).collect();
        forward_inplace_f16(&mut buf);
        inverse_inplace_f16(&mut buf);
        for (&orig, rec) in signal.iter().zip(buf.iter()) {
            assert!(
                (rec.re.to_f32() - orig).abs() < 1e-2,
                "N=4 round-trip error: got {}, expected {}",
                rec.re.to_f32(),
                orig
            );
        }
    }

    #[test]
    fn twiddle_table_length() {
        for log in 1..=8u32 {
            let n = 1usize << log;
            let fwd = build_forward_twiddle_table_f16(n);
            let inv = build_inverse_twiddle_table_f16(n);
            assert_eq!(fwd.len(), n - 1, "forward table len for N={n}");
            assert_eq!(inv.len(), n - 1, "inverse table len for N={n}");
        }
    }
}
