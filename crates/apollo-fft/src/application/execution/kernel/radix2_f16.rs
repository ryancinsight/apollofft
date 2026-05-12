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

use super::twiddle_table::build_twiddle_table;
use half::f16;

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
    #[inline]
    pub fn new(re: f16, im: f16) -> Self {
        Self { re, im }
    }

    /// Return the additive identity `0 + 0i`.
    #[inline]
    pub fn zero() -> Self {
        Self {
            re: f16::ZERO,
            im: f16::ZERO,
        }
    }

    /// Construct from f32 parts, rounding each to f16.
    #[inline]
    pub fn from_f32_pair(re: f32, im: f32) -> Self {
        Self {
            re: f16::from_f32(re),
            im: f16::from_f32(im),
        }
    }

    /// Expand both components to f32 for arithmetic.
    #[inline]
    pub fn to_f32_pair(self) -> (f32, f32) {
        (self.re.to_f32(), self.im.to_f32())
    }
}

// ── Twiddle tables ─────────────────────────────────────────────────────────────

/// Build contiguous per-stage forward twiddle table stored as `Cf16`.
///
/// Delegates to the generic SSOT in `twiddle_table`. Twiddles are computed
/// at f64 precision and rounded to f16 on store.
pub fn build_forward_twiddle_table_f16(n: usize) -> Vec<Cf16> {
    build_twiddle_table(n, -1.0)
}

/// Build contiguous per-stage inverse twiddle table stored as `Cf16`.
///
/// Identical layout to the forward table but with positive exponent sign.
pub fn build_inverse_twiddle_table_f16(n: usize) -> Vec<Cf16> {
    build_twiddle_table(n, 1.0)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::mixed_radix::{
        forward_inplace_64_with_twiddles, forward_inplace_f16, inverse_inplace_f16,
    };
    use crate::application::execution::kernel::radix2::build_forward_twiddle_table_64;
    use num_complex::Complex64;

    /// Reference f64 forward DFT for comparison.
    fn fft64(signal: &[f64]) -> Vec<Complex64> {
        let mut buf: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let table = build_forward_twiddle_table_64(buf.len());
        forward_inplace_64_with_twiddles(&mut buf, Some(&table));
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
        let mut buf: Vec<Cf16> = signal
            .iter()
            .map(|&x| Cf16::from_f32_pair(x as f32, 0.0))
            .collect();

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
        let mut buf: Vec<Cf16> = signal
            .iter()
            .map(|&x| Cf16::from_f32_pair(x, 0.0))
            .collect();
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
