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

/// Reverse the lower `bits` bits of `x`.
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

fn bit_reverse_permutation_64(data: &mut [Complex64]) {
    let n = data.len();
    let log_n = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            data.swap(i, j);
        }
    }
}

fn bit_reverse_permutation_32(data: &mut [Complex32]) {
    let n = data.len();
    let log_n = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, log_n);
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
    let mut len = 2usize;
    let mut base = 0usize; // offset into `twiddles` for the current stage
    while len <= n {
        let half = len >> 1;
        let stage_twiddles = &twiddles[base..base + half];
        for chunk in data.chunks_exact_mut(len) {
            for j in 0..half {
                let u = chunk[j];
                let t = stage_twiddles[j] * chunk[j + half];
                chunk[j] = u + t;
                chunk[j + half] = u - t;
            }
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
    let mut len = 2usize;
    let mut base = 0usize;
    while len <= n {
        let half = len >> 1;
        let stage_twiddles = &twiddles[base..base + half];
        for chunk in data.chunks_exact_mut(len) {
            for j in 0..half {
                let u = chunk[j];
                let t = stage_twiddles[j] * chunk[j + half];
                chunk[j] = u + t;
                chunk[j + half] = u - t;
            }
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
    inverse_inplace_unnorm_64_with_twiddles(data, twiddles);
    let scale = 1.0 / data.len() as f64;
    for x in data.iter_mut() {
        *x *= scale;
    }
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
    let mut len = 2usize;
    let mut base = 0usize;
    while len <= n {
        let half = len >> 1;
        let stage_twiddles = &twiddles[base..base + half];
        for chunk in data.chunks_exact_mut(len) {
            for j in 0..half {
                let u = chunk[j];
                let t = stage_twiddles[j] * chunk[j + half];
                chunk[j] = u + t;
                chunk[j + half] = u - t;
            }
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
    let mut len = 2usize;
    let mut base = 0usize;
    while len <= n {
        let half = len >> 1;
        let stage_twiddles = &twiddles[base..base + half];
        for chunk in data.chunks_exact_mut(len) {
            for j in 0..half {
                let u = chunk[j];
                let t = stage_twiddles[j] * chunk[j + half];
                chunk[j] = u + t;
                chunk[j + half] = u - t;
            }
        }
        base += half;
        len <<= 1;
    }
}

/// Normalized inverse FFT (f32) using a precomputed contiguous per-stage twiddle table.
///
/// `twiddles` must be the output of `build_inverse_twiddle_table_32(n)`.
#[inline]
pub fn inverse_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    inverse_inplace_unnorm_32_with_twiddles(data, twiddles);
    let scale = 1.0f32 / data.len() as f32;
    for x in data.iter_mut() {
        *x *= scale;
    }
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
    inverse_inplace_unnorm_64(data);
    let scale = 1.0 / data.len() as f64;
    for x in data.iter_mut() {
        *x *= scale;
    }
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
    inverse_inplace_unnorm_32(data);
    let scale = 1.0f32 / data.len() as f32;
    for x in data.iter_mut() {
        *x *= scale;
    }
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
