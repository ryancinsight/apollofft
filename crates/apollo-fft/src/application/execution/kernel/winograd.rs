//! Winograd short-DFT kernels for sizes 2, 4, 8, 16, 32, and 64.
//!
//! ## Mathematical foundation
//!
//! Winograd's theorem (1978) states that the number of multiplications required
//! to compute an N-point DFT equals `2N − log₂N − 2` for power-of-two N when
//! additions are "free". The construction decomposes the DFT matrix into a
//! product `A · M · B` where B performs only additions (pre-additions), M
//! applies a small set of multiplications by irrational constants, and A
//! performs final additions (post-additions). For our implementation we use
//! the recursive Cooley-Tukey structure to build each kernel from smaller
//! pieces.
//!
//! ## Implementation strategy
//!
//! Rather than deriving A·M·B factorings analytically for each size (which
//! generates kernel-specific hard-coded coefficient arrays), we use a hybrid:
//!
//! - **DFT-2 and DFT-4**: pure addition/subtraction butterflies; zero
//!   multiplications, zero trig (this is the classical Winograd DFT-2 and
//!   DFT-4 result).
//! - **DFT-8**: implemented as two DFT-4 sub-transforms plus the standard
//!   8-point butterfly with the four special twiddles
//!   `{1, -i, √2/2·(1-i), √2/2·(-1-i)}` expressed as precomputed constants,
//!   reducing multiplications vs the generic DFT matrix approach.
//! - **DFT-16 and DFT-32**: hierarchically decomposed as `R` DFT-4/8 sub-
//!   transforms of length `N/R` followed by twiddle multiplications and a
//!   final DFT-R butterfly stage.  The recursion exploits the fact that the
//!   inner short-DFT kernels require no multiplications, so only the inter-
//!   stage twiddle multiplications count against the Winograd bound.
//! - **DFT-64**: decomposed as 4 DFT-16 sub-transforms plus 64 twiddle
//!   multiplications.
//!
//! All kernels operate directly on a mutable slice segment and return values
//! in normal (output) order.  They are intended to be invoked per-group
//! inside the outer radix-R iterator in `radix8`, `radix16`, `radix32`, and
//! `radix64`.
//!
//! ## Notation
//!
//! `W_N^k = exp(-2πi·k/N)` (forward convention). For inverse transforms the
//! sign flips: `W_N^{-k} = exp(+2πi·k/N)`. All kernels accept an `inverse:
//! bool` flag and negate the imaginary twiddle component accordingly.
//!
//! ## References
//!
//! - Winograd, S. (1978). On computing the discrete Fourier transform.
//!   *Mathematics of Computation*, 32(141), 175-199.
//! - Van Loan, C. (1992). *Computational Frameworks for the Fast Fourier
//!   Transform*. SIAM.
//! - Blahut, R.E. (2010). *Fast Algorithms for Signal Processing*. Cambridge
//!   University Press.

use num_complex::{Complex32, Complex64};

// ── DFT-2 butterfly ───────────────────────────────────────────────────────────

/// In-place Winograd DFT-2 (butterfly).
///
/// **Contract**: `[a, b]` → `[a+b, a-b]` (forward/inverse sign-invariant).
///
/// No multiplications; one complex addition and one complex subtraction.
#[inline(always)]
pub fn dft2_64(a: &mut Complex64, b: &mut Complex64) {
    let t = *a;
    *a = t + *b;
    *b = t - *b;
}

/// In-place Winograd DFT-2 (f32 variant).
#[inline(always)]
pub fn dft2_32(a: &mut Complex32, b: &mut Complex32) {
    let t = *a;
    *a = t + *b;
    *b = t - *b;
}

// ── DFT-4 butterfly ───────────────────────────────────────────────────────────

/// In-place Winograd DFT-4.
///
/// **Contract** (forward, inverse=false):
/// ```text
/// y[0] = x[0] + x[1] + x[2] + x[3]
/// y[1] = x[0] - i·x[1] - x[2] + i·x[3]
/// y[2] = x[0] - x[1] + x[2] - x[3]
/// y[3] = x[0] + i·x[1] - x[2] - i·x[3]
/// ```
/// Inverse: replace `-i` ↔ `+i`.
///
/// **Multiplications**: 0 (all operations are ±1, ±i rotations ≡ swap+negate).
/// **Additions**: 8 complex (= 16 real).
///
/// Correctness reference: Cooley and Tukey (1965), 4-point special case.
#[inline(always)]
pub fn dft4_64(data: &mut [Complex64; 4], inverse: bool) {
    // Stage 1: two DFT-2 butterflies on even and odd indices.
    let (x0, x1, x2, x3) = (data[0], data[1], data[2], data[3]);

    let t0 = x0 + x2; // even, slot 0
    let t1 = x0 - x2; // even, slot 1
    let t2 = x1 + x3; // odd, slot 0
    let t3 = x1 - x3; // odd, slot 1

    // Stage 2: DFT-2 with twiddle W_4^1 = -i (forward) or +i (inverse).
    // Multiply t3 by ±i: (re, im)·(-i) = (im, -re); ·(+i) = (-im, re).
    let t3_tw = if inverse {
        Complex64::new(-t3.im, t3.re)
    } else {
        Complex64::new(t3.im, -t3.re)
    };

    data[0] = t0 + t2;
    data[2] = t0 - t2;
    data[1] = t1 + t3_tw;
    data[3] = t1 - t3_tw;
}

/// In-place Winograd DFT-4 (f32 variant).
#[inline(always)]
pub fn dft4_32(data: &mut [Complex32; 4], inverse: bool) {
    let (x0, x1, x2, x3) = (data[0], data[1], data[2], data[3]);
    let t0 = x0 + x2;
    let t1 = x0 - x2;
    let t2 = x1 + x3;
    let t3 = x1 - x3;
    let t3_tw = if inverse {
        Complex32::new(-t3.im, t3.re)
    } else {
        Complex32::new(t3.im, -t3.re)
    };
    data[0] = t0 + t2;
    data[2] = t0 - t2;
    data[1] = t1 + t3_tw;
    data[3] = t1 - t3_tw;
}

// ── DFT-8 butterfly ───────────────────────────────────────────────────────────

/// In-place Winograd DFT-8.
///
/// **Decomposition**: DFT-8 = stride-2 decimation-in-time into two DFT-4
/// sub-transforms followed by 8-point butterfly twiddle multiplications.
///
/// Twiddle factors for the 8-point stage (forward convention):
/// ```text
/// W_8^0 = 1
/// W_8^1 = (√2/2) - i·(√2/2) = SQ2O2·(1 - i)
/// W_8^2 = -i
/// W_8^3 = -(√2/2) - i·(√2/2)
/// ```
/// where `SQ2O2 = √2/2 ≈ 0.7071067811865476`.
///
/// **Multiplications**: 4 real (the ±SQ2O2 multiplications on the odd path).
/// All other twiddles are ×1 or ×(-i) / ×i, which are free sign/swap ops.
///
/// **Additions**: 26 real (Winograd 1978, Table 1, row N=8).
///
/// Correctness: Blahut (2010), §3.4, DFT-8 factoring.
#[inline(always)]
pub fn dft8_64(data: &mut [Complex64; 8], inverse: bool) {
    // Step 1: two DFT-4s on the stride-2 sub-arrays.
    let mut even = [data[0], data[2], data[4], data[6]];
    let mut odd  = [data[1], data[3], data[5], data[7]];
    dft4_64(&mut even, inverse);
    dft4_64(&mut odd, inverse);

    // Step 2: apply W_8^k twiddles to odd outputs, then butterfly.
    // Forward twiddles: W_8^k = exp(-2πi·k/8) for k = 0..3.
    // Inverse twiddles: conjugate (flip sign of imaginary part).
    // SQ2O2 = √2/2.
    const SQ2O2: f64 = std::f64::consts::FRAC_1_SQRT_2;

    // W_8^0 = 1: no-op.
    let o0 = odd[0];
    // W_8^1 = SQ2O2·(1-i) fwd or SQ2O2·(1+i) inv.
    let o1 = if inverse {
        let re = SQ2O2 * (odd[1].re - odd[1].im);
        let im = SQ2O2 * (odd[1].re + odd[1].im);
        Complex64::new(re, im)
    } else {
        let re = SQ2O2 * (odd[1].re + odd[1].im);
        let im = SQ2O2 * (odd[1].im - odd[1].re);
        Complex64::new(re, im)
    };
    // W_8^2 = -i fwd, +i inv.
    let o2 = if inverse {
        Complex64::new(-odd[2].im, odd[2].re)
    } else {
        Complex64::new(odd[2].im, -odd[2].re)
    };
    // W_8^3 = SQ2O2·(-1-i) fwd or SQ2O2·(-1+i) inv.
    let o3 = if inverse {
        // W_8^{-3} = -SQ2O2 + i*SQ2O2: re = SQ2O2*(-re-im), im = SQ2O2*(re-im)
        let re = SQ2O2 * (-odd[3].re - odd[3].im);
        let im = SQ2O2 * (odd[3].re - odd[3].im);
        Complex64::new(re, im)
    } else {
        // W_8^3 = -SQ2O2 - i*SQ2O2: re = SQ2O2*(-re+im), im = SQ2O2*(-re-im)
        let re = SQ2O2 * (-odd[3].re + odd[3].im);
        let im = SQ2O2 * (-odd[3].re - odd[3].im);
        Complex64::new(re, im)
    };

    // Step 3: combine.
    data[0] = even[0] + o0;
    data[1] = even[1] + o1;
    data[2] = even[2] + o2;
    data[3] = even[3] + o3;
    data[4] = even[0] - o0;
    data[5] = even[1] - o1;
    data[6] = even[2] - o2;
    data[7] = even[3] - o3;
}

/// In-place Winograd DFT-8 (f32 variant).
#[inline(always)]
pub fn dft8_32(data: &mut [Complex32; 8], inverse: bool) {
    const SQ2O2: f32 = std::f32::consts::FRAC_1_SQRT_2;

    let mut even = [data[0], data[2], data[4], data[6]];
    let mut odd  = [data[1], data[3], data[5], data[7]];
    dft4_32(&mut even, inverse);
    dft4_32(&mut odd, inverse);

    let o0 = odd[0];
    let o1 = if inverse {
        let re = SQ2O2 * (odd[1].re - odd[1].im);
        let im = SQ2O2 * (odd[1].re + odd[1].im);
        Complex32::new(re, im)
    } else {
        let re = SQ2O2 * (odd[1].re + odd[1].im);
        let im = SQ2O2 * (odd[1].im - odd[1].re);
        Complex32::new(re, im)
    };
    let o2 = if inverse {
        Complex32::new(-odd[2].im, odd[2].re)
    } else {
        Complex32::new(odd[2].im, -odd[2].re)
    };
    let o3 = if inverse {
        // W_8^{-3} = -SQ2O2 + i*SQ2O2: re = SQ2O2*(-re-im), im = SQ2O2*(re-im)
        let re = SQ2O2 * (-odd[3].re - odd[3].im);
        let im = SQ2O2 * (odd[3].re - odd[3].im);
        Complex32::new(re, im)
    } else {
        // W_8^3 = -SQ2O2 - i*SQ2O2: re = SQ2O2*(-re+im), im = SQ2O2*(-re-im)
        let re = SQ2O2 * (-odd[3].re + odd[3].im);
        let im = SQ2O2 * (-odd[3].re - odd[3].im);
        Complex32::new(re, im)
    };

    data[0] = even[0] + o0;
    data[1] = even[1] + o1;
    data[2] = even[2] + o2;
    data[3] = even[3] + o3;
    data[4] = even[0] - o0;
    data[5] = even[1] - o1;
    data[6] = even[2] - o2;
    data[7] = even[3] - o3;
}

// ── DFT-16 butterfly ─────────────────────────────────────────────────────────

/// Precomputed forward twiddle factors for the 16-point DFT stage.
///
/// `W_16^k = exp(-2πi·k/16)` for `k = 0..7`.
/// Stored as `(cos(2πk/16), -sin(2πk/16))`.
/// Only k=0..7 are needed (k=0 is trivial; k=1..7 are non-trivial).
fn twiddle16_fwd(k: usize) -> Complex64 {
    // Exact values for 16th roots of unity.
    match k {
        0 => Complex64::new(1.0, 0.0),
        1 => {
            // cos(π/8), -sin(π/8)
            let c = (2.0f64 + 2.0f64.sqrt()).sqrt() * 0.5;
            let s = (2.0f64 - 2.0f64.sqrt()).sqrt() * 0.5;
            Complex64::new(c, -s)
        }
        2 => {
            // cos(π/4), -sin(π/4)
            const SQ2O2: f64 = std::f64::consts::FRAC_1_SQRT_2;
            Complex64::new(SQ2O2, -SQ2O2)
        }
        3 => {
            // cos(3π/8), -sin(3π/8)
            let c = (2.0f64 - 2.0f64.sqrt()).sqrt() * 0.5;
            let s = (2.0f64 + 2.0f64.sqrt()).sqrt() * 0.5;
            Complex64::new(c, -s)
        }
        4 => Complex64::new(0.0, -1.0),
        5 => {
            let c = (2.0f64 - 2.0f64.sqrt()).sqrt() * 0.5;
            let s = (2.0f64 + 2.0f64.sqrt()).sqrt() * 0.5;
            Complex64::new(-c, -s)
        }
        6 => {
            const SQ2O2: f64 = std::f64::consts::FRAC_1_SQRT_2;
            Complex64::new(-SQ2O2, -SQ2O2)
        }
        7 => {
            let c = (2.0f64 + 2.0f64.sqrt()).sqrt() * 0.5;
            let s = (2.0f64 - 2.0f64.sqrt()).sqrt() * 0.5;
            Complex64::new(-c, -s)
        }
        _ => unreachable!(),
    }
}

fn twiddle16_inv(k: usize) -> Complex64 {
    let w = twiddle16_fwd(k);
    Complex64::new(w.re, -w.im)
}

fn twiddle16_fwd_32(k: usize) -> Complex32 {
    let w = twiddle16_fwd(k);
    Complex32::new(w.re as f32, w.im as f32)
}

fn twiddle16_inv_32(k: usize) -> Complex32 {
    let w = twiddle16_inv(k);
    Complex32::new(w.re as f32, w.im as f32)
}

/// In-place Winograd DFT-16.
///
/// **Decomposition**: DFT-16 = 2 × DFT-8 (stride-2 DIT) + 16-point twiddle
/// butterfly.
///
/// **Multiplications**: 2 × (DFT-8 ops) + 12 real twiddle mults (k=1..7
/// excluding k=0 trivial, k=4 = ×(-i) free, effectively 10 irrational mults).
///
/// Correctness: Van Loan (1992), §3.3.
#[inline(always)]
pub fn dft16_64(data: &mut [Complex64; 16], inverse: bool) {
    // Step 1: two DFT-8 sub-transforms on even and odd sub-arrays.
    let mut even = [
        data[0], data[2], data[4], data[6],
        data[8], data[10], data[12], data[14],
    ];
    let mut odd = [
        data[1], data[3], data[5], data[7],
        data[9], data[11], data[13], data[15],
    ];
    dft8_64(&mut even, inverse);
    dft8_64(&mut odd, inverse);

    // Step 2: apply W_16^k twiddles to odd outputs and butterfly.
    for k in 0..8 {
        let tw = if inverse { twiddle16_inv(k) } else { twiddle16_fwd(k) };
        let o = Complex64::new(
            odd[k].re * tw.re - odd[k].im * tw.im,
            odd[k].re * tw.im + odd[k].im * tw.re,
        );
        data[k]     = even[k] + o;
        data[k + 8] = even[k] - o;
    }
}

/// In-place Winograd DFT-16 (f32 variant).
#[inline(always)]
pub fn dft16_32(data: &mut [Complex32; 16], inverse: bool) {
    let mut even = [
        data[0], data[2], data[4], data[6],
        data[8], data[10], data[12], data[14],
    ];
    let mut odd = [
        data[1], data[3], data[5], data[7],
        data[9], data[11], data[13], data[15],
    ];
    dft8_32(&mut even, inverse);
    dft8_32(&mut odd, inverse);

    for k in 0..8 {
        let tw = if inverse { twiddle16_inv_32(k) } else { twiddle16_fwd_32(k) };
        let o = Complex32::new(
            odd[k].re * tw.re - odd[k].im * tw.im,
            odd[k].re * tw.im + odd[k].im * tw.re,
        );
        data[k]     = even[k] + o;
        data[k + 8] = even[k] - o;
    }
}

// ── DFT-32 butterfly ─────────────────────────────────────────────────────────

const TWIDDLE32_FWD_64: [Complex64; 16] = [
    Complex64::new(1.0, 0.0),
    Complex64::new(0.9807852804032304, -0.19509032201612825),
    Complex64::new(0.9238795325112867, -0.3826834323650898),
    Complex64::new(0.8314696123025452, -0.5555702330196022),
    Complex64::new(0.7071067811865476, -0.7071067811865475),
    Complex64::new(0.5555702330196023, -0.8314696123025452),
    Complex64::new(0.38268343236508984, -0.9238795325112867),
    Complex64::new(0.19509032201612833, -0.9807852804032304),
    Complex64::new(0.0, -1.0),
    Complex64::new(-0.1950903220161282, -0.9807852804032304),
    Complex64::new(-0.3826834323650897, -0.9238795325112867),
    Complex64::new(-0.555570233019602, -0.8314696123025455),
    Complex64::new(-0.7071067811865475, -0.7071067811865476),
    Complex64::new(-0.8314696123025453, -0.5555702330196022),
    Complex64::new(-0.9238795325112867, -0.3826834323650899),
    Complex64::new(-0.9807852804032304, -0.1950903220161286),
];

#[inline(always)]
fn twiddle32_64(k: usize, inverse: bool) -> Complex64 {
    let w = TWIDDLE32_FWD_64[k];
    if inverse {
        Complex64::new(w.re, -w.im)
    } else {
        w
    }
}

#[inline(always)]
fn twiddle32_32(k: usize, inverse: bool) -> Complex32 {
    let w = twiddle32_64(k, inverse);
    Complex32::new(w.re as f32, w.im as f32)
}

/// In-place Winograd DFT-32.
///
/// **Decomposition**: DFT-32 = 2 × DFT-16 (stride-2 DIT) + 32-point twiddle
/// butterfly.
///
/// **Multiplications**: 2 × (DFT-16 ops) + 28 twiddle mults (k=1..15 minus
/// the trivial/free roots).
///
/// Correctness: Van Loan (1992), §3.3 recursive formulation.
#[inline(always)]
pub fn dft32_64(data: &mut [Complex64; 32], inverse: bool) {
    let mut even = [
        data[0], data[2], data[4], data[6], data[8], data[10],
        data[12], data[14], data[16], data[18], data[20], data[22],
        data[24], data[26], data[28], data[30],
    ];
    let mut odd = [
        data[1], data[3], data[5], data[7], data[9], data[11],
        data[13], data[15], data[17], data[19], data[21], data[23],
        data[25], data[27], data[29], data[31],
    ];
    dft16_64(&mut even, inverse);
    dft16_64(&mut odd, inverse);
    for k in 0..16 {
        let tw = twiddle32_64(k, inverse);
        let o = Complex64::new(
            odd[k].re * tw.re - odd[k].im * tw.im,
            odd[k].re * tw.im + odd[k].im * tw.re,
        );
        data[k]      = even[k] + o;
        data[k + 16] = even[k] - o;
    }
}

/// In-place Winograd DFT-32 (f32 variant).
#[inline(always)]
pub fn dft32_32(data: &mut [Complex32; 32], inverse: bool) {
    let mut even = [
        data[0], data[2], data[4], data[6], data[8], data[10],
        data[12], data[14], data[16], data[18], data[20], data[22],
        data[24], data[26], data[28], data[30],
    ];
    let mut odd = [
        data[1], data[3], data[5], data[7], data[9], data[11],
        data[13], data[15], data[17], data[19], data[21], data[23],
        data[25], data[27], data[29], data[31],
    ];
    dft16_32(&mut even, inverse);
    dft16_32(&mut odd, inverse);
    for k in 0..16 {
        let tw = twiddle32_32(k, inverse);
        let o = Complex32::new(
            odd[k].re * tw.re - odd[k].im * tw.im,
            odd[k].re * tw.im + odd[k].im * tw.re,
        );
        data[k]      = even[k] + o;
        data[k + 16] = even[k] - o;
    }
}

// ── DFT-64 butterfly ─────────────────────────────────────────────────────────

#[inline(always)]
fn twiddle64_64(k: usize, inverse: bool) -> Complex64 {
    // W_64^(2m) = W_32^m and W_64^(2m+1) = W_32^m * W_64^1.
    // This avoids per-call trig evaluation and avoids lock checks in hot loops.
    let base = twiddle32_64(k >> 1, inverse);
    if (k & 1) == 0 {
        base
    } else {
        // cos(pi/32) ± i*sin(pi/32)
        let w1 = if inverse {
            Complex64::new(0.9951847266721969, 0.0980171403295606)
        } else {
            Complex64::new(0.9951847266721969, -0.0980171403295606)
        };
        apply_twiddle_64(base, w1)
    }
}

#[inline(always)]
fn twiddle64_32(k: usize, inverse: bool) -> Complex32 {
    let w = twiddle64_64(k, inverse);
    Complex32::new(w.re as f32, w.im as f32)
}

/// In-place Winograd DFT-64.
///
/// **Decomposition**: DFT-64 = 2 × DFT-32 (stride-2 DIT) + 64-point twiddle
/// butterfly.
///
/// **Multiplications**: 2 × (DFT-32 ops) + 60 twiddle mults (k=1..31 minus
/// the trivial/free roots).
///
/// Correctness: Van Loan (1992), §3.3 recursive formulation.
#[inline(always)]
pub fn dft64_64(data: &mut [Complex64; 64], inverse: bool) {
    let mut even = core::array::from_fn(|i| data[2 * i]);
    let mut odd  = core::array::from_fn(|i| data[2 * i + 1]);
    dft32_64(&mut even, inverse);
    dft32_64(&mut odd, inverse);
    for k in 0..32 {
        let tw = twiddle64_64(k, inverse);
        let o = Complex64::new(
            odd[k].re * tw.re - odd[k].im * tw.im,
            odd[k].re * tw.im + odd[k].im * tw.re,
        );
        data[k]      = even[k] + o;
        data[k + 32] = even[k] - o;
    }
}

/// In-place Winograd DFT-64 (f32 variant).
#[inline(always)]
pub fn dft64_32(data: &mut [Complex32; 64], inverse: bool) {
    let mut even: [Complex32; 32] = core::array::from_fn(|i| data[2 * i]);
    let mut odd:  [Complex32; 32] = core::array::from_fn(|i| data[2 * i + 1]);
    dft32_32(&mut even, inverse);
    dft32_32(&mut odd, inverse);
    for k in 0..32 {
        let tw = twiddle64_32(k, inverse);
        let o = Complex32::new(
            odd[k].re * tw.re - odd[k].im * tw.im,
            odd[k].re * tw.im + odd[k].im * tw.re,
        );
        data[k]      = even[k] + o;
        data[k + 32] = even[k] - o;
    }
}

// ── helpers used by radix kernels ─────────────────────────────────────────────

/// Apply `W_N^{k·j}` twiddle multiplication in-place.
/// Used by the radix outer loop to apply inter-group twiddles.
#[inline(always)]
pub fn apply_twiddle_64(v: Complex64, tw: Complex64) -> Complex64 {
    Complex64::new(
        v.re * tw.re - v.im * tw.im,
        v.re * tw.im + v.im * tw.re,
    )
}

#[inline(always)]
/// Apply inter-group twiddle factor to a single element (f32).
///
/// Computes `v * tw` using the standard complex multiplication identity.
pub fn apply_twiddle_32(v: Complex32, tw: Complex32) -> Complex32 {
    Complex32::new(
        v.re * tw.re - v.im * tw.im,
        v.re * tw.im + v.im * tw.re,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward_64, dft_inverse_64};

    fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).norm()).fold(0.0f64, f64::max)
    }

    // ── DFT-2 ────────────────────────────────────────────────────────────────

    #[test]
    fn dft2_forward_matches_direct() {
        let input = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
        ];
        let expected = dft_forward_64(&input);
        let mut a = input[0];
        let mut b = input[1];
        dft2_64(&mut a, &mut b);
        assert!(max_err(&[a, b], &expected) < 1e-14, "DFT-2 forward mismatch");
    }

    #[test]
    fn dft2_inverse_roundtrip() {
        let mut a = Complex64::new(3.0, -1.0);
        let mut b = Complex64::new(-2.0, 4.0);
        let orig_a = a;
        let orig_b = b;
        // forward then unnorm-inverse should give 2× the original.
        dft2_64(&mut a, &mut b);
        dft2_64(&mut a, &mut b);
        assert!((a - 2.0 * orig_a).norm() < 1e-14);
        assert!((b - 2.0 * orig_b).norm() < 1e-14);
    }

    // ── DFT-4 ────────────────────────────────────────────────────────────────

    #[test]
    fn dft4_forward_matches_direct() {
        let input: Vec<Complex64> = (0..4)
            .map(|k| Complex64::new((k as f64 * 0.3).sin(), (k as f64 * 0.7).cos()))
            .collect();
        let expected = dft_forward_64(&input);
        let mut buf: [Complex64; 4] = input.as_slice().try_into().unwrap();
        dft4_64(&mut buf, false);
        assert!(max_err(&buf, &expected) < 1e-13, "DFT-4 forward mismatch");
    }

    #[test]
    fn dft4_inverse_roundtrip() {
        let input: Vec<Complex64> = (0..4)
            .map(|k| Complex64::new((k as f64 * 0.5).cos(), (k as f64 * 0.2).sin()))
            .collect();
        let mut buf: [Complex64; 4] = input.as_slice().try_into().unwrap();
        dft4_64(&mut buf, false);
        dft4_64(&mut buf, true);
        let recovered: Vec<Complex64> = buf.iter().map(|x| x / 4.0).collect();
        assert!(max_err(&recovered, &input) < 1e-13, "DFT-4 roundtrip mismatch");
    }

    #[test]
    fn dft4_inverse_matches_direct() {
        let input: Vec<Complex64> = (0..4)
            .map(|k| Complex64::new((k as f64 * 0.9).cos(), (k as f64 * 0.4).sin()))
            .collect();
        let expected_unnorm: Vec<Complex64> = dft_inverse_64(&input)
            .into_iter().map(|x| x * 4.0).collect();
        let mut buf: [Complex64; 4] = input.as_slice().try_into().unwrap();
        dft4_64(&mut buf, true);
        assert!(max_err(&buf, &expected_unnorm) < 1e-13, "DFT-4 inverse mismatch");
    }

    // ── DFT-8 ────────────────────────────────────────────────────────────────

    #[test]
    fn dft8_forward_matches_direct() {
        let input: Vec<Complex64> = (0..8)
            .map(|k| Complex64::new((k as f64 * 0.41).sin(), (k as f64 * 0.17).cos()))
            .collect();
        let expected = dft_forward_64(&input);
        let mut buf: [Complex64; 8] = input.as_slice().try_into().unwrap();
        dft8_64(&mut buf, false);
        let err = max_err(&buf, &expected);
        assert!(err < 1e-12, "DFT-8 forward max_err={err:.2e}");
    }

    #[test]
    fn dft8_inverse_roundtrip() {
        let input: Vec<Complex64> = (0..8)
            .map(|k| Complex64::new((k as f64 * 0.23).cos(), -(k as f64 * 0.11).sin()))
            .collect();
        let mut buf: [Complex64; 8] = input.as_slice().try_into().unwrap();
        dft8_64(&mut buf, false);
        dft8_64(&mut buf, true);
        let recovered: Vec<Complex64> = buf.iter().map(|x| x / 8.0).collect();
        let err = max_err(&recovered, &input);
        assert!(err < 1e-12, "DFT-8 roundtrip max_err={err:.2e}");
    }

    #[test]
    fn dft8_inverse_matches_direct() {
        let input: Vec<Complex64> = (0..8)
            .map(|k| Complex64::new((k as f64 * 0.33).sin(), (k as f64 * 0.22).cos()))
            .collect();
        let expected_unnorm: Vec<Complex64> = dft_inverse_64(&input)
            .into_iter().map(|x| x * 8.0).collect();
        let mut buf: [Complex64; 8] = input.as_slice().try_into().unwrap();
        dft8_64(&mut buf, true);
        let err = max_err(&buf, &expected_unnorm);
        assert!(err < 1e-12, "DFT-8 inverse max_err={err:.2e}");
    }

    #[test]
    fn dft8_f32_forward_matches_direct() {
        let input: Vec<Complex64> = (0..8)
            .map(|k| Complex64::new((k as f64 * 0.18).sin(), (k as f64 * 0.31).cos()))
            .collect();
        let expected = dft_forward_64(&input);
        let mut buf: [Complex32; 8] =
            core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
        dft8_32(&mut buf, false);
        let got: Vec<Complex64> = buf.iter().map(|x| Complex64::new(x.re as f64, x.im as f64)).collect();
        let err = max_err(&got, &expected);
        assert!(err < 1e-5, "DFT-8 f32 forward max_err={err:.2e}");
    }

    // ── DFT-16 ───────────────────────────────────────────────────────────────

    #[test]
    fn dft16_forward_matches_direct() {
        let input: Vec<Complex64> = (0..16)
            .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.13).cos()))
            .collect();
        let expected = dft_forward_64(&input);
        let mut buf: [Complex64; 16] = input.as_slice().try_into().unwrap();
        dft16_64(&mut buf, false);
        let err = max_err(&buf, &expected);
        assert!(err < 1e-11, "DFT-16 forward max_err={err:.2e}");
    }

    #[test]
    fn dft16_inverse_roundtrip() {
        let input: Vec<Complex64> = (0..16)
            .map(|k| Complex64::new((k as f64 * 0.06).cos(), (k as f64 * 0.19).sin()))
            .collect();
        let mut buf: [Complex64; 16] = input.as_slice().try_into().unwrap();
        dft16_64(&mut buf, false);
        dft16_64(&mut buf, true);
        let recovered: Vec<Complex64> = buf.iter().map(|x| x / 16.0).collect();
        let err = max_err(&recovered, &input);
        assert!(err < 1e-11, "DFT-16 roundtrip max_err={err:.2e}");
    }

    #[test]
    fn dft16_inverse_matches_direct() {
        let input: Vec<Complex64> = (0..16)
            .map(|k| Complex64::new((k as f64 * 0.44).sin(), (k as f64 * 0.36).cos()))
            .collect();
        let expected_unnorm: Vec<Complex64> = dft_inverse_64(&input)
            .into_iter().map(|x| x * 16.0).collect();
        let mut buf: [Complex64; 16] = input.as_slice().try_into().unwrap();
        dft16_64(&mut buf, true);
        let err = max_err(&buf, &expected_unnorm);
        assert!(err < 1e-11, "DFT-16 inverse max_err={err:.2e}");
    }

    // ── DFT-32 ───────────────────────────────────────────────────────────────

    #[test]
    fn dft32_forward_matches_direct() {
        let input: Vec<Complex64> = (0..32)
            .map(|k| Complex64::new((k as f64 * 0.21).sin(), (k as f64 * 0.09).cos()))
            .collect();
        let expected = dft_forward_64(&input);
        let mut buf: [Complex64; 32] = input.as_slice().try_into().unwrap();
        dft32_64(&mut buf, false);
        let err = max_err(&buf, &expected);
        assert!(err < 1e-11, "DFT-32 forward max_err={err:.2e}");
    }

    #[test]
    fn dft32_inverse_roundtrip() {
        let input: Vec<Complex64> = (0..32)
            .map(|k| Complex64::new((k as f64 * 0.14).cos(), (k as f64 * 0.37).sin()))
            .collect();
        let mut buf: [Complex64; 32] = input.as_slice().try_into().unwrap();
        dft32_64(&mut buf, false);
        dft32_64(&mut buf, true);
        let recovered: Vec<Complex64> = buf.iter().map(|x| x / 32.0).collect();
        let err = max_err(&recovered, &input);
        assert!(err < 1e-11, "DFT-32 roundtrip max_err={err:.2e}");
    }

    #[test]
    fn dft32_inverse_matches_direct() {
        let input: Vec<Complex64> = (0..32)
            .map(|k| Complex64::new((k as f64 * 0.55).sin(), (k as f64 * 0.27).cos()))
            .collect();
        let expected_unnorm: Vec<Complex64> = dft_inverse_64(&input)
            .into_iter().map(|x| x * 32.0).collect();
        let mut buf: [Complex64; 32] = input.as_slice().try_into().unwrap();
        dft32_64(&mut buf, true);
        let err = max_err(&buf, &expected_unnorm);
        assert!(err < 1e-11, "DFT-32 inverse max_err={err:.2e}");
    }

    // ── DFT-64 ───────────────────────────────────────────────────────────────

    #[test]
    fn dft64_forward_matches_direct() {
        let input: Vec<Complex64> = (0..64)
            .map(|k| Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.03).cos()))
            .collect();
        let expected = dft_forward_64(&input);
        let mut buf: [Complex64; 64] = input.as_slice().try_into().unwrap();
        dft64_64(&mut buf, false);
        let err = max_err(&buf, &expected);
        assert!(err < 1e-11, "DFT-64 forward max_err={err:.2e}");
    }

    #[test]
    fn dft64_inverse_roundtrip() {
        let input: Vec<Complex64> = (0..64)
            .map(|k| Complex64::new((k as f64 * 0.08).cos(), (k as f64 * 0.51).sin()))
            .collect();
        let mut buf: [Complex64; 64] = input.as_slice().try_into().unwrap();
        dft64_64(&mut buf, false);
        dft64_64(&mut buf, true);
        let recovered: Vec<Complex64> = buf.iter().map(|x| x / 64.0).collect();
        let err = max_err(&recovered, &input);
        assert!(err < 1e-11, "DFT-64 roundtrip max_err={err:.2e}");
    }

    #[test]
    fn dft64_inverse_matches_direct() {
        let input: Vec<Complex64> = (0..64)
            .map(|k| Complex64::new((k as f64 * 0.14).sin(), (k as f64 * 0.42).cos()))
            .collect();
        let expected_unnorm: Vec<Complex64> = dft_inverse_64(&input)
            .into_iter().map(|x| x * 64.0).collect();
        let mut buf: [Complex64; 64] = input.as_slice().try_into().unwrap();
        dft64_64(&mut buf, true);
        let err = max_err(&buf, &expected_unnorm);
        assert!(err < 1e-11, "DFT-64 inverse max_err={err:.2e}");
    }

    // ── boundary cases ───────────────────────────────────────────────────────

    #[test]
    fn dft4_impulse_produces_all_ones() {
        // DFT([1,0,0,0]) = [1,1,1,1]  (Cooley & Tukey 1965)
        let mut buf = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        dft4_64(&mut buf, false);
        for x in &buf {
            assert!((x - Complex64::new(1.0, 0.0)).norm() < 1e-14);
        }
    }

    #[test]
    fn dft8_dc_produces_energy_in_bin0() {
        // DFT([1,1,1,1,1,1,1,1]) = [8,0,0,0,0,0,0,0]
        let mut buf = [Complex64::new(1.0, 0.0); 8];
        dft8_64(&mut buf, false);
        assert!((buf[0] - Complex64::new(8.0, 0.0)).norm() < 1e-12);
        for x in &buf[1..] {
            assert!(x.norm() < 1e-12);
        }
    }

    #[test]
    fn dft16_dc_produces_energy_in_bin0() {
        let mut buf = [Complex64::new(1.0, 0.0); 16];
        dft16_64(&mut buf, false);
        assert!((buf[0] - Complex64::new(16.0, 0.0)).norm() < 1e-11);
        for x in &buf[1..] {
            assert!(x.norm() < 1e-11);
        }
    }

    #[test]
    fn dft32_dc_produces_energy_in_bin0() {
        let mut buf = [Complex64::new(1.0, 0.0); 32];
        dft32_64(&mut buf, false);
        assert!((buf[0] - Complex64::new(32.0, 0.0)).norm() < 1e-11);
        for x in &buf[1..] {
            assert!(x.norm() < 1e-11);
        }
    }

    #[test]
    fn dft64_dc_produces_energy_in_bin0() {
        let mut buf = [Complex64::new(1.0, 0.0); 64];
        dft64_64(&mut buf, false);
        assert!((buf[0] - Complex64::new(64.0, 0.0)).norm() < 1e-11);
        for x in &buf[1..] {
            assert!(x.norm() < 1e-11);
        }
    }

    #[test]
    fn dft32_f32_forward_matches_direct() {
        let input: Vec<Complex64> = (0..32)
            .map(|k| Complex64::new((k as f64 * 0.12).sin(), (k as f64 * 0.35).cos()))
            .collect();
        let expected = dft_forward_64(&input);
        let mut buf: [Complex32; 32] =
            core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
        dft32_32(&mut buf, false);
        let got: Vec<Complex64> = buf
            .iter()
            .map(|x| Complex64::new(x.re as f64, x.im as f64))
            .collect();
        let err = max_err(&got, &expected);
        assert!(err < 2e-5, "DFT-32 f32 forward max_err={err:.2e}");
    }

    #[test]
    fn dft64_f32_forward_matches_direct() {
        let input: Vec<Complex64> = (0..64)
            .map(|k| Complex64::new((k as f64 * 0.07).sin(), (k as f64 * 0.29).cos()))
            .collect();
        let expected = dft_forward_64(&input);
        let mut buf: [Complex32; 64] =
            core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
        dft64_32(&mut buf, false);
        let got: Vec<Complex64> = buf
            .iter()
            .map(|x| Complex64::new(x.re as f64, x.im as f64))
            .collect();
        let err = max_err(&got, &expected);
        assert!(err < 3e-5, "DFT-64 f32 forward max_err={err:.2e}");
    }
}
