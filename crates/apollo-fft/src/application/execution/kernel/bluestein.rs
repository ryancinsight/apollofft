//! Bluestein chirp-Z FFT for arbitrary-length DFT.
//!
//! ## Mathematical derivation
//!
//! Using the Bluestein identity kn = [-(k-n)^2 + k^2 + n^2] / 2:
//! X[k] = chirp[k] * (a_M circular_conv b_M)[k]
//! where chirp[k] = exp(-pi*i*k^2/N), a_M[n] = x[n]*chirp[n] (zero-padded to M),
//! b_M is the filter exp(+pi*i*m^2/N) arranged for circular convolution,
//! and M = next_pow2(2N-1).

use super::radix2;
use super::radix_stage::normalize_inplace;
use num_complex::{Complex32, Complex64};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

static BLUESTEIN_PLAN_CACHE_64: Lazy<RwLock<HashMap<usize, Arc<BluesteinPlan64>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static BLUESTEIN_PLAN_CACHE_32: Lazy<RwLock<HashMap<usize, Arc<BluesteinPlan32>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

thread_local! {
    static BLUESTEIN_SCRATCH_64: RefCell<HashMap<usize, Vec<Complex64>>> =
        RefCell::new(HashMap::new());
    static BLUESTEIN_SCRATCH_32: RefCell<HashMap<usize, Vec<Complex32>>> =
        RefCell::new(HashMap::new());
}

#[inline]
fn cached_plan64(n: usize) -> Arc<BluesteinPlan64> {
    if let Some(plan) = BLUESTEIN_PLAN_CACHE_64.read().get(&n).cloned() {
        return plan;
    }
    let plan = Arc::new(BluesteinPlan64::new(n));
    BLUESTEIN_PLAN_CACHE_64
        .write()
        .entry(n)
        .or_insert_with(|| Arc::clone(&plan))
        .clone()
}

#[inline]
fn cached_plan32(n: usize) -> Arc<BluesteinPlan32> {
    if let Some(plan) = BLUESTEIN_PLAN_CACHE_32.read().get(&n).cloned() {
        return plan;
    }
    let plan = Arc::new(BluesteinPlan32::new(n));
    BLUESTEIN_PLAN_CACHE_32
        .write()
        .entry(n)
        .or_insert_with(|| Arc::clone(&plan))
        .clone()
}

#[inline]
fn with_scratch64<R, F>(m: usize, f: F) -> R
where
    F: FnOnce(&mut [Complex64]) -> R,
{
    BLUESTEIN_SCRATCH_64.with(|map_cell| {
        let mut map = map_cell.borrow_mut();
        let scratch = map
            .entry(m)
            .or_insert_with(|| vec![Complex64::new(0.0, 0.0); m]);
        if scratch.len() != m {
            scratch.resize(m, Complex64::new(0.0, 0.0));
        }
        let out = f(scratch.as_mut_slice());
        out
    })
}

#[inline]
fn with_scratch32<R, F>(m: usize, f: F) -> R
where
    F: FnOnce(&mut [Complex32]) -> R,
{
    BLUESTEIN_SCRATCH_32.with(|map_cell| {
        let mut map = map_cell.borrow_mut();
        let scratch = map
            .entry(m)
            .or_insert_with(|| vec![Complex32::new(0.0, 0.0); m]);
        if scratch.len() != m {
            scratch.resize(m, Complex32::new(0.0, 0.0));
        }
        let out = f(scratch.as_mut_slice());
        out
    })
}

/// Precomputed context for arbitrary-length Bluestein chirp-Z transform.
/// Eliminates `O(N)` dynamic memory allocations per kernel evaluation.
///
/// # Chirp filter semantics
///
/// - `chirp[k]` = `exp(-πi k²/N)` — forward chirp factor
/// - `b_m` = FFT of filter `b[m] = chirp[m].conj()` = `exp(+πi m²/N)`,
///   used for the forward-transform convolution step
/// - `b_m_forward` = FFT of filter `b[m] = chirp[m]` = `exp(-πi m²/N)`,
///   used for the inverse-transform convolution step
/// - `twiddle_fwd` = contiguous per-stage forward twiddle table for the M-point FFT,
///   cached in the plan so hot-path transforms bypass O(M log₂M / 2) trig calls.
/// - `twiddle_inv` = contiguous per-stage inverse twiddle table for the M-point IFFT.
#[derive(Clone, Debug)]
pub struct BluesteinPlan64 {
    n: usize,
    m: usize,
    chirp: Vec<Complex64>,
    b_m: Vec<Complex64>,
    b_m_forward: Vec<Complex64>,
    twiddle_fwd: Vec<Complex64>,
    twiddle_inv: Vec<Complex64>,
}

impl BluesteinPlan64 {
    /// Initialize a new Bluestein plan for length `n`.
    ///
    /// Precomputes the M-point forward and inverse twiddle tables so hot-path
    /// calls to `forward_with_scratch` and `inverse_unnorm_with_scratch` bypass
    /// O(M/2) trigonometric calls that would otherwise occur per transform.
    pub fn new(n: usize) -> Self {
        let m = (2 * n.saturating_sub(1).max(1)).next_power_of_two();
        let chirp: Vec<Complex64> = (0..n)
            .map(|k| {
                let angle = -std::f64::consts::PI * (k * k) as f64 / n as f64;
                Complex64::new(angle.cos(), angle.sin())
            })
            .collect();

        // Precompute M-point twiddle tables once; reused by every hot-path call.
        let twiddle_fwd = radix2::build_forward_twiddle_table_64(m);
        let twiddle_inv = radix2::build_inverse_twiddle_table_64(m);

        let mut b_m = vec![Complex64::new(0.0, 0.0); m];
        b_m[0] = Complex64::new(1.0, 0.0);
        for k in 1..n {
            let bk = chirp[k].conj();
            b_m[k] = bk;
            b_m[m - k] = bk;
        }
        radix2::forward_inplace_64_with_twiddles(&mut b_m, &twiddle_fwd);

        let mut b_m_forward = vec![Complex64::new(0.0, 0.0); m];
        b_m_forward[0] = Complex64::new(1.0, 0.0);
        for k in 1..n {
            let bk = chirp[k];
            b_m_forward[k] = bk;
            b_m_forward[m - k] = bk;
        }
        radix2::forward_inplace_64_with_twiddles(&mut b_m_forward, &twiddle_fwd);

        Self {
            n,
            m,
            chirp,
            b_m,
            b_m_forward,
            twiddle_fwd,
            twiddle_inv,
        }
    }

    /// Retrieve the working padded dimension M.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Forward transform into data given a pre-allocated scratch sequence of length `M`.
    ///
    /// Uses the plan-cached M-point twiddle tables; avoids O(M/2) trig calls per call.
    pub fn forward_with_scratch(&self, data: &mut [Complex64], scratch_a: &mut [Complex64]) {
        assert_eq!(data.len(), self.n);
        assert_eq!(scratch_a.len(), self.m);

        for (dst, (&x, &c)) in scratch_a[..self.n]
            .iter_mut()
            .zip(data.iter().zip(self.chirp.iter()))
        {
            *dst = x * c;
        }
        if self.n < scratch_a.len() {
            scratch_a[self.n..].fill(Complex64::new(0.0, 0.0));
        }
        radix2::forward_inplace_64_with_twiddles(scratch_a, &self.twiddle_fwd);

        for (a, b) in scratch_a.iter_mut().zip(self.b_m.iter()) {
            *a *= b;
        }
        radix2::inverse_inplace_64_with_twiddles(scratch_a, &self.twiddle_inv);

        for (k, x_k) in data.iter_mut().enumerate() {
            *x_k = self.chirp[k] * scratch_a[k];
        }
    }

    /// Inverse unnormalized transform.
    ///
    /// Uses the plan-cached M-point twiddle tables; avoids O(M/2) trig calls per call.
    pub fn inverse_unnorm_with_scratch(&self, data: &mut [Complex64], scratch_a: &mut [Complex64]) {
        assert_eq!(data.len(), self.n);
        assert_eq!(scratch_a.len(), self.m);

        for (dst, (&x, &c)) in scratch_a[..self.n]
            .iter_mut()
            .zip(data.iter().zip(self.chirp.iter()))
        {
            *dst = x * c.conj();
        }
        if self.n < scratch_a.len() {
            scratch_a[self.n..].fill(Complex64::new(0.0, 0.0));
        }
        radix2::forward_inplace_64_with_twiddles(scratch_a, &self.twiddle_fwd);

        for (a, b) in scratch_a.iter_mut().zip(self.b_m_forward.iter()) {
            *a *= b;
        }
        radix2::inverse_inplace_64_with_twiddles(scratch_a, &self.twiddle_inv);

        for (n_idx, y_n) in data.iter_mut().enumerate() {
            *y_n = self.chirp[n_idx].conj() * scratch_a[n_idx];
        }
    }
}

/// Cached per-length Bluestein context for `Complex32` transforms.
///
/// Stores chirp vectors, padded convolution kernels, and the M-point radix-2
/// twiddle tables needed on every call. The structure is cloneable via `Arc`
/// for lock-free sharing across transform invocations.
#[derive(Clone, Debug)]
pub struct BluesteinPlan32 {
    n: usize,
    m: usize,
    chirp: Vec<Complex32>,
    b_m: Vec<Complex32>,
    b_m_forward: Vec<Complex32>,
    twiddle_fwd: Vec<Complex32>,
    twiddle_inv: Vec<Complex32>,
}

impl BluesteinPlan32 {
    /// Build a reusable plan for a non-power-of-two length `n` transform.
    pub fn new(n: usize) -> Self {
        let m = (2 * n.saturating_sub(1).max(1)).next_power_of_two();
        let chirp: Vec<Complex32> = (0..n)
            .map(|k| {
                let angle = -(std::f64::consts::PI * (k as f64 * k as f64) / n as f64) as f32;
                Complex32::new(angle.cos(), angle.sin())
            })
            .collect();

        let twiddle_fwd = radix2::build_forward_twiddle_table_32(m);
        let twiddle_inv = radix2::build_inverse_twiddle_table_32(m);

        let mut b_m = vec![Complex32::new(0.0, 0.0); m];
        b_m[0] = Complex32::new(1.0, 0.0);
        for k in 1..n {
            let bk = chirp[k].conj();
            b_m[k] = bk;
            b_m[m - k] = bk;
        }
        radix2::forward_inplace_32_with_twiddles(&mut b_m, &twiddle_fwd);

        let mut b_m_forward = vec![Complex32::new(0.0, 0.0); m];
        b_m_forward[0] = Complex32::new(1.0, 0.0);
        for k in 1..n {
            let bk = chirp[k];
            b_m_forward[k] = bk;
            b_m_forward[m - k] = bk;
        }
        radix2::forward_inplace_32_with_twiddles(&mut b_m_forward, &twiddle_fwd);

        Self {
            n,
            m,
            chirp,
            b_m,
            b_m_forward,
            twiddle_fwd,
            twiddle_inv,
        }
    }

    /// Padded convolution size (next power-of-two of `2n - 1`).
    pub fn m(&self) -> usize {
        self.m
    }

    /// Forward transform using preallocated scratch and plan-owned metadata.
    pub fn forward_with_scratch(&self, data: &mut [Complex32], scratch_a: &mut [Complex32]) {
        assert_eq!(data.len(), self.n);
        assert_eq!(scratch_a.len(), self.m);

        for (dst, (&x, &c)) in scratch_a[..self.n]
            .iter_mut()
            .zip(data.iter().zip(self.chirp.iter()))
        {
            *dst = x * c;
        }
        if self.n < scratch_a.len() {
            scratch_a[self.n..].fill(Complex32::new(0.0, 0.0));
        }
        radix2::forward_inplace_32_with_twiddles(scratch_a, &self.twiddle_fwd);

        for (a, b) in scratch_a.iter_mut().zip(self.b_m.iter()) {
            *a *= b;
        }
        radix2::inverse_inplace_32_with_twiddles(scratch_a, &self.twiddle_inv);

        for (k, x_k) in data.iter_mut().enumerate() {
            *x_k = self.chirp[k] * scratch_a[k];
        }
    }

    /// Forward inverse (unnormalized) transform using preallocated scratch.
    pub fn inverse_unnorm_with_scratch(&self, data: &mut [Complex32], scratch_a: &mut [Complex32]) {
        assert_eq!(data.len(), self.n);
        assert_eq!(scratch_a.len(), self.m);

        for (dst, (&x, &c)) in scratch_a[..self.n]
            .iter_mut()
            .zip(data.iter().zip(self.chirp.iter()))
        {
            *dst = x * c.conj();
        }
        if self.n < scratch_a.len() {
            scratch_a[self.n..].fill(Complex32::new(0.0, 0.0));
        }
        radix2::forward_inplace_32_with_twiddles(scratch_a, &self.twiddle_fwd);

        for (a, b) in scratch_a.iter_mut().zip(self.b_m_forward.iter()) {
            *a *= b;
        }
        radix2::inverse_inplace_32_with_twiddles(scratch_a, &self.twiddle_inv);

        for (n_idx, y_n) in data.iter_mut().enumerate() {
            *y_n = self.chirp[n_idx].conj() * scratch_a[n_idx];
        }
    }
}

/// In-place forward Bluestein chirp-Z transform for `Complex64`.
///
/// Computes the exact discrete Fourier transform in $O(N \log N)$ time for arbitrary
/// and non-power-of-two lengths by zero-padding and using cyclic convolution.
/// When `data.len()` is a power-of-two, execution delegates to the radix-2 kernel.
pub fn forward_inplace_64(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        radix2::forward_inplace_64(data);
        return;
    }
    let plan = cached_plan64(n);
    with_scratch64(plan.m(), |a_m| plan.forward_with_scratch(data, a_m));
}

/// In-place unnormalized inverse Bluestein chirp-Z transform for `Complex64`.
///
/// Computes the adjoint discrete Fourier transform without the $1/N$ scaling factor.
/// Employs conjugated chirp sequences while maintaining cyclic convolution guarantees.
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        radix2::inverse_inplace_unnorm_64(data);
        return;
    }
    let plan = cached_plan64(n);
    with_scratch64(plan.m(), |a_m| plan.inverse_unnorm_with_scratch(data, a_m));
}

/// In-place normalized inverse Bluestein chirp-Z transform for `Complex64`.
///
/// Computes the exact inverse discrete Fourier transform by evaluating the unnormalized
/// transformation and applying a $1/N$ scaling factor.
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    inverse_inplace_unnorm_64(data);
    normalize_inplace(data, 1.0 / data.len() as f64);
}

/// In-place forward Bluestein chirp-Z transform for `Complex32`.
///
/// Evaluates arbitrary-length DFT in-place on native `Complex32` with a cached
/// plan and reusable scratch.
pub fn forward_inplace_32(data: &mut [Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        radix2::forward_inplace_32(data);
        return;
    }
    let plan = cached_plan32(n);
    with_scratch32(plan.m(), |a_m| plan.forward_with_scratch(data, a_m));
}

/// In-place unnormalized inverse Bluestein chirp-Z transform for `Complex32`.
///
/// Evaluates the adjoint arbitrary-length DFT without scaling on native `Complex32`
/// using cached kernels and reusable scratch.
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        radix2::inverse_inplace_unnorm_32(data);
        return;
    }
    let plan = cached_plan32(n);
    with_scratch32(plan.m(), |a_m| plan.inverse_unnorm_with_scratch(data, a_m));
}

/// In-place normalized inverse Bluestein chirp-Z transform for `Complex32`.
///
/// Sequentially computes the unnormalized inverse and scales by $1/N$.
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    inverse_inplace_unnorm_32(data);
    normalize_inplace(data, 1.0f32 / data.len() as f32);
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::max_abs_err_64 as max_abs_err;
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward_64, dft_inverse_64};

    fn sig(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|k| {
                let t = k as f64 / n as f64;
                Complex64::new(
                    (std::f64::consts::TAU * 3.0 * t).sin()
                        + 0.5 * (std::f64::consts::TAU * 7.0 * t).cos(),
                    0.25 * (std::f64::consts::TAU * 2.0 * t).sin(),
                )
            })
            .collect()
    }

    #[test]
    fn forward_matches_direct_for_n3() {
        let input = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let expected = dft_forward_64(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        assert!(max_abs_err(&got, &expected) < 1e-12);
    }

    #[test]
    fn forward_matches_direct_for_n5() {
        let input = sig(5);
        let expected = dft_forward_64(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        assert!(
            max_abs_err(&got, &expected) < 1e-12,
            "n=5 err={}",
            max_abs_err(&got, &expected)
        );
    }

    #[test]
    fn forward_matches_direct_for_n6() {
        let input = sig(6);
        let expected = dft_forward_64(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        assert!(
            max_abs_err(&got, &expected) < 1e-12,
            "n=6 err={}",
            max_abs_err(&got, &expected)
        );
    }

    #[test]
    fn forward_matches_direct_for_n7() {
        let input = sig(7);
        let expected = dft_forward_64(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        assert!(
            max_abs_err(&got, &expected) < 1e-12,
            "n=7 err={}",
            max_abs_err(&got, &expected)
        );
    }

    #[test]
    fn forward_matches_direct_for_n11() {
        let input = sig(11);
        let expected = dft_forward_64(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        assert!(
            max_abs_err(&got, &expected) < 1e-11,
            "n=11 err={}",
            max_abs_err(&got, &expected)
        );
    }

    #[test]
    fn roundtrip_for_non_power_of_two() {
        for &n in &[3usize, 5, 6, 7, 9, 11] {
            let input = sig(n);
            let mut data = input.clone();
            forward_inplace_64(&mut data);
            inverse_inplace_64(&mut data);
            let err = max_abs_err(&data, &input);
            assert!(err < 1e-11, "roundtrip n={n} err={err:.2e}");
        }
    }

    #[test]
    fn repeated_forward_same_input_same_output() {
        let input = sig(45);
        let mut a = input.clone();
        let mut b = input;
        forward_inplace_64(&mut a);
        forward_inplace_64(&mut b);
        let err = max_abs_err(&a, &b);
        assert!(err < 1e-12, "repeat determinism err={err:.2e}");
    }

    #[test]
    fn unnorm_inverse_differs_from_norm_by_n() {
        let n = 7usize;
        let input = sig(n);
        let mut spec = input.clone();
        forward_inplace_64(&mut spec);
        let mut unnorm = spec.clone();
        inverse_inplace_unnorm_64(&mut unnorm);
        let mut norm = spec.clone();
        inverse_inplace_64(&mut norm);
        let err = max_abs_err(
            &unnorm,
            &norm.iter().map(|x| x * n as f64).collect::<Vec<_>>(),
        );
        assert!(err < 1e-11, "unnorm/norm ratio failed n={n}: err={err:.2e}");
    }

    #[test]
    fn inverse_matches_direct_for_n5() {
        let input = sig(5);
        let expected = dft_inverse_64(&input);
        let mut got = input.clone();
        inverse_inplace_64(&mut got);
        assert!(
            max_abs_err(&got, &expected) < 1e-12,
            "inverse n=5 err={}",
            max_abs_err(&got, &expected)
        );
    }

    #[test]
    fn power_of_two_falls_through_to_radix2() {
        let n = 8usize;
        let input = sig(n);
        let mut bl = input.clone();
        forward_inplace_64(&mut bl);
        let mut r2 = input.clone();
        radix2::forward_inplace_64(&mut r2);
        assert!(max_abs_err(&bl, &r2) < 1e-14, "bluestein vs radix2 n=8");
    }
}
