use rayon::prelude::*;

/// Parallel dispatch threshold. Below this length, serial iteration avoids rayon spawn overhead.
const PAR_THRESHOLD: usize = 256;

/// Direct O(N²) analytical kernel for the Type-II Discrete Cosine Transform (DCT-II).
///
/// # Theorem: DCT-II Direct Projection
///
/// For a real sequence x ∈ ℝᴺ, the unnormalized DCT-II is defined by the
/// closed-form projection:
///
/// \\[ X_k = \sum_{n=0}^{N-1} x_n \cos\left[ \frac{\pi}{N} \left(n + \frac{1}{2}\right) k \right], \quad k = 0, \ldots, N-1 \\]
///
/// The basis functions `cos(πk(n + ½)/N)` are orthogonal over the half-sample-shifted
/// grid `{n + ½}_{n=0}^{N-1}`. This property follows from the finite trigonometric sum
/// identity:
///
/// \\[ \sum_{n=0}^{N-1} \cos\left[\frac{\pi k (2n+1)}{2N}\right] \cos\left[\frac{\pi l (2n+1)}{2N}\right] = \frac{N}{2} \delta_{kl} \quad (k, l \geq 1) \\]
///
/// confirming that DCT-III is the inverse of DCT-II up to the factor N/2.
///
/// # Complexity
///
/// Time: O(N²) — each of N output coefficients requires a length-N inner product.
/// Space: O(1) auxiliary — the caller owns both the input and output buffers.
///
/// For N ≥ 16 the O(N log N) fast kernel in `infrastructure::kernel::fast` is preferred.
///
/// # References
///
/// - Rao, K. R. & Yip, P. (1990). *Discrete Cosine Transform: Algorithms, Advantages,
///   Applications*. Academic Press. (Definition 2.1, pp. 24–26.)
/// - Makhoul, J. (1980). A fast cosine transform in one and two dimensions. *IEEE Trans.
///   Acoust. Speech Signal Process.*, 28(1), 27–34.
pub fn dct2(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n == 0 {
        return;
    }
    let factor = std::f64::consts::PI / n as f64;

    if n >= PAR_THRESHOLD {
        output.par_iter_mut().enumerate().for_each(|(k, out)| {
            let mut sum = 0.0;
            for n_idx in 0..n {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 0.5) * k as f64).cos();
            }
            *out = sum;
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, out)| {
            let mut sum = 0.0;
            for n_idx in 0..n {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 0.5) * k as f64).cos();
            }
            *out = sum;
        });
    }
}

/// Direct O(N²) analytical kernel for the Type-III Discrete Cosine Transform (DCT-III).
///
/// # Theorem: DCT-III Direct Projection (Inverse of DCT-II)
///
/// For a real coefficient sequence X ∈ ℝᴺ, the unnormalized DCT-III is defined by:
///
/// \\[ x_k = \frac{1}{2} X_0 + \sum_{n=1}^{N-1} X_n \cos\left[ \frac{\pi}{N} n \left(k + \frac{1}{2}\right) \right], \quad k = 0, \ldots, N-1 \\]
///
/// The leading `X_0 / 2` term arises from the n = 0 basis function evaluating to
/// `cos(0) = 1` while sharing the same orthogonality weight as the n ≥ 1 terms under
/// the standard half-sample-shifted inner product.
///
/// # Inverse relationship with DCT-II
///
/// The DCT-III is the exact inverse of DCT-II up to a scaling factor:
///
/// \\[ \text{DCT-III}(\text{DCT-II}(x)) = \frac{N}{2} \cdot x \\]
///
/// Proof: The cosine basis functions satisfy the finite orthogonality relation
/// `Σ_{n=0}^{N-1} cos(πk(2n+1)/(2N)) cos(πl(2n+1)/(2N)) = (N/2) δ_{kl}` for k, l ≥ 1.
/// The k = 0 row has norm N, and the factor of ½ in the definition compensates to yield
/// a uniform N/2 diagonal.
///
/// # Complexity
///
/// Time: O(N²). Space: O(1) auxiliary.
///
/// For N ≥ 16 the O(N log N) fast kernel in `infrastructure::kernel::fast` is preferred.
///
/// # References
///
/// - Rao, K. R. & Yip, P. (1990). *Discrete Cosine Transform: Algorithms, Advantages,
///   Applications*. Academic Press. (Definition 2.3, Inverse DCT, pp. 30–33.)
/// - Makhoul, J. (1980). A fast cosine transform in one and two dimensions. *IEEE Trans.
///   Acoust. Speech Signal Process.*, 28(1), 27–34.
pub fn dct3(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n == 0 {
        return;
    }
    let factor = std::f64::consts::PI / n as f64;
    let x0 = signal[0] * 0.5;

    if n >= PAR_THRESHOLD {
        output.par_iter_mut().enumerate().for_each(|(k, out)| {
            let mut sum = x0;
            for n_idx in 1..n {
                sum += signal[n_idx] * (factor * n_idx as f64 * (k as f64 + 0.5)).cos();
            }
            *out = sum;
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, out)| {
            let mut sum = x0;
            for n_idx in 1..n {
                sum += signal[n_idx] * (factor * n_idx as f64 * (k as f64 + 0.5)).cos();
            }
            *out = sum;
        });
    }
}

/// Direct O(N²) analytical kernel for the Type-II Discrete Sine Transform (DST-II).
///
/// # Theorem: DST-II Direct Projection
///
/// For a real sequence x ∈ ℝᴺ, the unnormalized DST-II is defined by:
///
/// \\[ X_k = \sum_{n=0}^{N-1} x_n \sin\left[ \frac{\pi}{N} \left(n + \frac{1}{2}\right) (k + 1) \right], \quad k = 0, \ldots, N-1 \\]
///
/// The basis functions `sin(π(k+1)(n+½)/N)` encode antisymmetric (odd) boundary
/// conditions on the half-sample-shifted grid `{n + ½}_{n=0}^{N-1}`. Their orthogonality
/// follows from the finite sine sum identity:
///
/// \\[ \sum_{n=0}^{N-1} \sin\left[\frac{\pi (k+1)(2n+1)}{2N}\right] \sin\left[\frac{\pi (l+1)(2n+1)}{2N}\right] = \frac{N}{2} \delta_{kl} \\]
///
/// confirming that DST-III is the inverse of DST-II up to the factor N/2.
///
/// # Complexity
///
/// Time: O(N²). Space: O(1) auxiliary.
///
/// For N ≥ 16 the O(N log N) fast kernel in `infrastructure::kernel::fast` is preferred.
///
/// # References
///
/// - Rao, K. R. & Yip, P. (1990). *Discrete Cosine Transform: Algorithms, Advantages,
///   Applications*. Academic Press. (Definition 3.1, DST-II, pp. 162–165.)
/// - Makhoul, J. (1980). A fast cosine transform in one and two dimensions. *IEEE Trans.
///   Acoust. Speech Signal Process.*, 28(1), 27–34.
pub fn dst2(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n == 0 {
        return;
    }
    let factor = std::f64::consts::PI / n as f64;

    if n >= PAR_THRESHOLD {
        output.par_iter_mut().enumerate().for_each(|(k, out)| {
            let mut sum = 0.0;
            for n_idx in 0..n {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 0.5) * (k as f64 + 1.0)).sin();
            }
            *out = sum;
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, out)| {
            let mut sum = 0.0;
            for n_idx in 0..n {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 0.5) * (k as f64 + 1.0)).sin();
            }
            *out = sum;
        });
    }
}

/// Direct O(N²) analytical kernel for the Type-III Discrete Sine Transform (DST-III).
///
/// # Theorem: DST-III Direct Projection (Inverse of DST-II)
///
/// For a real coefficient sequence X ∈ ℝᴺ, the unnormalized DST-III is defined by:
///
/// \\[ x_k = \frac{(-1)^k}{2} X_{N-1} + \sum_{n=0}^{N-2} X_n \sin\left[ \frac{\pi}{N} (n + 1) \left(k + \frac{1}{2}\right) \right], \quad k = 0, \ldots, N-1 \\]
///
/// The boundary term `(-1)^k X_{N-1} / 2` corresponds to the n = N-1 basis function
/// `sin(πN(2k+1)/(2N)) = sin(π(2k+1)/2) = (-1)^k`, which oscillates with period 2
/// at the Nyquist frequency. The factor ½ is required for the same orthogonality
/// normalization reason as in DCT-III.
///
/// # Inverse relationship with DST-II
///
/// The DST-III is the exact inverse of DST-II up to a scaling factor:
///
/// \\[ \text{DST-III}(\text{DST-II}(x)) = \frac{N}{2} \cdot x \\]
///
/// Proof: The sine basis functions satisfy the finite orthogonality relation
/// `Σ_{n=0}^{N-1} sin(π(k+1)(2n+1)/(2N)) sin(π(l+1)(2n+1)/(2N)) = (N/2) δ_{kl}`.
/// The n = N-1 boundary term contributes (-1)^k per output, and the factor ½ in the
/// definition yields a uniform N/2 diagonal under the full N×N product.
///
/// # Complexity
///
/// Time: O(N²). Space: O(1) auxiliary.
///
/// For N ≥ 16 the O(N log N) fast kernel in `infrastructure::kernel::fast` is preferred.
///
/// # References
///
/// - Rao, K. R. & Yip, P. (1990). *Discrete Cosine Transform: Algorithms, Advantages,
///   Applications*. Academic Press. (Definition 3.3, DST-III, Inverse DST, pp. 166–169.)
/// - Makhoul, J. (1980). A fast cosine transform in one and two dimensions. *IEEE Trans.
///   Acoust. Speech Signal Process.*, 28(1), 27–34.
pub fn dst3(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n == 0 {
        return;
    }
    let factor = std::f64::consts::PI / n as f64;
    let xn = signal[n - 1] * 0.5;

    if n >= PAR_THRESHOLD {
        output.par_iter_mut().enumerate().for_each(|(k, out)| {
            let sign = if k % 2 == 1 { -1.0 } else { 1.0 };
            let mut sum = sign * xn;
            for n_idx in 0..(n - 1) {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 1.0) * (k as f64 + 0.5)).sin();
            }
            *out = sum;
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, out)| {
            let sign = if k % 2 == 1 { -1.0 } else { 1.0 };
            let mut sum = sign * xn;
            for n_idx in 0..(n - 1) {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 1.0) * (k as f64 + 0.5)).sin();
            }
            *out = sum;
        });
    }
}
