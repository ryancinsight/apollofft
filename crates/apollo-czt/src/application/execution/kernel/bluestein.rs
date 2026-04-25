use apollo_fft::application::plan::FftPlan1D;
use ndarray::{Array1, Zip};
use num_complex::Complex64;

/// Evaluates Bluestein's fast algorithm over precomputed chirp variables
/// using an optimized invariant `fft_kernel`.
///
/// # Theorem
///
/// For the chirp z-transform
/// `X_k = sum_n x_n a^-n w^(nk)`, the identity
/// `nk = (n^2 + k^2 - (k - n)^2) / 2` rewrites the transform as a linear
/// convolution between `x_n a^-n w^(n^2/2)` and `w^(-(k-n)^2/2)`, followed by
/// multiplication by `w^(k^2/2)`. This function evaluates that convolution
/// with Apollo FFT plans over zero-padded buffers.
///
/// # Proof sketch
///
/// Substituting the identity into `w^(nk)` factors all `n`-only and `k`-only
/// terms out of the summation. The remaining `(k-n)` term is Toeplitz and
/// becomes a cyclic convolution after zero-padding to at least `n + m - 1`.
/// The FFT convolution theorem then gives the same result as direct CZT in
/// exact arithmetic.
#[must_use]
pub fn czt_bluestein_forward(
    input: &Array1<Complex64>,
    output_len: usize,
    convolution_len: usize,
    chirp_n: &[Complex64],
    chirp_k: &[Complex64],
    fft_kernel: &Array1<Complex64>,
    fft_plan: &FftPlan1D,
) -> Array1<Complex64> {
    let mut output = Array1::<Complex64>::zeros(output_len);
    czt_bluestein_forward_into(
        input,
        &mut output,
        convolution_len,
        chirp_n,
        chirp_k,
        fft_kernel,
        fft_plan,
    );
    output
}

/// Evaluates Bluestein's fast CZT into caller-owned output storage.
///
/// This path uses one convolution workspace. The workspace is transformed,
/// multiplied by the precomputed FFT kernel in place, inverse transformed in
/// place, and then sampled into `output`.
pub fn czt_bluestein_forward_into(
    input: &Array1<Complex64>,
    output: &mut Array1<Complex64>,
    convolution_len: usize,
    chirp_n: &[Complex64],
    chirp_k: &[Complex64],
    fft_kernel: &Array1<Complex64>,
    fft_plan: &FftPlan1D,
) {
    assert_eq!(
        output.len(),
        chirp_k.len(),
        "CZT output length must match chirp_k length"
    );
    assert_eq!(
        input.len(),
        chirp_n.len(),
        "CZT input length must match chirp_n length"
    );
    assert_eq!(
        fft_kernel.len(),
        convolution_len,
        "CZT FFT kernel length must match convolution length"
    );

    let mut workspace = vec![Complex64::new(0.0, 0.0); convolution_len];
    for n_idx in 0..input.len() {
        workspace[n_idx] = input[n_idx] * chirp_n[n_idx];
    }

    let mut workspace = Array1::from_vec(workspace);
    fft_plan.forward_complex_inplace(&mut workspace);

    Zip::from(&mut workspace)
        .and(fft_kernel)
        .for_each(|value, &kernel_value| *value *= kernel_value);

    fft_plan.inverse_complex_inplace(&mut workspace);

    for (k, out) in output.iter_mut().enumerate() {
        *out = chirp_k[k] * workspace[k];
    }
}
