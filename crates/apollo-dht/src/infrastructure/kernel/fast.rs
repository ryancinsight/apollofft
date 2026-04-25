//! Fast \(O(N \log N)\) FFT-based kernel execution for large arrays.
//! Leverages the exact correspondence between the Discrete Hartley Transform and the Discrete Fourier Transform.

use apollo_fft::application::execution::kernel::fft_forward_64;
use num_complex::Complex64;

/// Computes the unnormalized DHT via an \(O(N \log N)\) FFT mapped path.
///
/// Ensures exact mathematical equivalence:
/// \(H_k = \operatorname{Re}(X_k) - \operatorname{Im}(X_k)\) where \(X = \operatorname{FFT}(x)\).
pub fn dht_fast(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n == 0 {
        return;
    }

    let mut scratch = vec![Complex64::new(0.0, 0.0); n];
    for (i, &x) in signal.iter().enumerate() {
        scratch[i] = Complex64::new(x, 0.0);
    }

    fft_forward_64(&mut scratch);

    for (out, x_k) in output.iter_mut().zip(scratch.iter()) {
        *out = x_k.re - x_k.im;
    }
}
