use crate::domain::contracts::error::CztError;
use ndarray::Array1;
use num_complex::Complex64;

/// Direct execution of the Chirp Z-Transform logic.
/// Evaluates sequentially against $O(NM)$ limits.
pub fn czt_direct_forward(
    input: &Array1<Complex64>,
    output_len: usize,
    a: Complex64,
    w: Complex64,
) -> Result<Array1<Complex64>, CztError> {
    let output = Array1::from_shape_fn(output_len, |k| {
        let z_k = a * w.powf(-(k as f64));
        let mut sum = Complex64::new(0.0, 0.0);
        let mut z_pow = Complex64::new(1.0, 0.0);
        let z_inv = Complex64::new(1.0, 0.0) / z_k;
        for value in input.iter() {
            sum += *value * z_pow;
            z_pow *= z_inv;
        }
        sum
    });

    Ok(output)
}
