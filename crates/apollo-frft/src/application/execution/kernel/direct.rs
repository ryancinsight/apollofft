use num_complex::Complex64;
use std::f64::consts::PI;

/// Evaluate the direct $O(N^2)$ fractional Fourier kernel on a centered grid into a user-provided buffer.
#[allow(clippy::similar_names)]
pub fn direct_frft_forward_into(
    input: &[Complex64],
    output: &mut [Complex64],
    order: f64,
    cot: f64,
    csc: f64,
    scale: Complex64,
) {
    let n = input.len();
    if n == 0 {
        return;
    }
    assert_eq!(n, output.len(), "FrFT output buffer length mismatch");

    let reduced = order.rem_euclid(4.0);
    if (reduced - 0.0).abs() < 1.0e-12 {
        output.copy_from_slice(input);
        return;
    }
    if (reduced - 2.0).abs() < 1.0e-12 {
        for (i, &val) in input.iter().enumerate() {
            output[n - 1 - i] = val;
        }
        return;
    }

    let sign = if (reduced - 3.0).abs() < 1.0e-12 {
        1.0
    } else {
        -1.0
    };

    let center = (n as f64 - 1.0) * 0.5;

    if (reduced - 1.0).abs() < 1.0e-12 || (reduced - 3.0).abs() < 1.0e-12 {
        let scale = 1.0 / (n as f64).sqrt();
        for (k, out) in output.iter_mut().enumerate() {
            let u = k as f64 - center;
            let mut sum = Complex64::new(0.0, 0.0);
            for (j, &value) in input.iter().enumerate() {
                let x = j as f64 - center;
                let angle = sign * 2.0 * PI * (x * u) / n as f64;
                sum += value * Complex64::new(angle.cos(), angle.sin());
            }
            *out = sum * scale;
        }
        return;
    }

    for (k, out) in output.iter_mut().enumerate() {
        let u = k as f64 - center;
        let mut sum = Complex64::new(0.0, 0.0);
        for (j, &value) in input.iter().enumerate() {
            let x = j as f64 - center;
            let phase = PI * ((x * x + u * u) * cot - 2.0 * x * u * csc) / n as f64;
            sum += value * Complex64::new(phase.cos(), phase.sin());
        }
        *out = sum * scale;
    }
}
