//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use apollo_sft::SparseFftPlan;
    use num_complex::{Complex32, Complex64};

    use crate::{SftWgpuBackend, SftWgpuPlan, WgpuCapabilities, WgpuError};

    #[test]
    fn capabilities_advertise_direct_dense_sparse_execution() {
        let capabilities = WgpuCapabilities::direct_dense_spectrum(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
    }

    #[test]
    fn plan_preserves_logical_length_and_sparsity() {
        let plan = SftWgpuPlan::new(64, 5);
        assert_eq!(plan.len(), 64);
        assert_eq!(plan.sparsity(), 5);
        assert!(!SftWgpuPlan::new(64, 5).is_empty());
        assert!(SftWgpuPlan::new(0, 5).is_empty());
        assert!(SftWgpuPlan::new(64, 0).is_empty());
    }

    #[test]
    fn unsupported_execution_error_identifies_operation() {
        let err = WgpuError::UnsupportedExecution {
            operation: "forward",
        };
        assert_eq!(
            err.to_string(),
            "forward is unsupported by the current WGPU capability set"
        );
    }

    #[test]
    fn invalid_plan_rejects_zero_length_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let error = backend
            .execute_forward(&SftWgpuPlan::new(0, 1), &[])
            .expect_err("zero length must be invalid");
        assert_eq!(
            error,
            WgpuError::InvalidPlan {
                len: 0,
                sparsity: 1,
                message: "transform length must be greater than zero"
            }
        );
    }

    #[test]
    fn input_length_mismatch_reports_expected_and_actual_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let error = backend
            .execute_forward(&SftWgpuPlan::new(8, 2), &[Complex32::new(0.0, 0.0); 4])
            .expect_err("mismatched input length must be invalid");
        assert_eq!(
            error,
            WgpuError::InputLengthMismatch {
                expected: 8,
                actual: 4
            }
        );
    }

    #[test]
    fn forward_matches_cpu_sparse_support_and_coefficients_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let plan = SftWgpuPlan::new(8, 2);
        let signal = two_tone_signal(8, &[(1, 3.0), (3, 1.25)]);
        let signal_f32: Vec<Complex32> = signal
            .iter()
            .map(|value| Complex32::new(value.re as f32, value.im as f32))
            .collect();

        let cpu = SparseFftPlan::new(plan.len(), plan.sparsity())
            .expect("valid CPU plan")
            .forward(&signal)
            .expect("CPU SFT");
        let gpu = backend
            .execute_forward(&plan, &signal_f32)
            .expect("GPU SFT");

        assert_eq!(gpu.frequencies, cpu.frequencies);
        assert_eq!(gpu.values.len(), cpu.values.len());
        for (actual, expected) in gpu.values.iter().zip(cpu.values.iter()) {
            assert_complex64_close(*actual, *expected, 2.0e-4);
        }
    }

    #[test]
    fn inverse_matches_cpu_sparse_reconstruction_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let plan = SftWgpuPlan::new(8, 2);
        let signal = two_tone_signal(8, &[(1, 3.0), (3, 1.25)]);
        let cpu_plan = SparseFftPlan::new(plan.len(), plan.sparsity()).expect("valid CPU plan");
        let spectrum = cpu_plan.forward(&signal).expect("CPU SFT");
        let expected = cpu_plan.inverse(&spectrum).expect("CPU inverse");

        let actual = backend
            .execute_inverse(&plan, &spectrum)
            .expect("GPU inverse");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex32_close(
                *actual,
                Complex32::new(expected.re as f32, expected.im as f32),
                2.0e-4,
            );
        }
    }

    fn backend_or_skip() -> Option<SftWgpuBackend> {
        match SftWgpuBackend::try_default() {
            Ok(backend) => Some(backend),
            Err(error) => {
                eprintln!("skipping WGPU-dependent SFT test: {error}");
                None
            }
        }
    }

    fn two_tone_signal(len: usize, tones: &[(usize, f64)]) -> Vec<Complex64> {
        (0..len)
            .map(|n| {
                tones
                    .iter()
                    .map(|(frequency, amplitude)| {
                        let angle = 2.0 * std::f64::consts::PI * (*frequency as f64) * (n as f64)
                            / (len as f64);
                        Complex64::new(amplitude * angle.cos(), amplitude * angle.sin())
                    })
                    .sum()
            })
            .collect()
    }

    fn assert_complex64_close(actual: Complex64, expected: Complex64, tolerance: f64) {
        assert!(
            (actual.re - expected.re).abs() <= tolerance,
            "real mismatch: actual={actual:?}, expected={expected:?}"
        );
        assert!(
            (actual.im - expected.im).abs() <= tolerance,
            "imag mismatch: actual={actual:?}, expected={expected:?}"
        );
    }

    fn assert_complex32_close(actual: Complex32, expected: Complex32, tolerance: f32) {
        assert!(
            (actual.re - expected.re).abs() <= tolerance,
            "real mismatch: actual={actual:?}, expected={expected:?}"
        );
        assert!(
            (actual.im - expected.im).abs() <= tolerance,
            "imag mismatch: actual={actual:?}, expected={expected:?}"
        );
    }
}
