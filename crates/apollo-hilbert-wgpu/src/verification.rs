//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use crate::{HilbertWgpuBackend, HilbertWgpuPlan, WgpuCapabilities, WgpuError};
    use apollo_hilbert::HilbertPlan;

    #[test]
    fn capabilities_reflect_forward_only_kernel_surface() {
        let capabilities = WgpuCapabilities::forward_only(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(!capabilities.supports_inverse);
        assert!(capabilities.supports_mixed_precision);
        assert_eq!(
            capabilities.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
    }

    #[test]
    fn plan_preserves_logical_length() {
        let plan = HilbertWgpuPlan::new(64);
        assert_eq!(plan.len(), 64);
        assert!(!HilbertWgpuPlan::new(64).is_empty());
        assert!(HilbertWgpuPlan::new(0).is_empty());
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
    fn backend_reports_forward_only_when_device_exists() {
        let Ok(backend) = HilbertWgpuBackend::try_default() else {
            return;
        };
        let capabilities = backend.capabilities();
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(!capabilities.supports_inverse);
    }

    #[test]
    fn analytic_signal_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = HilbertWgpuBackend::try_default() else {
            return;
        };
        let input = vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
        let plan = backend.plan(input.len());
        let gpu = backend
            .execute_analytic_signal(&plan, &input)
            .expect("wgpu analytic execution");

        let cpu_plan = HilbertPlan::new(input.len()).expect("cpu plan");
        let cpu = cpu_plan
            .analytic_signal(
                &input
                    .iter()
                    .map(|&value| f64::from(value))
                    .collect::<Vec<_>>(),
            )
            .expect("cpu analytic");

        assert_eq!(gpu.len(), cpu.values().len());
        for (actual, expected) in gpu.iter().zip(cpu.values().iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 5.0e-4);
            assert!((f64::from(actual.im) - expected.im).abs() < 5.0e-4);
        }
    }

    #[test]
    fn quadrature_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = HilbertWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.5_f32, -1.25, 2.75, 4.0, -3.5, 1.0];
        let plan = backend.plan(input.len());
        let gpu = backend
            .execute_forward(&plan, &input)
            .expect("wgpu transform execution");

        let cpu_plan = HilbertPlan::new(input.len()).expect("cpu plan");
        let cpu = cpu_plan
            .transform(
                &input
                    .iter()
                    .map(|&value| f64::from(value))
                    .collect::<Vec<_>>(),
            )
            .expect("cpu transform");

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 5.0e-4);
        }
    }

    #[test]
    fn rejects_invalid_lengths_before_dispatch() {
        let Ok(backend) = HilbertWgpuBackend::try_default() else {
            return;
        };
        let empty_err = backend
            .execute_forward(&HilbertWgpuPlan::new(0), &[])
            .expect_err("empty plan must fail");
        assert_eq!(
            empty_err,
            WgpuError::InvalidLength {
                len: 0,
                message: "length must be greater than zero",
            }
        );

        let mismatch_err = backend
            .execute_forward(&HilbertWgpuPlan::new(8), &[0.0; 4])
            .expect_err("length mismatch must fail");
        assert_eq!(
            mismatch_err,
            WgpuError::LengthMismatch {
                expected: 8,
                actual: 4,
            }
        );
    }

    #[test]
    fn typed_mixed_storage_matches_represented_f32_execution_when_device_exists() {
        let Ok(backend) = HilbertWgpuBackend::try_default() else {
            return;
        };
        use apollo_fft::{f16, PrecisionProfile};
        let represented = [1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
        let input: Vec<f16> = represented.iter().copied().map(f16::from_f32).collect();
        let represented_input: Vec<f32> = input.iter().map(|v| v.to_f64() as f32).collect();
        let plan = backend.plan(input.len());
        let expected_fwd = backend
            .execute_forward(&plan, &represented_input)
            .expect("represented forward");
        let mut typed_fwd = vec![f16::from_f32(0.0); input.len()];
        backend
            .execute_forward_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &input,
                &mut typed_fwd,
            )
            .expect("typed mixed forward");
        assert_eq!(typed_fwd.len(), expected_fwd.len());
        for (actual, expected) in typed_fwd.iter().zip(expected_fwd.iter()) {
            let expected_f16 = f16::from_f32(*expected);
            assert_eq!(actual.to_bits(), expected_f16.to_bits());
        }
    }

    #[test]
    fn typed_path_rejects_profile_mismatch_when_device_exists() {
        let Ok(backend) = HilbertWgpuBackend::try_default() else {
            return;
        };
        use apollo_fft::{f16, PrecisionProfile};
        let plan = backend.plan(8);
        let input = vec![f16::from_f32(1.0); 8];
        let mut output = vec![f16::from_f32(0.0); 8];
        let err = backend
            .execute_forward_typed_into(
                &plan,
                PrecisionProfile::LOW_PRECISION_F32,
                &input,
                &mut output,
            )
            .expect_err("profile mismatch must fail");
        assert_eq!(err, WgpuError::InvalidPrecisionProfile);
    }
}
