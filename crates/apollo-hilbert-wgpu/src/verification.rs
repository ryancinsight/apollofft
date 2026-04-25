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
}
