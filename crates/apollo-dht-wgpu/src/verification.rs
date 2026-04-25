//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use crate::{DhtWgpuBackend, DhtWgpuPlan, WgpuCapabilities, WgpuError};
    use apollo_dht::DhtPlan;

    #[test]
    fn capabilities_reflect_implemented_kernel_surface() {
        let capabilities = WgpuCapabilities::implemented(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
    }

    #[test]
    fn plan_preserves_logical_length() {
        let plan = DhtWgpuPlan::new(64);
        assert_eq!(plan.len(), 64);
        assert!(!DhtWgpuPlan::new(64).is_empty());
        assert!(DhtWgpuPlan::new(0).is_empty());
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
    fn backend_reports_forward_and_inverse_when_device_exists() {
        let Ok(backend) = DhtWgpuBackend::try_default() else {
            return;
        };
        let capabilities = backend.capabilities();
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
    }

    #[test]
    fn forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = DhtWgpuBackend::try_default() else {
            return;
        };
        let input = vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
        let gpu_plan = backend.plan(input.len());
        let gpu = backend
            .execute_forward(&gpu_plan, &input)
            .expect("wgpu forward execution");

        let cpu_plan = DhtPlan::new(input.len()).expect("cpu plan");
        let cpu = cpu_plan
            .forward(&input.iter().map(|&value| value as f64).collect::<Vec<_>>())
            .expect("cpu forward");

        assert_eq!(gpu.len(), cpu.values().len());
        for (actual, expected) in gpu.iter().zip(cpu.values().iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn inverse_recovers_input_when_device_exists() {
        let Ok(backend) = DhtWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.75_f32, -1.25, 2.0, -0.5, 3.0, 1.5];
        let plan = backend.plan(input.len());
        let spectrum = backend
            .execute_forward(&plan, &input)
            .expect("wgpu forward execution");
        let recovered = backend
            .execute_inverse(&plan, &spectrum)
            .expect("wgpu inverse execution");

        assert_eq!(recovered.len(), input.len());
        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn rejects_invalid_plan_shape_before_dispatch() {
        let Ok(backend) = DhtWgpuBackend::try_default() else {
            return;
        };
        let empty_err = backend
            .execute_forward(&DhtWgpuPlan::new(0), &[])
            .expect_err("empty plan must fail");
        assert_eq!(
            empty_err,
            WgpuError::InvalidLength {
                len: 0,
                message: "length must be greater than zero",
            }
        );

        let mismatch_err = backend
            .execute_forward(&DhtWgpuPlan::new(8), &[0.0; 4])
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
