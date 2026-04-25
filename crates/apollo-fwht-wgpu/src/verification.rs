//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use crate::{FwhtWgpuBackend, FwhtWgpuPlan, WgpuCapabilities, WgpuError};
    use apollo_fwht::FwhtPlan;
    use ndarray::Array1;

    #[test]
    fn capabilities_reflect_implemented_kernel_surface() {
        let capabilities = WgpuCapabilities::implemented(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
    }

    #[test]
    fn plan_preserves_logical_length() {
        let plan = FwhtWgpuPlan::new(64);
        assert_eq!(plan.len(), 64);
        assert!(!FwhtWgpuPlan::new(64).is_empty());
        assert!(FwhtWgpuPlan::new(0).is_empty());
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
        let Ok(backend) = FwhtWgpuBackend::try_default() else {
            return;
        };
        let capabilities = backend.capabilities();
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
    }

    #[test]
    fn forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = FwhtWgpuBackend::try_default() else {
            return;
        };
        let input = vec![1.0_f32, -2.0, 3.5, 0.25, -1.5, 2.0, 0.0, 4.0];
        let gpu_plan = backend.plan(input.len());
        let gpu = backend
            .execute_forward(&gpu_plan, &input)
            .expect("wgpu forward execution");

        let cpu_plan = FwhtPlan::new(input.len()).expect("cpu plan");
        let cpu_input = Array1::from_vec(input.iter().map(|&value| value as f64).collect());
        let cpu = cpu_plan.forward(&cpu_input).expect("cpu forward");

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn inverse_recovers_input_when_device_exists() {
        let Ok(backend) = FwhtWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.5_f32, -1.25, 2.75, 4.0, -3.5, 1.0, 0.25, -0.125];
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
        let Ok(backend) = FwhtWgpuBackend::try_default() else {
            return;
        };
        let empty_err = backend
            .execute_forward(&FwhtWgpuPlan::new(0), &[])
            .expect_err("empty plan must fail");
        assert_eq!(
            empty_err,
            WgpuError::InvalidLength {
                len: 0,
                message: "length must be greater than zero",
            }
        );

        let non_power_err = backend
            .execute_forward(&FwhtWgpuPlan::new(6), &[0.0; 6])
            .expect_err("non-power-of-two plan must fail");
        assert_eq!(
            non_power_err,
            WgpuError::InvalidLength {
                len: 6,
                message: "length must be a power of two",
            }
        );

        let mismatch_err = backend
            .execute_forward(&FwhtWgpuPlan::new(8), &[0.0; 4])
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
