//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use apollo_ntt::{NttPlan, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT};
    use ndarray::Array1;

    use crate::{NttWgpuBackend, NttWgpuPlan, WgpuCapabilities, WgpuError};

    #[test]
    fn capabilities_reflect_full_kernel_surface() {
        let capabilities = WgpuCapabilities::full(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
        assert!(!capabilities.supports_mixed_precision);
        assert_eq!(
            capabilities.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
    }

    #[test]
    fn plan_preserves_modular_configuration() {
        let plan = NttWgpuPlan::new(64, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT);
        assert_eq!(plan.len(), 64);
        assert_eq!(plan.modulus(), DEFAULT_MODULUS);
        assert_eq!(plan.primitive_root(), DEFAULT_PRIMITIVE_ROOT);
        assert!(!NttWgpuPlan::new(64, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT).is_empty());
        assert!(NttWgpuPlan::new(0, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT).is_empty());
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
        let Ok(backend) = NttWgpuBackend::try_default() else {
            return;
        };
        let capabilities = backend.capabilities();
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
    }

    #[test]
    fn forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = NttWgpuBackend::try_default() else {
            return;
        };
        let input = vec![1_u64, 1, 2, 3, 5, 8, 13, 21];
        let plan = backend.plan(input.len());
        let gpu = backend
            .execute_forward(&plan, &input)
            .expect("wgpu forward execution");

        let cpu_plan = NttPlan::new(input.len()).expect("cpu plan");
        let cpu = cpu_plan
            .forward(&Array1::from_vec(input.clone()))
            .expect("cpu forward");

        assert_eq!(gpu, cpu.to_vec());
    }

    #[test]
    fn inverse_recovers_input_when_device_exists() {
        let Ok(backend) = NttWgpuBackend::try_default() else {
            return;
        };
        let input = vec![1_u64, 1, 2, 3, 5, 8, 13, 21];
        let plan = backend.plan(input.len());
        let spectrum = backend
            .execute_forward(&plan, &input)
            .expect("wgpu forward execution");
        let recovered = backend
            .execute_inverse(&plan, &spectrum)
            .expect("wgpu inverse execution");

        assert_eq!(recovered, input);
    }

    #[test]
    fn rejects_invalid_plan_and_length_before_dispatch() {
        let Ok(backend) = NttWgpuBackend::try_default() else {
            return;
        };
        let empty_err = backend
            .execute_forward(
                &NttWgpuPlan::new(0, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT),
                &[],
            )
            .expect_err("empty plan must fail");
        assert_eq!(
            empty_err,
            WgpuError::InvalidPlan {
                len: 0,
                modulus: DEFAULT_MODULUS,
                primitive_root: DEFAULT_PRIMITIVE_ROOT,
                message: "length must be greater than zero",
            }
        );

        let non_power_err = backend
            .execute_forward(
                &NttWgpuPlan::new(6, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT),
                &[0; 6],
            )
            .expect_err("non-power-of-two plan must fail");
        assert_eq!(
            non_power_err,
            WgpuError::InvalidPlan {
                len: 6,
                modulus: DEFAULT_MODULUS,
                primitive_root: DEFAULT_PRIMITIVE_ROOT,
                message: "length must be a power of two",
            }
        );

        let mismatch_err = backend
            .execute_forward(
                &NttWgpuPlan::new(8, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT),
                &[0; 4],
            )
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
