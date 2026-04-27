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
        assert!(capabilities.supports_quantized_storage);
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
    fn quantized_u32_storage_matches_allocating_u64_execution_when_device_exists() {
        let Ok(backend) = NttWgpuBackend::try_default() else {
            return;
        };
        let input = vec![1_u32, 1, 2, 3, 5, 8, 13, 21];
        let input64: Vec<u64> = input.iter().map(|&value| u64::from(value)).collect();
        let plan = backend.plan(input.len());

        let expected_forward = backend
            .execute_forward(&plan, &input64)
            .expect("allocating forward");
        let mut quantized_forward = vec![0_u32; input.len()];
        backend
            .execute_forward_quantized_into(&plan, &input, &mut quantized_forward)
            .expect("quantized forward");
        assert_eq!(
            quantized_forward,
            expected_forward
                .iter()
                .map(|&value| value as u32)
                .collect::<Vec<_>>()
        );

        let mut quantized_inverse = vec![0_u32; input.len()];
        backend
            .execute_inverse_quantized_into(&plan, &quantized_forward, &mut quantized_inverse)
            .expect("quantized inverse");
        assert_eq!(quantized_inverse, input);
    }

    #[test]
    fn quantized_u32_reusable_buffers_match_allocating_quantized_path_when_device_exists() {
        let Ok(backend) = NttWgpuBackend::try_default() else {
            return;
        };
        let input = vec![3_u32, 1, 4, 1, 5, 9, 2, 6];
        let plan = backend.plan(input.len());
        let mut buffers = backend.create_buffers(&plan).expect("reusable buffers");

        let mut allocating_forward = vec![0_u32; input.len()];
        backend
            .execute_forward_quantized_into(&plan, &input, &mut allocating_forward)
            .expect("allocating quantized forward");
        backend
            .execute_forward_quantized_with_buffers(&plan, &input, &mut buffers)
            .expect("buffered quantized forward");
        let expected_forward: Vec<u64> = allocating_forward
            .iter()
            .map(|&value| u64::from(value))
            .collect();
        assert_eq!(backend.buffer_output(&buffers), expected_forward.as_slice());

        let spectrum = allocating_forward;
        let mut allocating_inverse = vec![0_u32; input.len()];
        backend
            .execute_inverse_quantized_into(&plan, &spectrum, &mut allocating_inverse)
            .expect("allocating quantized inverse");
        backend
            .execute_inverse_quantized_with_buffers(&plan, &spectrum, &mut buffers)
            .expect("buffered quantized inverse");
        let expected_inverse: Vec<u64> = allocating_inverse
            .iter()
            .map(|&value| u64::from(value))
            .collect();
        assert_eq!(backend.buffer_output(&buffers), expected_inverse.as_slice());
        assert_eq!(allocating_inverse, input);
    }

    #[test]
    fn quantized_u32_storage_rejects_output_length_mismatch_when_device_exists() {
        let Ok(backend) = NttWgpuBackend::try_default() else {
            return;
        };
        let plan = backend.plan(8);
        let mut output = vec![0_u32; 4];
        let err = backend
            .execute_forward_quantized_into(&plan, &[0; 8], &mut output)
            .expect_err("output length mismatch must fail");
        assert_eq!(
            err,
            WgpuError::OutputLengthMismatch {
                expected: 8,
                actual: 4,
            }
        );
    }

    #[test]
    fn reusable_buffers_match_allocating_forward_and_inverse_when_device_exists() {
        let Ok(backend) = NttWgpuBackend::try_default() else {
            return;
        };
        let input = vec![1_u64, 4, 9, 16, 25, 36, 49, 64];
        let plan = backend.plan(input.len());
        let mut buffers = backend
            .create_buffers(&plan)
            .expect("reusable buffers for plan");

        let allocating_forward = backend
            .execute_forward(&plan, &input)
            .expect("allocating forward");
        backend
            .execute_forward_with_buffers(&plan, &input, &mut buffers)
            .expect("buffered forward");
        assert_eq!(
            backend.buffer_output(&buffers),
            allocating_forward.as_slice()
        );

        let spectrum = backend.buffer_output(&buffers).to_vec();
        let allocating_inverse = backend
            .execute_inverse(&plan, &spectrum)
            .expect("allocating inverse");
        backend
            .execute_inverse_with_buffers(&plan, &spectrum, &mut buffers)
            .expect("buffered inverse");
        assert_eq!(
            backend.buffer_output(&buffers),
            allocating_inverse.as_slice()
        );
        assert_eq!(backend.buffer_output(&buffers), input.as_slice());
    }

    #[test]
    fn reusable_buffers_reject_plan_length_mismatch_when_device_exists() {
        let Ok(backend) = NttWgpuBackend::try_default() else {
            return;
        };
        let plan = backend.plan(8);
        let short_plan = backend.plan(4);
        let mut short_buffers = backend
            .create_buffers(&short_plan)
            .expect("short reusable buffers");
        let err = backend
            .execute_forward_with_buffers(&plan, &[0; 8], &mut short_buffers)
            .expect_err("buffer length mismatch must fail");
        assert_eq!(
            err,
            WgpuError::BufferLengthMismatch {
                expected: 8,
                actual: 4,
            }
        );
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
