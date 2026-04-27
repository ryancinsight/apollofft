//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use apollo_dctdst::{DctDstPlan, RealTransformKind};

    use crate::{DctDstWgpuBackend, DctDstWgpuPlan, WgpuCapabilities, WgpuError};

    #[test]
    fn capabilities_reflect_full_kernel_surface() {
        let capabilities = WgpuCapabilities::full(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
        assert!(capabilities.supports_dct);
        assert!(capabilities.supports_dst);
        assert!(capabilities.supports_mixed_precision);
        assert_eq!(
            capabilities.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
    }

    #[test]
    fn plan_preserves_logical_length() {
        let plan = DctDstWgpuPlan::new(64, RealTransformKind::DctII);
        assert_eq!(plan.len(), 64);
        assert_eq!(plan.kind(), RealTransformKind::DctII);
        assert!(!DctDstWgpuPlan::new(64, RealTransformKind::DctIII).is_empty());
        assert!(DctDstWgpuPlan::new(0, RealTransformKind::DctII).is_empty());
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
    fn backend_reports_dct_and_dst_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let capabilities = backend.capabilities();
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
        assert!(capabilities.supports_dct);
        assert!(capabilities.supports_dst);
    }

    #[test]
    fn dct2_forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
        let gpu_plan = backend.plan(input.len(), RealTransformKind::DctII);
        let gpu = backend
            .execute_forward(&gpu_plan, &input)
            .expect("wgpu forward execution");

        let cpu_plan = DctDstPlan::new(input.len(), RealTransformKind::DctII).expect("cpu plan");
        let cpu = cpu_plan
            .forward(&input.iter().map(|&value| value as f64).collect::<Vec<_>>())
            .expect("cpu forward");

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dct3_forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.75_f32, -1.0, 2.5, -0.25, 3.0, 0.5];
        let gpu_plan = backend.plan(input.len(), RealTransformKind::DctIII);
        let gpu = backend
            .execute_forward(&gpu_plan, &input)
            .expect("wgpu forward execution");

        let cpu_plan = DctDstPlan::new(input.len(), RealTransformKind::DctIII).expect("cpu plan");
        let cpu = cpu_plan
            .forward(&input.iter().map(|&value| value as f64).collect::<Vec<_>>())
            .expect("cpu forward");

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dct2_inverse_recovers_input_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.25_f32, -1.25, 2.0, -0.5, 3.0, 1.5];
        let plan = backend.plan(input.len(), RealTransformKind::DctII);
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
    fn dst2_forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![1.0_f32, -0.5, 2.0, -1.5, 0.25, 3.0];
        let gpu_plan = backend.plan(input.len(), RealTransformKind::DstII);
        let gpu = backend
            .execute_forward(&gpu_plan, &input)
            .expect("wgpu forward execution");

        let cpu_plan = DctDstPlan::new(input.len(), RealTransformKind::DstII).expect("cpu plan");
        let cpu = cpu_plan
            .forward(&input.iter().map(|&value| value as f64).collect::<Vec<_>>())
            .expect("cpu forward");

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dst3_forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.25_f32, -1.0, 2.0, 0.5, -0.75, 1.25];
        let gpu_plan = backend.plan(input.len(), RealTransformKind::DstIII);
        let gpu = backend
            .execute_forward(&gpu_plan, &input)
            .expect("wgpu forward execution");

        let cpu_plan = DctDstPlan::new(input.len(), RealTransformKind::DstIII).expect("cpu plan");
        let cpu = cpu_plan
            .forward(&input.iter().map(|&value| value as f64).collect::<Vec<_>>())
            .expect("cpu forward");

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dst2_inverse_recovers_input_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.75_f32, -1.25, 0.5, 2.0, -0.5, 1.0];
        let plan = backend.plan(input.len(), RealTransformKind::DstII);
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
    fn typed_mixed_storage_dct2_matches_represented_f32_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        use apollo_fft::{f16, PrecisionProfile};

        let represented = [
            0.75_f32, -1.25_f32, 2.0_f32, -0.5_f32, 3.0_f32, 1.5_f32, 0.25_f32, -0.875_f32,
        ];
        let input: Vec<f16> = represented.iter().copied().map(f16::from_f32).collect();
        let represented_input: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
        let plan = backend.plan(input.len(), RealTransformKind::DctII);
        let expected_forward = backend
            .execute_forward(&plan, &represented_input)
            .expect("represented f32 forward");
        let mut typed_output = vec![f16::from_f32(0.0); input.len()];
        backend
            .execute_forward_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &input,
                &mut typed_output,
            )
            .expect("typed mixed forward");
        for (actual, expected) in typed_output.iter().zip(expected_forward.iter()) {
            let expected_quantized = f16::from_f32(*expected);
            assert_eq!(
                actual.to_bits(),
                expected_quantized.to_bits(),
                "f16 forward bit mismatch: actual={}, expected={}",
                actual.to_f32(),
                expected_quantized.to_f32()
            );
        }
    }

    #[test]
    fn typed_path_rejects_profile_storage_mismatch_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        use apollo_fft::PrecisionProfile;

        let plan = backend.plan(4, RealTransformKind::DctII);
        let input = [1.0_f32, -1.0_f32, 0.5_f32, -0.5_f32];
        let mut output = [0.0_f32; 4];
        let error = backend
            .execute_forward_typed_into(
                &plan,
                PrecisionProfile::HIGH_ACCURACY_F64,
                &input,
                &mut output,
            )
            .expect_err("profile mismatch must fail");
        assert_eq!(error, WgpuError::InvalidPrecisionProfile);
    }

    #[test]
    fn rejects_invalid_lengths() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let empty_err = backend
            .execute_forward(&DctDstWgpuPlan::new(0, RealTransformKind::DctII), &[])
            .expect_err("empty plan must fail");
        assert_eq!(
            empty_err,
            WgpuError::InvalidLength {
                len: 0,
                message: "length must be greater than zero",
            }
        );

        let mismatch_err = backend
            .execute_forward(&DctDstWgpuPlan::new(8, RealTransformKind::DctII), &[0.0; 4])
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
