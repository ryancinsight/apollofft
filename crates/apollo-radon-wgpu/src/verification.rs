//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use apollo_fft::{f16, PrecisionProfile};
    use apollo_radon::RadonPlan;
    use ndarray::{array, Array2};

    use crate::{RadonWgpuBackend, RadonWgpuPlan, WgpuCapabilities, WgpuError};

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
    fn capabilities_reflect_forward_and_inverse_surface() {
        let caps = WgpuCapabilities::forward_and_inverse(true);
        assert!(caps.device_available);
        assert!(caps.supports_forward);
        assert!(caps.supports_inverse);
        assert!(caps.supports_mixed_precision);
        assert_eq!(
            caps.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
        let caps_off = WgpuCapabilities::forward_and_inverse(false);
        assert!(!caps_off.device_available);
        assert!(!caps_off.supports_forward);
        assert!(!caps_off.supports_inverse);
    }

    #[test]
    fn plan_preserves_geometry_configuration() {
        let plan = RadonWgpuPlan::new(8, 9, 3, 11, 0.5_f64.to_bits());
        assert_eq!(plan.rows(), 8);
        assert_eq!(plan.cols(), 9);
        assert_eq!(plan.angle_count(), 3);
        assert_eq!(plan.detector_count(), 11);
        assert_eq!(plan.detector_spacing(), 0.5);
        assert!(!plan.is_empty());
        assert!(RadonWgpuPlan::new(0, 9, 3, 11, 0.5_f64.to_bits()).is_empty());
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
    fn backend_reports_forward_and_backproject_when_device_exists() {
        let Ok(backend) = RadonWgpuBackend::try_default() else {
            return;
        };
        let capabilities = backend.capabilities();
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
    }

    #[test]
    fn forward_projection_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = RadonWgpuBackend::try_default() else {
            return;
        };
        let image = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];
        let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
        let plan = backend.plan(3, 3, angles.len(), 5, 1.0);
        let gpu = backend
            .execute_forward(&plan, &image, &angles)
            .expect("wgpu forward execution");

        let cpu_plan = RadonPlan::new(
            3,
            3,
            angles.iter().map(|&angle| f64::from(angle)).collect(),
            5,
            1.0,
        )
        .expect("cpu plan");
        let cpu = cpu_plan
            .forward(&image.mapv(f64::from))
            .expect("cpu forward");

        assert_eq!(gpu.dim(), cpu.values().dim());
        for (index, (actual, expected)) in gpu.iter().zip(cpu.values().iter()).enumerate() {
            let error = (f64::from(*actual) - *expected).abs();
            assert!(
                error < 5.0e-4,
                "mismatch at linear index {index}: actual={}, expected={}, error={error}",
                actual,
                expected
            );
        }
    }

    #[test]
    fn backproject_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = RadonWgpuBackend::try_default() else {
            return;
        };
        let image_f64 = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let angles_f64: Vec<f64> = vec![
            0.0,
            std::f64::consts::FRAC_PI_4,
            std::f64::consts::FRAC_PI_2,
        ];
        let angles_f32: Vec<f32> = angles_f64.iter().map(|&a| a as f32).collect();
        // CPU forward then backproject (f64 reference path).
        let cpu_plan = RadonPlan::new(3, 3, angles_f64.clone(), 7, 1.0).expect("cpu plan");
        let sinogram_cpu = cpu_plan.forward(&image_f64).expect("cpu forward");
        let cpu_bp = cpu_plan
            .backproject(&sinogram_cpu)
            .expect("cpu backproject");
        // GPU backproject using f32 sinogram derived from the f64 CPU result.
        let sinogram_f32 = sinogram_cpu.values().mapv(|v| v as f32);
        let gpu_plan = backend.plan(3, 3, angles_f32.len(), 7, 1.0);
        let gpu_bp = backend
            .execute_inverse(&gpu_plan, &sinogram_f32, &angles_f32)
            .expect("gpu backproject");
        assert_eq!(gpu_bp.dim(), (3, 3));
        for ((r, c), gpu_val) in gpu_bp.indexed_iter() {
            let cpu_val = cpu_bp[(r, c)] as f32;
            let err = (gpu_val - cpu_val).abs();
            assert!(
                err < 5e-3,
                "mismatch at ({r},{c}): gpu={gpu_val:.6}, cpu={cpu_val:.6}, err={err:.2e}"
            );
        }
    }

    #[test]
    fn execute_inverse_rejects_sinogram_shape_mismatch() {
        let Ok(backend) = RadonWgpuBackend::try_default() else {
            return;
        };
        // Plan expects 2 angles × 5 detectors; supply 3 × 5.
        let plan = backend.plan(3, 3, 2, 5, 1.0);
        let wrong_sinogram = Array2::<f32>::zeros((3, 5));
        let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
        let err = backend
            .execute_inverse(&plan, &wrong_sinogram, &angles)
            .expect_err("sinogram shape mismatch must fail");
        assert_eq!(
            err,
            WgpuError::SinogramShapeMismatch {
                expected_angles: 2,
                expected_detectors: 5,
                actual_angles: 3,
                actual_detectors: 5,
            }
        );
    }

    #[test]
    fn typed_flat_mixed_storage_matches_represented_f32_execution_when_device_exists() {
        let Ok(backend) = RadonWgpuBackend::try_default() else {
            return;
        };
        let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
        let plan = backend.plan(3, 3, angles.len(), 5, 1.0);

        // Build flat f32 image matching the existing forward test.
        let flat_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // Quantize to f16 and recover represented f32 for the reference path.
        let flat_f16: Vec<f16> = flat_f32.iter().copied().map(f16::from_f32).collect();
        let represented_f32: Vec<f32> = flat_f16.iter().map(|v| v.to_f32()).collect();
        let image_represented = Array2::from_shape_vec((3, 3), represented_f32).expect("reshape");

        let expected = backend
            .execute_forward(&plan, &image_represented, &angles)
            .expect("represented f32 forward");
        let actual = backend
            .execute_forward_flat_typed(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &flat_f16,
                &angles,
            )
            .expect("typed flat mixed forward");

        assert_eq!(actual.dim(), expected.dim());
        for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (f64::from(*a) - f64::from(*e)).abs() < 0.1,
                "mismatch at index {index}: actual={a}, expected={e}"
            );
        }
    }

    #[test]
    fn typed_flat_path_rejects_profile_mismatch_when_device_exists() {
        let Ok(backend) = RadonWgpuBackend::try_default() else {
            return;
        };
        let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
        let plan = backend.plan(3, 3, angles.len(), 5, 1.0);
        let flat_f16: Vec<f16> = vec![f16::from_f32(1.0); plan.rows() * plan.cols()];

        // f16 carries MIXED_PRECISION_F16_F32; passing LOW_PRECISION_F32 must fail.
        let err = backend
            .execute_forward_flat_typed::<f16>(
                &plan,
                PrecisionProfile::LOW_PRECISION_F32,
                &flat_f16,
                &angles,
            )
            .expect_err("profile mismatch must fail");
        assert_eq!(err, WgpuError::InvalidPrecisionProfile);
    }

    #[test]
    fn rejects_invalid_plan_and_input_shape_before_dispatch() {
        let Ok(backend) = RadonWgpuBackend::try_default() else {
            return;
        };
        let empty_plan_err = backend
            .execute_forward(
                &RadonWgpuPlan::new(0, 3, 1, 3, 1.0_f64.to_bits()),
                &array![[1.0_f32]],
                &[0.0_f32],
            )
            .expect_err("empty plan must fail");
        assert_eq!(
            empty_plan_err,
            WgpuError::InvalidPlan {
                rows: 0,
                cols: 3,
                angle_count: 1,
                detector_count: 3,
                detector_spacing: 1.0,
                message: "geometry dimensions must be greater than zero",
            }
        );

        let shape_err = backend
            .execute_forward(
                &RadonWgpuPlan::new(3, 3, 1, 3, 1.0_f64.to_bits()),
                &array![[1.0_f32, 2.0]],
                &[0.0_f32],
            )
            .expect_err("image shape mismatch must fail");
        assert_eq!(
            shape_err,
            WgpuError::ImageShapeMismatch {
                expected_rows: 3,
                expected_cols: 3,
                actual_rows: 1,
                actual_cols: 2,
            }
        );

        let angle_err = backend
            .execute_forward(
                &RadonWgpuPlan::new(3, 3, 2, 3, 1.0_f64.to_bits()),
                &array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                &[0.0_f32],
            )
            .expect_err("angle mismatch must fail");
        assert_eq!(
            angle_err,
            WgpuError::AngleCountMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }
}
