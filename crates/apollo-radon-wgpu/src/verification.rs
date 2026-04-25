//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use apollo_radon::RadonPlan;
    use ndarray::array;

    use crate::{RadonWgpuBackend, RadonWgpuPlan, WgpuCapabilities, WgpuError};

    #[test]
    fn capabilities_reflect_forward_only_kernel_surface() {
        let capabilities = WgpuCapabilities::forward_only(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(!capabilities.supports_inverse);
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
    fn backend_reports_forward_only_when_device_exists() {
        let Ok(backend) = RadonWgpuBackend::try_default() else {
            return;
        };
        let capabilities = backend.capabilities();
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(!capabilities.supports_inverse);
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
