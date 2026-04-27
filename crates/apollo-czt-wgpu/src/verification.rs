//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use apollo_czt::CztPlan;
    use ndarray::Array1;
    use num_complex::{Complex32, Complex64};

    use crate::{
        Complex32 as GpuComplex32, CztWgpuBackend, CztWgpuPlan, WgpuCapabilities, WgpuError,
    };

    #[test]
    fn capabilities_reflect_forward_only_kernel_surface() {
        let capabilities = WgpuCapabilities::forward_only(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(!capabilities.supports_inverse);
        assert!(!capabilities.supports_mixed_precision);
        assert_eq!(
            capabilities.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
    }

    #[test]
    fn plan_preserves_logical_parameters() {
        let plan = CztWgpuPlan::new(
            64,
            96,
            [1.0_f32.to_bits(), 0.5_f32.to_bits()],
            [0.9_f32.to_bits(), (-0.25_f32).to_bits()],
        );
        assert_eq!(plan.input_len(), 64);
        assert_eq!(plan.output_len(), 96);
        assert_eq!(plan.a(), GpuComplex32::new(1.0, 0.5));
        assert_eq!(plan.w(), GpuComplex32::new(0.9, -0.25));
        assert!(!plan.is_empty());
        assert!(CztWgpuPlan::new(0, 64, [0, 0], [0, 0]).is_empty());
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
        let Ok(backend) = CztWgpuBackend::try_default() else {
            return;
        };
        let capabilities = backend.capabilities();
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(!capabilities.supports_inverse);
    }

    #[test]
    fn forward_matches_cpu_direct_reference_when_device_exists() {
        let Ok(backend) = CztWgpuBackend::try_default() else {
            return;
        };
        let a32 = Complex32::new(0.95, 0.1);
        let w32 = Complex32::from_polar(1.0, -std::f32::consts::TAU / 9.0);
        let input = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(-0.5, 1.0),
            Complex32::new(0.25, -0.75),
            Complex32::new(1.25, 0.5),
        ];
        let gpu_plan = backend.plan(input.len(), 6, a32, w32);
        let gpu = backend
            .execute_forward(&gpu_plan, &input)
            .expect("wgpu forward execution");

        let cpu_plan = CztPlan::new(
            input.len(),
            6,
            Complex64::new(f64::from(a32.re), f64::from(a32.im)),
            Complex64::new(f64::from(w32.re), f64::from(w32.im)),
        )
        .expect("cpu plan");
        let cpu_input = Array1::from_vec(
            input
                .iter()
                .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
                .collect(),
        );
        let cpu = cpu_plan.forward_direct(&cpu_input).expect("cpu direct");

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu.iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 5.0e-4);
            assert!((f64::from(actual.im) - expected.im).abs() < 5.0e-4);
        }
    }

    #[test]
    fn rejects_invalid_lengths_and_parameters_before_dispatch() {
        let Ok(backend) = CztWgpuBackend::try_default() else {
            return;
        };
        let empty_err = backend
            .execute_forward(&CztWgpuPlan::new(0, 5, [0, 0], [0, 0]), &[])
            .expect_err("empty plan must fail");
        assert_eq!(
            empty_err,
            WgpuError::InvalidLength {
                input_len: 0,
                output_len: 5,
                message: "lengths must be greater than zero",
            }
        );

        let mismatch_err = backend
            .execute_forward(
                &CztWgpuPlan::new(
                    8,
                    8,
                    [1.0_f32.to_bits(), 0.0_f32.to_bits()],
                    [1.0_f32.to_bits(), 0.0_f32.to_bits()],
                ),
                &[Complex32::new(0.0, 0.0); 4],
            )
            .expect_err("length mismatch must fail");
        assert_eq!(
            mismatch_err,
            WgpuError::LengthMismatch {
                expected: 8,
                actual: 4,
            }
        );

        let invalid_param_err = backend
            .execute_forward(
                &CztWgpuPlan::new(
                    4,
                    4,
                    [0.0_f32.to_bits(), 0.0_f32.to_bits()],
                    [1.0_f32.to_bits(), 0.0_f32.to_bits()],
                ),
                &[Complex32::new(0.0, 0.0); 4],
            )
            .expect_err("zero a must fail");
        assert_eq!(
            invalid_param_err,
            WgpuError::InvalidParameters {
                message: "spiral parameters must have finite non-zero magnitude",
            }
        );
    }
}
