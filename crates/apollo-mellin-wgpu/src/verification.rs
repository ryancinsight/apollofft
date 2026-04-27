//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use apollo_mellin::MellinPlan;

    use crate::{MellinWgpuBackend, MellinWgpuPlan, WgpuCapabilities, WgpuError};

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
    fn plan_preserves_scale_configuration() {
        let plan = MellinWgpuPlan::new(64, 0.25_f64.to_bits(), 4.0_f64.to_bits());
        assert_eq!(plan.samples(), 64);
        assert_eq!(plan.min_scale(), 0.25);
        assert_eq!(plan.max_scale(), 4.0);
        assert!(!plan.is_empty());
        assert!(MellinWgpuPlan::new(0, 0.25_f64.to_bits(), 4.0_f64.to_bits()).is_empty());
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
        let Ok(backend) = MellinWgpuBackend::try_default() else {
            return;
        };
        let capabilities = backend.capabilities();
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(!capabilities.supports_inverse);
    }

    #[test]
    fn forward_spectrum_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = MellinWgpuBackend::try_default() else {
            return;
        };
        let signal = vec![1.0_f32, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
        let signal_min = 1.0_f64;
        let signal_max = 8.0_f64;
        let plan = backend.plan(8, 1.0, 8.0);
        let gpu = backend
            .execute_forward(&plan, &signal, signal_min, signal_max)
            .expect("wgpu forward execution");

        let cpu_plan = MellinPlan::new(8, 1.0, 8.0).expect("cpu plan");
        let cpu = cpu_plan
            .forward_spectrum(
                &signal
                    .iter()
                    .map(|&value| f64::from(value))
                    .collect::<Vec<_>>(),
                signal_min,
                signal_max,
            )
            .expect("cpu forward");

        assert_eq!(gpu.len(), cpu.values().len());
        for (actual, expected) in gpu.iter().zip(cpu.values().iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 5.0e-4);
            assert!((f64::from(actual.im) - expected.im).abs() < 5.0e-4);
        }
    }

    #[test]
    fn rejects_invalid_plan_and_signal_domain_before_dispatch() {
        let Ok(backend) = MellinWgpuBackend::try_default() else {
            return;
        };

        let invalid_plan = backend
            .execute_forward(
                &MellinWgpuPlan::new(0, 1.0_f64.to_bits(), 8.0_f64.to_bits()),
                &[1.0],
                1.0,
                2.0,
            )
            .expect_err("empty plan must fail");
        assert_eq!(
            invalid_plan,
            WgpuError::InvalidPlan {
                samples: 0,
                min_scale: 1.0,
                max_scale: 8.0,
                message: "sample count must be greater than zero",
            }
        );

        let empty_signal = backend
            .execute_forward(
                &MellinWgpuPlan::new(8, 1.0_f64.to_bits(), 8.0_f64.to_bits()),
                &[],
                1.0,
                2.0,
            )
            .expect_err("empty signal must fail");
        assert_eq!(
            empty_signal,
            WgpuError::LengthMismatch {
                expected: 1,
                actual: 0,
            }
        );

        let invalid_domain = backend
            .execute_forward(
                &MellinWgpuPlan::new(8, 1.0_f64.to_bits(), 8.0_f64.to_bits()),
                &[1.0, 2.0, 3.0],
                2.0,
                1.0,
            )
            .expect_err("invalid signal domain must fail");
        assert_eq!(
            invalid_domain,
            WgpuError::InvalidSignalDomain {
                signal_min: 2.0,
                signal_max: 1.0,
                message: "signal_min must be less than signal_max",
            }
        );
    }
}
