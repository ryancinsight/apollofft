//! WGPU value-semantic verification for the SDFT backend.

#[cfg(test)]
mod tests {
    use apollo_sdft::SdftPlan;

    use crate::{SdftWgpuBackend, SdftWgpuPlan, WgpuCapabilities, WgpuError};

    #[test]
    fn capabilities_reflect_forward_only_surface() {
        let capabilities = WgpuCapabilities::forward_only(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(!capabilities.supports_inverse);
    }

    #[test]
    fn plan_preserves_window_and_bin_parameters() {
        let plan = SdftWgpuPlan::new(16, 8);
        assert_eq!(plan.window_len(), 16);
        assert_eq!(plan.bin_count(), 8);
        assert_eq!(plan.len(), 16);
        assert!(!plan.is_empty());
        assert!(SdftWgpuPlan::new(0, 8).is_empty());
        assert!(SdftWgpuPlan::new(8, 0).is_empty());
    }

    #[test]
    fn unsupported_execution_error_identifies_operation() {
        let err = WgpuError::UnsupportedExecution {
            operation: "inverse",
        };
        assert_eq!(
            err.to_string(),
            "inverse is unsupported by the current WGPU capability set"
        );
    }

    #[test]
    fn backend_reports_forward_only_when_device_exists() {
        let Ok(backend) = SdftWgpuBackend::try_default() else {
            return;
        };
        let capabilities = backend.capabilities();
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(!capabilities.supports_inverse);
    }

    #[test]
    fn forward_matches_cpu_direct_bins_when_device_exists() {
        let Ok(backend) = SdftWgpuBackend::try_default() else {
            return;
        };
        let window_f32: [f32; 8] = [1.0, 0.5, -0.5, -1.0, 0.5, 1.0, -0.25, 0.75];
        let window_f64: Vec<f64> = window_f32.iter().map(|&x| f64::from(x)).collect();

        let plan = backend.plan(window_f32.len(), 4);
        let gpu = backend
            .execute_forward(&plan, &window_f32)
            .expect("wgpu sdft forward execution");

        let cpu_plan = SdftPlan::new(8, 4).expect("cpu sdft plan");
        let cpu = cpu_plan.direct_bins(&window_f64).expect("cpu direct bins");

        assert_eq!(gpu.len(), cpu.len(), "output bin count must match");
        for (index, (actual, expected)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let real_error = (f64::from(actual.re) - expected.re).abs();
            let imag_error = (f64::from(actual.im) - expected.im).abs();
            assert!(
                real_error < 1.0e-3 && imag_error < 1.0e-3,
                "sdft mismatch at bin {index}: gpu=({},{}) cpu=({},{}) re_err={real_error} im_err={imag_error}",
                actual.re, actual.im, expected.re, expected.im,
            );
        }
    }

    #[test]
    fn rejects_invalid_plan_and_input_before_dispatch() {
        let Ok(backend) = SdftWgpuBackend::try_default() else {
            return;
        };

        // Zero window_len must fail with InvalidPlan.
        let empty_err = backend
            .execute_forward(&SdftWgpuPlan::new(0, 4), &[])
            .expect_err("zero window_len must fail");
        assert_eq!(
            empty_err,
            WgpuError::InvalidPlan {
                window_len: 0,
                bin_count: 4,
                message: "window_len and bin_count must each be greater than zero",
            }
        );

        // Window length mismatch must fail with WindowLengthMismatch.
        let mismatch_err = backend
            .execute_forward(&SdftWgpuPlan::new(8, 4), &[0.0_f32; 4])
            .expect_err("window length mismatch must fail");
        assert_eq!(
            mismatch_err,
            WgpuError::WindowLengthMismatch {
                expected: 8,
                actual: 4
            }
        );
    }
}
