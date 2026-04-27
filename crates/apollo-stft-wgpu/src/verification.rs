//! Value-semantic verification for the STFT WGPU backend.

#[cfg(test)]
mod tests {
    use crate::{StftWgpuBackend, StftWgpuPlan, WgpuCapabilities, WgpuError};

    // -----------------------------------------------------------------------
    // Structural / plan tests (no GPU required)
    // -----------------------------------------------------------------------

    #[test]
    fn capabilities_reflect_forward_only_surface() {
        let caps = WgpuCapabilities::forward_only(true);
        assert!(caps.device_available);
        assert!(caps.supports_forward);
        assert!(!caps.supports_inverse);
        assert!(!caps.supports_mixed_precision);
        assert_eq!(
            caps.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
        let caps_off = WgpuCapabilities::forward_only(false);
        assert!(!caps_off.device_available);
        assert!(!caps_off.supports_forward);
        assert!(!caps_off.supports_inverse);
        assert!(!caps_off.supports_mixed_precision);
        assert_eq!(
            caps_off.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
    }

    #[test]
    fn plan_preserves_frame_and_hop_length() {
        let plan = StftWgpuPlan::new(8, 4);
        assert_eq!(plan.frame_len(), 8);
        assert_eq!(plan.hop_len(), 4);
        assert_eq!(plan.len(), 8);
        assert!(!plan.is_empty());
        assert!(StftWgpuPlan::new(0, 4).is_empty());
        assert!(StftWgpuPlan::new(8, 0).is_empty());
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
    fn rejects_invalid_plan_before_dispatch() {
        match StftWgpuBackend::try_default() {
            Err(_) => return, // no GPU: skip GPU-dependent part
            Ok(backend) => {
                // zero frame_len
                let r = backend.execute_forward(&StftWgpuPlan::new(0, 4), &[0.0f32; 8]);
                assert!(matches!(r, Err(WgpuError::InvalidPlan { .. })), "{r:?}");

                // zero hop_len
                let r = backend.execute_forward(&StftWgpuPlan::new(8, 0), &[0.0f32; 8]);
                assert!(matches!(r, Err(WgpuError::InvalidPlan { .. })), "{r:?}");

                // hop > frame
                let r = backend.execute_forward(&StftWgpuPlan::new(4, 8), &[0.0f32; 4]);
                assert!(matches!(r, Err(WgpuError::InvalidPlan { .. })), "{r:?}");

                // signal too short
                let r = backend.execute_forward(&StftWgpuPlan::new(8, 4), &[0.0f32; 4]);
                assert!(matches!(r, Err(WgpuError::InputTooShort { .. })), "{r:?}");
            }
        }
    }

    #[test]
    fn backend_reports_forward_only_when_device_exists() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        let caps = backend.capabilities();
        assert!(caps.device_available);
        assert!(caps.supports_forward);
        assert!(!caps.supports_inverse);
    }

    #[test]
    fn execute_inverse_returns_unsupported() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        let plan = StftWgpuPlan::new(8, 4);
        let result = backend.execute_inverse(&plan, &[]);
        assert!(
            matches!(
                result,
                Err(WgpuError::UnsupportedExecution {
                    operation: "inverse"
                })
            ),
            "{result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // GPU parity test: forward matches CPU StftPlan::forward
    // -----------------------------------------------------------------------

    #[test]
    fn forward_matches_cpu_reference_when_device_exists() {
        use ndarray::Array1;
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };

        // 16-sample alternating signal: values 0, 1, 0, -1, ...
        let signal_f32: Vec<f32> = vec![
            0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0,
        ];
        let signal_f64: Array1<f64> =
            Array1::from_vec(signal_f32.iter().map(|x| *x as f64).collect());

        let plan = StftWgpuPlan::new(8, 4);

        let gpu_out = backend
            .execute_forward(&plan, &signal_f32)
            .expect("GPU forward STFT");

        let cpu_plan = apollo_stft::StftPlan::new(8, 4).expect("CPU plan");
        let cpu_out = cpu_plan.forward(&signal_f64).expect("CPU forward STFT");

        assert_eq!(
            gpu_out.len(),
            cpu_out.len(),
            "output length mismatch: gpu={}, cpu={}",
            gpu_out.len(),
            cpu_out.len()
        );

        const TOL: f32 = 1e-3;
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let re_err = (g.re - c.re as f32).abs();
            let im_err = (g.im - c.im as f32).abs();
            assert!(
                re_err < TOL,
                "re mismatch at index {i}: gpu={:.6}, cpu={:.6}, err={:.2e}",
                g.re,
                c.re,
                re_err
            );
            assert!(
                im_err < TOL,
                "im mismatch at index {i}: gpu={:.6}, cpu={:.6}, err={:.2e}",
                g.im,
                c.im,
                im_err
            );
        }
    }
}
