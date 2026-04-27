//! WGPU value-semantic verification for the FrFT GPU backend.

#[cfg(test)]
mod tests {
    use apollo_fft::{f16, PrecisionProfile};
    use num_complex::{Complex32, Complex64};

    use crate::{FrftWgpuBackend, FrftWgpuPlan, WgpuCapabilities, WgpuError};

    #[test]
    fn capabilities_reflect_implemented_kernel_surface() {
        let capabilities = WgpuCapabilities::implemented(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
        assert!(capabilities.supports_mixed_precision);
        assert_eq!(
            capabilities.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
    }

    #[test]
    fn plan_preserves_logical_length() {
        let plan = FrftWgpuPlan::new(64, 1.0_f32);
        assert_eq!(plan.len(), 64);
        assert_eq!(plan.order(), 1.0_f32);
        assert!(!FrftWgpuPlan::new(64, 1.0).is_empty());
        assert!(FrftWgpuPlan::new(0, 1.0).is_empty());
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
        let Ok(backend) = FrftWgpuBackend::try_default() else {
            return;
        };
        let caps = backend.capabilities();
        assert!(caps.device_available);
        assert!(caps.supports_forward);
        assert!(caps.supports_inverse);
    }

    #[test]
    fn forward_at_order_zero_is_identity_when_device_exists() {
        let Ok(backend) = FrftWgpuBackend::try_default() else {
            return;
        };
        let n = 8_usize;
        let input: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(i as f32 * 0.1_f32, -(i as f32) * 0.05_f32))
            .collect();
        let plan = FrftWgpuPlan::new(n, 0.0_f32);
        let output = backend
            .execute_forward(&plan, &input)
            .expect("forward order 0");
        assert_eq!(output.len(), n);
        for (k, (actual, expected)) in output.iter().zip(input.iter()).enumerate() {
            assert!(
                (actual.re - expected.re).abs() < 1.0e-6_f32,
                "k={} identity re: got={} want={}",
                k,
                actual.re,
                expected.re
            );
            assert!(
                (actual.im - expected.im).abs() < 1.0e-6_f32,
                "k={} identity im: got={} want={}",
                k,
                actual.im,
                expected.im
            );
        }
    }

    #[test]
    fn forward_at_order_one_matches_cpu_frft_when_device_exists() {
        let Ok(backend) = FrftWgpuBackend::try_default() else {
            return;
        };
        let n = 16_usize;
        let input_f32: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new((i as f32 * 0.31_f32).sin(), 0.0_f32))
            .collect();
        let input_f64 = ndarray::Array1::from_vec(
            input_f32
                .iter()
                .map(|v| Complex64::new(v.re as f64, v.im as f64))
                .collect(),
        );
        let plan = FrftWgpuPlan::new(n, 1.0_f32);
        let gpu = backend
            .execute_forward(&plan, &input_f32)
            .expect("gpu forward order 1");
        let cpu = apollo_frft::frft(&input_f64, 1.0_f64).expect("cpu frft order 1");
        assert_eq!(gpu.len(), n);
        for (k, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert!(
                (g.re as f64 - c.re).abs() < 1.0e-3_f64,
                "k={} re: gpu={} cpu={}",
                k,
                g.re,
                c.re
            );
            assert!(
                (g.im as f64 - c.im).abs() < 1.0e-3_f64,
                "k={} im: gpu={} cpu={}",
                k,
                g.im,
                c.im
            );
        }
    }

    /// Validates GPU general chirp kernel (mode=4) against CPU reference.
    /// The direct O(N^2) discrete FrFT is non-unitary for non-integer orders
    /// so we compare GPU to CPU (not roundtrip) at tolerance 1e-3.
    #[test]
    fn general_order_matches_cpu_frft_when_device_exists() {
        let Ok(backend) = FrftWgpuBackend::try_default() else {
            return;
        };
        let n = 8_usize;
        let order_f32 = 0.5_f32;
        let input_f32: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new((i as f32 * 0.4_f32).cos(), (i as f32 * 0.3_f32).sin()))
            .collect();
        let input_f64 = ndarray::Array1::from_vec(
            input_f32
                .iter()
                .map(|v| Complex64::new(v.re as f64, v.im as f64))
                .collect(),
        );
        let plan = FrftWgpuPlan::new(n, order_f32);
        let gpu = backend
            .execute_forward(&plan, &input_f32)
            .expect("gpu general frft");
        let cpu = apollo_frft::frft(&input_f64, order_f32 as f64).expect("cpu general frft");
        assert_eq!(gpu.len(), n);
        for (k, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert!(
                (g.re as f64 - c.re).abs() < 1.0e-3_f64,
                "k={} re: gpu={:.6} cpu={:.6}",
                k,
                g.re,
                c.re
            );
            assert!(
                (g.im as f64 - c.im).abs() < 1.0e-3_f64,
                "k={} im: gpu={:.6} cpu={:.6}",
                k,
                g.im,
                c.im
            );
        }
    }

    /// Typed mixed [f16; 2] storage forward dispatch must produce output bits
    /// identical to f16::from_f32 applied to the f32 GPU result.
    #[test]
    fn typed_mixed_storage_frft_matches_represented_f32_when_device_exists() {
        let Ok(backend) = FrftWgpuBackend::try_default() else {
            return;
        };
        let n = 8_usize;
        let order = 0.5_f32;
        // Build [f16; 2] input from well-conditioned f32 source values.
        let source_re = [0.5_f32, -1.0, 2.0, 0.25, -0.5, 1.5, 0.0, -0.75];
        let source_im = [-0.25_f32, 0.75, -1.5, 0.5, 1.0, -0.25, 0.125, -1.0];
        let input: Vec<[f16; 2]> = source_re
            .iter()
            .zip(source_im.iter())
            .map(|(&re, &im)| [f16::from_f32(re), f16::from_f32(im)])
            .collect();
        // Represented f32 form is the round-trip through f16 quantization.
        let represented_f32: Vec<Complex32> = input
            .iter()
            .map(|[re, im]| Complex32::new(re.to_f32(), im.to_f32()))
            .collect();
        let plan = FrftWgpuPlan::new(n, order);
        let expected = backend
            .execute_forward(&plan, &represented_f32)
            .expect("f32 forward reference");
        let mut typed_output = vec![[f16::from_f32(0.0); 2]; n];
        backend
            .execute_forward_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &input,
                &mut typed_output,
            )
            .expect("typed mixed forward");
        for (actual, expected) in typed_output.iter().zip(expected.iter()) {
            let expected_re = f16::from_f32(expected.re);
            let expected_im = f16::from_f32(expected.im);
            assert_eq!(
                actual[0].to_bits(),
                expected_re.to_bits(),
                "re bits mismatch: actual={}, expected={}",
                actual[0].to_f32(),
                expected_re.to_f32()
            );
            assert_eq!(
                actual[1].to_bits(),
                expected_im.to_bits(),
                "im bits mismatch: actual={}, expected={}",
                actual[1].to_f32(),
                expected_im.to_f32()
            );
        }
    }

    /// Profile mismatch: [f16; 2] storage requires MIXED_PRECISION_F16_F32;
    /// supplying LOW_PRECISION_F32 must return InvalidPrecisionProfile.
    #[test]
    fn typed_path_rejects_profile_storage_mismatch_when_device_exists() {
        let Ok(backend) = FrftWgpuBackend::try_default() else {
            return;
        };
        let plan = FrftWgpuPlan::new(2, 0.5_f32);
        let input = [
            [f16::from_f32(1.0), f16::from_f32(0.0)],
            [f16::from_f32(-1.0), f16::from_f32(0.5)],
        ];
        let mut output = [[f16::from_f32(0.0); 2]; 2];
        let error = backend
            .execute_forward_typed_into(
                &plan,
                PrecisionProfile::LOW_PRECISION_F32,
                &input,
                &mut output,
            )
            .expect_err("profile mismatch must fail");
        assert_eq!(error, WgpuError::InvalidPrecisionProfile);
    }

    /// Roundtrip test using integer order=1: FrFT(1)=centred-DFT (mode=1) and
    /// FrFT(-1)=centred-IDFT (mode=3) form an exactly unitary pair, so
    /// inverse(forward(x, 1), 1) = x to within f32 machine precision.
    #[test]
    fn inverse_recovers_input_when_device_exists() {
        let Ok(backend) = FrftWgpuBackend::try_default() else {
            return;
        };
        let n = 16_usize;
        let order = 1.0_f32;
        let input: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new((i as f32 * 0.31_f32).sin(), (i as f32 * 0.17_f32).cos()))
            .collect();
        let plan = FrftWgpuPlan::new(n, order);
        let fwd = backend
            .execute_forward(&plan, &input)
            .expect("forward for roundtrip");
        let recovered = backend
            .execute_inverse(&plan, &fwd)
            .expect("inverse for roundtrip");
        assert_eq!(recovered.len(), n);
        for (k, (actual, expected)) in recovered.iter().zip(input.iter()).enumerate() {
            assert!(
                (actual.re - expected.re).abs() < 1.0e-3_f32,
                "roundtrip k={} re: got={:.6} want={:.6}",
                k,
                actual.re,
                expected.re
            );
            assert!(
                (actual.im - expected.im).abs() < 1.0e-3_f32,
                "roundtrip k={} im: got={:.6} want={:.6}",
                k,
                actual.im,
                expected.im
            );
        }
    }
}
