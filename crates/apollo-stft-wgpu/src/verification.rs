//! Value-semantic verification for the STFT WGPU backend.

#[cfg(test)]
mod tests {
    use crate::{StftWgpuBackend, StftWgpuPlan, WgpuCapabilities, WgpuError};
    use apollo_fft::{f16, PrecisionProfile};
    use ndarray::Array1;
    use num_complex::Complex32;

    // -----------------------------------------------------------------------
    // Structural / plan tests (no GPU required)
    // -----------------------------------------------------------------------

    #[test]
    fn capabilities_reflect_forward_only_surface() {
        let caps = WgpuCapabilities::forward_only(true);
        assert!(caps.device_available);
        assert!(caps.supports_forward);
        assert!(!caps.supports_inverse);
        assert!(caps.supports_mixed_precision);
        assert_eq!(
            caps.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
        let caps_off = WgpuCapabilities::forward_only(false);
        assert!(!caps_off.device_available);
        assert!(!caps_off.supports_forward);
        assert!(!caps_off.supports_inverse);
        assert!(caps_off.supports_mixed_precision);
        assert_eq!(
            caps_off.default_precision_profile,
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
        let caps_off = WgpuCapabilities::forward_and_inverse(false);
        assert!(!caps_off.device_available);
        assert!(!caps_off.supports_forward);
        assert!(!caps_off.supports_inverse);
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
    fn backend_reports_forward_and_inverse_when_device_exists() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        let caps = backend.capabilities();
        assert!(caps.device_available);
        assert!(caps.supports_forward);
        assert!(caps.supports_inverse);
    }

    // -----------------------------------------------------------------------
    // GPU parity test: forward matches CPU StftPlan::forward
    // -----------------------------------------------------------------------

    #[test]
    fn forward_matches_cpu_reference_when_device_exists() {
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

    // -----------------------------------------------------------------------
    // GPU inverse: WOLA roundtrip tests
    // -----------------------------------------------------------------------

    /// frame_len=8, hop_len=4 (Hann COLA condition: hop = frame_len/2).
    /// CPU forward spectrum → GPU inverse → compare against CPU inverse as reference.
    #[test]
    fn inverse_roundtrip_recovers_cola_signal_when_device_exists() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };

        // Alternating signal, smooth, non-trivial, well inside f32 dynamic range.
        let signal_f32: Vec<f32> = vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        let signal_f64: Array1<f64> =
            Array1::from_vec(signal_f32.iter().map(|&x| x as f64).collect());
        let signal_len = 8usize;

        let plan = StftWgpuPlan::new(8, 4);

        // Compute spectrum on CPU (f64) as the authoritative input.
        // frame_count = 1 + 8.div_ceil(4) = 3; spectrum_len = 3 * 8 = 24.
        let cpu_plan = apollo_stft::StftPlan::new(8, 4).expect("CPU plan");
        let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("CPU forward");

        // Downcast spectrum to f32 for GPU inverse.
        let gpu_spectrum: Vec<Complex32> = cpu_spectrum
            .iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect();

        let recovered = backend
            .execute_inverse(&plan, &gpu_spectrum, signal_len)
            .expect("GPU inverse STFT");

        // CPU inverse as reference for value-semantic comparison.
        let cpu_recovered = cpu_plan
            .inverse(&cpu_spectrum, signal_len)
            .expect("CPU inverse");

        assert_eq!(
            recovered.len(),
            signal_len,
            "recovered length mismatch: got {}, expected {signal_len}",
            recovered.len()
        );
        assert_eq!(
            cpu_recovered.len(),
            signal_len,
            "cpu_recovered length mismatch"
        );

        // Tolerance accounts for f64→f32 downcast of spectrum and f32 GPU arithmetic.
        const TOL: f32 = 5e-4;
        for (i, (gpu_val, cpu_val)) in recovered.iter().zip(cpu_recovered.iter()).enumerate() {
            let err = (gpu_val - *cpu_val as f32).abs();
            assert!(
                err < TOL,
                "mismatch at {i}: gpu={gpu_val:.6}, cpu={cpu_val:.6}, err={err:.2e}"
            );
        }
    }

    /// Similar WOLA roundtrip test with a 16-sample signal and frame_len=8, hop_len=4.
    #[test]
    fn inverse_matches_cpu_reference_for_16sample_signal() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };

        let signal_f32: Vec<f32> = vec![
            0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0,
        ];
        let signal_f64: Array1<f64> =
            Array1::from_vec(signal_f32.iter().map(|&x| x as f64).collect());
        let signal_len = 16usize;

        let plan = StftWgpuPlan::new(8, 4);

        // frame_count = 1 + 16.div_ceil(4) = 5; spectrum_len = 5 * 8 = 40.
        let cpu_plan = apollo_stft::StftPlan::new(8, 4).expect("CPU plan");
        let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("CPU forward");

        let gpu_spectrum: Vec<Complex32> = cpu_spectrum
            .iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect();

        let recovered = backend
            .execute_inverse(&plan, &gpu_spectrum, signal_len)
            .expect("GPU inverse STFT");

        let cpu_recovered = cpu_plan
            .inverse(&cpu_spectrum, signal_len)
            .expect("CPU inverse");

        assert_eq!(
            recovered.len(),
            signal_len,
            "recovered length mismatch: got {}, expected {signal_len}",
            recovered.len()
        );

        const TOL: f32 = 5e-4;
        for (i, (gpu_val, cpu_val)) in recovered.iter().zip(cpu_recovered.iter()).enumerate() {
            let err = (gpu_val - *cpu_val as f32).abs();
            assert!(
                err < TOL,
                "mismatch at {i}: gpu={gpu_val:.6}, cpu={cpu_val:.6}, err={err:.2e}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Typed mixed-precision dispatch tests
    // -----------------------------------------------------------------------

    #[test]
    fn typed_mixed_storage_matches_represented_f32_execution_when_device_exists() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        // 8-sample signal: f16 quantization of the alternating pattern is exact for these values.
        let signal_f32: Vec<f32> = vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        let signal_f16: Vec<f16> = signal_f32.iter().map(|&x| f16::from_f32(x)).collect();
        // Represented input: f16 promoted back to f32 (round-trip defines the reference).
        let represented: Vec<f32> = signal_f16.iter().map(|v| v.to_f32()).collect();
        let plan = StftWgpuPlan::new(4, 2);
        // frame_count = 1 + 8.div_ceil(2) = 5; output_len = 5 * 4 = 20.
        let f32_result = backend
            .execute_forward(&plan, &represented)
            .expect("f32 reference");
        let mut typed_out: Vec<[f16; 2]> =
            vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; f32_result.len()];
        backend
            .execute_forward_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &signal_f16,
                &mut typed_out,
            )
            .expect("typed mixed forward");
        for (actual, expected) in typed_out.iter().zip(f32_result.iter()) {
            let expected_f16 = [f16::from_f32(expected.re), f16::from_f32(expected.im)];
            assert_eq!(
                actual[0].to_bits(),
                expected_f16[0].to_bits(),
                "re bits mismatch: actual={:?} expected={:?}",
                actual[0],
                expected_f16[0]
            );
            assert_eq!(
                actual[1].to_bits(),
                expected_f16[1].to_bits(),
                "im bits mismatch: actual={:?} expected={:?}",
                actual[1],
                expected_f16[1]
            );
        }
    }

    #[test]
    fn typed_path_rejects_profile_mismatch() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        // f16 signal: I::PROFILE == MIXED_PRECISION_F16_F32.
        // Passing LOW_PRECISION_F32 as input_precision must trigger InvalidPrecisionProfile
        // before any GPU work is attempted.
        let signal_f16: Vec<f16> = vec![
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(-1.0),
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(-1.0),
        ];
        let plan = StftWgpuPlan::new(4, 2);
        let frame_count = 1 + signal_f16.len().div_ceil(plan.hop_len());
        let output_len = frame_count * plan.frame_len();
        let mut out: Vec<[f16; 2]> = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; output_len];
        let error = backend
            .execute_forward_typed_into(
                &plan,
                PrecisionProfile::LOW_PRECISION_F32,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &signal_f16,
                &mut out,
            )
            .expect_err("profile mismatch must fail");
        assert_eq!(error, WgpuError::InvalidPrecisionProfile);
    }

    /// GPU inverse STFT WOLA roundtrip for three COLA-compliant parameter sets.
    ///
    /// COLA (Constant Overlap-Add) condition for Hann window: hop_len <= frame_len / 2.
    /// Exact COLA at 50% overlap: hop_len = frame_len / 2.
    ///
    /// For each parameter set, the test:
    /// 1. Computes the CPU STFT spectrum of the analytical signal (f64 reference).
    /// 2. Downcasts the spectrum to f32 for GPU inverse dispatch.
    /// 3. Asserts that GPU inverse output matches CPU inverse within TOL = 5e-3.
    ///    Tolerance: 5e-3 accounts for f64→f32 spectrum downcast (ε_f32 ≈ 1.2e-7 per
    ///    coefficient, amplified by the N-term IDFT sum).
    #[test]
    fn inverse_roundtrip_for_multiple_cola_parameter_sets() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };

        fn run_roundtrip(
            backend: &StftWgpuBackend,
            frame_len: usize,
            hop_len: usize,
            signal: &[f32],
        ) {
            let signal_len = signal.len();
            let plan = StftWgpuPlan::new(frame_len, hop_len);
            let cpu_plan = apollo_stft::StftPlan::new(frame_len, hop_len).expect("cpu plan");
            let signal_f64 = Array1::from_vec(signal.iter().map(|&x| x as f64).collect());
            let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("cpu forward");
            let gpu_spectrum: Vec<Complex32> = cpu_spectrum
                .iter()
                .map(|c| Complex32::new(c.re as f32, c.im as f32))
                .collect();
            let recovered = backend
                .execute_inverse(&plan, &gpu_spectrum, signal_len)
                .unwrap_or_else(|e| {
                    panic!("GPU inverse failed for frame_len={frame_len}, hop_len={hop_len}: {e}")
                });
            let cpu_recovered = cpu_plan
                .inverse(&cpu_spectrum, signal_len)
                .expect("cpu inverse");
            assert_eq!(
                recovered.len(),
                signal_len,
                "length mismatch (frame={frame_len}, hop={hop_len})"
            );
            const TOL: f32 = 5e-3;
            for (i, (g, c)) in recovered.iter().zip(cpu_recovered.iter()).enumerate() {
                let err = (g - *c as f32).abs();
                assert!(
                    err < TOL,
                    "mismatch at sample {i} (frame={frame_len}, hop={hop_len}): \
                     gpu={g:.6}, cpu={c:.6}, err={err:.2e}"
                );
            }
        }

        // Case 1: frame_len=8, hop_len=4 — canonical 50% overlap, single frame.
        run_roundtrip(
            &backend,
            8,
            4,
            &[0.5_f32, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
        );

        // Case 2: frame_len=16, hop_len=8 — 50% overlap, sine wave reference signal.
        let sig16: Vec<f32> = (0..16)
            .map(|i| ((i as f32) * std::f32::consts::FRAC_PI_4).sin())
            .collect();
        run_roundtrip(&backend, 16, 8, &sig16);

        // Case 3: frame_len=32, hop_len=16 — 50% overlap, cosine reference signal.
        let sig32: Vec<f32> = (0..32)
            .map(|i| ((i as f32) * std::f32::consts::PI / 8.0).cos())
            .collect();
        run_roundtrip(&backend, 32, 16, &sig32);
    }

    #[test]
    fn inverse_rejects_non_power_of_two_frame_len() {
        // frame_len = 6 is not a power of two; the FFT-accelerated inverse path
        // must return FrameLenNotPowerOfTwo without attempting GPU dispatch.
        //
        // frame_count = 1 + 6.div_ceil(3) = 3; spectrum_len = 3 * 6 = 18.
        match StftWgpuBackend::try_default() {
            Err(_) => return,
            Ok(backend) => {
                let plan = StftWgpuPlan::new(6, 3);
                let dummy_spectrum = vec![Complex32::new(0.0, 0.0); 18];
                let result = backend.execute_inverse(&plan, &dummy_spectrum, 6);
                match result {
                    Err(WgpuError::FrameLenNotPowerOfTwo { frame_len }) => {
                        assert_eq!(frame_len, 6, "reported frame_len must equal 6");
                    }
                    other => panic!(
                        "expected FrameLenNotPowerOfTwo for frame_len=6, got {:?}",
                        other
                    ),
                }
            }
        }
    }

    #[test]
    #[ignore = "requires wgpu device"]
    fn inverse_roundtrip_large_frame_1024_samples_when_device_exists() {
        // Verifies the FFT-accelerated inverse path for a large power-of-two
        // frame_len (1024) that exercises multiple butterfly stages (log₂(1024) = 10).
        //
        // Reference signal: analytic sine x[n] = sin(2π·n/SIGNAL_LEN).
        // Forward (CPU, f64) → Inverse (GPU, f32). Max absolute deviation ≤ 5e-3.
        const FRAME_LEN: usize = 1024;
        const HOP_LEN: usize = 512;
        const SIGNAL_LEN: usize = 8192;
        const TOL: f32 = 5e-3;

        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };

        let signal_f32: Vec<f32> = (0..SIGNAL_LEN)
            .map(|n| (std::f32::consts::TAU * n as f32 / SIGNAL_LEN as f32).sin())
            .collect();

        let plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);
        let cpu_plan = apollo_stft::StftPlan::new(FRAME_LEN, HOP_LEN).expect("cpu plan");
        let signal_f64 = Array1::from_vec(signal_f32.iter().map(|&x| x as f64).collect());
        let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("cpu forward");
        let gpu_spectrum: Vec<Complex32> = cpu_spectrum
            .iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect();

        let gpu_result = backend
            .execute_inverse(&plan, &gpu_spectrum, SIGNAL_LEN)
            .expect("GPU inverse must succeed for power-of-two frame_len");

        let cpu_reference = cpu_plan
            .inverse(&cpu_spectrum, SIGNAL_LEN)
            .expect("cpu inverse");

        assert_eq!(
            gpu_result.len(),
            SIGNAL_LEN,
            "output length must equal signal_len"
        );
        let max_err = gpu_result
            .iter()
            .zip(cpu_reference.iter())
            .map(|(g, c)| (g - *c as f32).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err <= TOL,
            "max |GPU - CPU_ref| = {max_err:.2e} exceeds TOL = {TOL:.2e} for frame_len={FRAME_LEN}"
        );
    }

    /// Verifies that `execute_forward` rejects a non-power-of-two `frame_len` with
    /// `WgpuError::FrameLenNotPowerOfTwo` before any GPU dispatch.
    /// CPU-side validation: does not require a GPU device.
    #[test]
    fn forward_rejects_non_power_of_two_frame_len() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        // frame_len = 6 is not a power of two; signal length satisfies InputTooShort check.
        let signal = vec![0.0f32; 12];
        let plan = StftWgpuPlan::new(6, 3);
        let r = backend.execute_forward(&plan, &signal);
        assert!(
            matches!(r, Err(WgpuError::FrameLenNotPowerOfTwo { frame_len: 6 })),
            "expected FrameLenNotPowerOfTwo {{ frame_len: 6 }}, got {r:?}"
        );
    }

    /// GPU forward FFT (1024-frame, log₂(1024) = 10 butterfly stages) followed by GPU inverse
    /// FFT must recover the original signal within tolerance on interior samples.
    ///
    /// Signal: sum of two bin-aligned sinusoids; hop = frame_len/2 satisfies Hann COLA.
    /// Tolerance: 1e-2 (f32 accumulation over 10 butterfly stages + WOLA normalisation).
    #[test]
    #[ignore = "requires wgpu device"]
    fn forward_fft_roundtrip_large_frame_when_device_exists() {
        const FRAME_LEN: usize = 1024;
        const HOP_LEN: usize = 512;
        const SIGNAL_LEN: usize = 4096;
        const TOL: f32 = 1e-2;

        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };

        // Analytical signal: two bin-aligned sinusoids (k₁=16, k₂=64).
        // Bin alignment ensures the DFT concentrates energy at known frequencies
        // with no spectral leakage, making the analytical spectrum exact.
        let signal_f32: Vec<f32> = (0..SIGNAL_LEN)
            .map(|n| {
                let t = n as f32;
                (2.0 * std::f32::consts::PI * 16.0 * t / FRAME_LEN as f32).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * 64.0 * t / FRAME_LEN as f32).sin()
            })
            .collect();

        let plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);
        let frame_count = 1 + SIGNAL_LEN.div_ceil(HOP_LEN);

        let spectrum = backend
            .execute_forward(&plan, &signal_f32)
            .expect("GPU forward FFT");
        assert_eq!(
            spectrum.len(),
            frame_count * FRAME_LEN,
            "spectrum length mismatch"
        );

        let recovered = backend
            .execute_inverse(&plan, &spectrum, SIGNAL_LEN)
            .expect("GPU inverse FFT");
        assert_eq!(recovered.len(), SIGNAL_LEN, "recovered length mismatch");

        // Exclude edge samples affected by zero-padding at frame boundaries.
        let margin = FRAME_LEN / 2;
        for i in margin..(SIGNAL_LEN - margin) {
            let err = (recovered[i] - signal_f32[i]).abs();
            assert!(
                err < TOL,
                "sample {i}: recovered={:.6}, expected={:.6}, err={:.2e}",
                recovered[i],
                signal_f32[i],
                err
            );
        }
    }
}
