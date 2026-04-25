//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use apollo_qft::{QftPlan, QuantumStateDimension};
    use ndarray::Array1;
    use num_complex::{Complex32, Complex64};

    use crate::{QftWgpuBackend, QftWgpuPlan, WgpuCapabilities, WgpuError};

    #[test]
    fn capabilities_reflect_direct_unitary_kernel_surface() {
        let capabilities = WgpuCapabilities::direct_unitary(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
    }

    #[test]
    fn plan_preserves_logical_length() {
        let plan = QftWgpuPlan::new(64);
        assert_eq!(plan.len(), 64);
        assert!(!QftWgpuPlan::new(64).is_empty());
        assert!(QftWgpuPlan::new(0).is_empty());
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
        let Ok(backend) = QftWgpuBackend::try_default() else {
            return;
        };
        let capabilities = backend.capabilities();
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
    }

    #[test]
    fn forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = QftWgpuBackend::try_default() else {
            return;
        };
        let input = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(-0.5, 0.75),
            Complex32::new(0.25, -1.25),
            Complex32::new(2.0, 0.5),
        ];
        let plan = backend.plan(input.len());
        let gpu = backend
            .execute_forward(&plan, &input)
            .expect("wgpu forward execution");

        let cpu_plan = QftPlan::new(QuantumStateDimension::new(input.len()).expect("dimension"));
        let cpu_input = Array1::from_vec(
            input
                .iter()
                .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
                .collect(),
        );
        let cpu = cpu_plan.forward(&cpu_input).expect("cpu forward");

        assert_eq!(gpu.len(), cpu.len());
        for (index, (actual, expected)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let real_error = (f64::from(actual.re) - expected.re).abs();
            let imag_error = (f64::from(actual.im) - expected.im).abs();
            assert!(
                real_error < 2.0e-4 && imag_error < 2.0e-4,
                "forward mismatch at index {index}: actual=({},{}) expected=({},{}) real_error={} imag_error={}",
                actual.re,
                actual.im,
                expected.re,
                expected.im,
                real_error,
                imag_error
            );
        }
    }

    #[test]
    fn inverse_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = QftWgpuBackend::try_default() else {
            return;
        };
        let input = vec![
            Complex32::new(0.25, -0.5),
            Complex32::new(1.0, 1.5),
            Complex32::new(-2.0, 0.25),
            Complex32::new(0.75, -1.0),
        ];
        let plan = backend.plan(input.len());
        let gpu = backend
            .execute_inverse(&plan, &input)
            .expect("wgpu inverse execution");

        let cpu_plan = QftPlan::new(QuantumStateDimension::new(input.len()).expect("dimension"));
        let cpu_input = Array1::from_vec(
            input
                .iter()
                .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
                .collect(),
        );
        let cpu = cpu_plan.inverse(&cpu_input).expect("cpu inverse");

        assert_eq!(gpu.len(), cpu.len());
        for (index, (actual, expected)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let real_error = (f64::from(actual.re) - expected.re).abs();
            let imag_error = (f64::from(actual.im) - expected.im).abs();
            assert!(
                real_error < 2.0e-4 && imag_error < 2.0e-4,
                "inverse mismatch at index {index}: actual=({},{}) expected=({},{}) real_error={} imag_error={}",
                actual.re,
                actual.im,
                expected.re,
                expected.im,
                real_error,
                imag_error
            );
        }
    }

    #[test]
    fn inverse_recovers_forward_input_when_device_exists() {
        let Ok(backend) = QftWgpuBackend::try_default() else {
            return;
        };
        let input = vec![
            Complex32::new(0.5, -0.25),
            Complex32::new(-1.25, 0.75),
            Complex32::new(2.0, 1.0),
            Complex32::new(-0.5, -1.5),
        ];
        let plan = backend.plan(input.len());
        let transformed = backend
            .execute_forward(&plan, &input)
            .expect("wgpu forward execution");
        let recovered = backend
            .execute_inverse(&plan, &transformed)
            .expect("wgpu inverse execution");

        assert_eq!(recovered.len(), input.len());
        for (index, (actual, expected)) in recovered.iter().zip(input.iter()).enumerate() {
            let real_error = (actual.re - expected.re).abs();
            let imag_error = (actual.im - expected.im).abs();
            assert!(
                real_error < 5.0e-4 && imag_error < 5.0e-4,
                "roundtrip mismatch at index {index}: actual=({},{}) expected=({},{}) real_error={} imag_error={}",
                actual.re,
                actual.im,
                expected.re,
                expected.im,
                real_error,
                imag_error
            );
        }
    }

    #[test]
    fn rejects_invalid_plan_and_length_mismatch_before_dispatch() {
        let Ok(backend) = QftWgpuBackend::try_default() else {
            return;
        };
        let invalid_err = backend
            .execute_forward(&QftWgpuPlan::new(0), &[])
            .expect_err("zero-length plan must fail");
        assert_eq!(
            invalid_err,
            WgpuError::InvalidPlan {
                len: 0,
                message: "transform length must be greater than zero",
            }
        );

        let mismatch_err = backend
            .execute_forward(
                &QftWgpuPlan::new(4),
                &[Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)],
            )
            .expect_err("length mismatch must fail");
        assert_eq!(
            mismatch_err,
            WgpuError::InputLengthMismatch {
                expected: 4,
                actual: 2,
            }
        );
    }
}
