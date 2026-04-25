//! WGPU value-semantic verification for the GFT GPU backend.

#[cfg(test)]
mod tests {
    use crate::{GftWgpuBackend, GftWgpuPlan, WgpuCapabilities, WgpuError};

    #[test]
    fn capabilities_reflect_implemented_kernel_surface() {
        let capabilities = WgpuCapabilities::implemented(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
    }

    #[test]
    fn plan_preserves_logical_length() {
        let plan = GftWgpuPlan::new(4);
        assert_eq!(plan.len(), 4);
        assert!(!GftWgpuPlan::new(4).is_empty());
        assert!(GftWgpuPlan::new(0).is_empty());
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
        let Ok(backend) = GftWgpuBackend::try_default() else {
            return;
        };
        let caps = backend.capabilities();
        assert!(caps.device_available);
        assert!(caps.supports_forward);
        assert!(caps.supports_inverse);
    }

    /// Build the 4-node path graph CPU plan and extract basis as f32.
    /// Adjacency: [[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]]
    fn path4_plan_and_basis() -> (apollo_gft::GftPlan, Vec<f32>, Vec<f32>) {
        let adj = nalgebra::DMatrix::<f64>::from_row_slice(
            4,
            4,
            &[
                0.0_f64, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
        );
        let cpu_plan = apollo_gft::GftPlan::from_adjacency(&adj).expect("path-4 gft plan");
        let basis_f32: Vec<f32> = cpu_plan.basis().iter().map(|&v| v as f32).collect();
        let signal_f32 = vec![1.0_f32, -0.5, 2.0, 0.5];
        (cpu_plan, basis_f32, signal_f32)
    }

    #[test]
    fn forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = GftWgpuBackend::try_default() else {
            return;
        };
        let (cpu_plan, basis_f32, signal_f32) = path4_plan_and_basis();
        let gpu_plan = GftWgpuPlan::new(4);
        let gpu_fwd = backend
            .execute_forward(&gpu_plan, &signal_f32, &basis_f32)
            .expect("gft forward");
        let signal_f64 = ndarray::Array1::from_vec(signal_f32.iter().map(|&v| v as f64).collect());
        let cpu_fwd = cpu_plan.forward(&signal_f64).expect("cpu gft forward");
        assert_eq!(gpu_fwd.len(), 4);
        for (k, (g, c)) in gpu_fwd.iter().zip(cpu_fwd.iter()).enumerate() {
            assert!(
                (*g as f64 - c).abs() < 1.0e-3_f64,
                "forward k={}: gpu={} cpu={}",
                k,
                g,
                c
            );
        }
    }

    #[test]
    fn inverse_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = GftWgpuBackend::try_default() else {
            return;
        };
        let (cpu_plan, basis_f32, signal_f32) = path4_plan_and_basis();
        let signal_f64 = ndarray::Array1::from_vec(signal_f32.iter().map(|&v| v as f64).collect());
        // Use the CPU forward spectrum as input for the inverse.
        let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("cpu spectrum");
        let spectrum_f32: Vec<f32> = cpu_spectrum.iter().map(|&v| v as f32).collect();
        let gpu_plan = GftWgpuPlan::new(4);
        let gpu_inv = backend
            .execute_inverse(&gpu_plan, &spectrum_f32, &basis_f32)
            .expect("gft inverse");
        let cpu_inv = cpu_plan.inverse(&cpu_spectrum).expect("cpu inverse");
        assert_eq!(gpu_inv.len(), 4);
        for (k, (g, c)) in gpu_inv.iter().zip(cpu_inv.iter()).enumerate() {
            assert!(
                (*g as f64 - c).abs() < 1.0e-3_f64,
                "inverse k={}: gpu={} cpu={}",
                k,
                g,
                c
            );
        }
    }

    #[test]
    fn roundtrip_recovers_signal_when_device_exists() {
        let Ok(backend) = GftWgpuBackend::try_default() else {
            return;
        };
        let (_cpu_plan, basis_f32, signal_f32) = path4_plan_and_basis();
        let gpu_plan = GftWgpuPlan::new(4);
        let fwd = backend
            .execute_forward(&gpu_plan, &signal_f32, &basis_f32)
            .expect("roundtrip forward");
        let recovered = backend
            .execute_inverse(&gpu_plan, &fwd, &basis_f32)
            .expect("roundtrip inverse");
        assert_eq!(recovered.len(), 4);
        for (k, (actual, expected)) in recovered.iter().zip(signal_f32.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-3_f32,
                "roundtrip k={}: got={} want={}",
                k,
                actual,
                expected
            );
        }
    }
}
