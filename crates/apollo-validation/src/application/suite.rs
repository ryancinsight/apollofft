//! Validation suite orchestration.

use crate::domain::report::{
    BenchmarkReport, CpuFftReport, EnvironmentReport, ExternalComparisonReport, GpuFftReport,
    NufftReport, ValidationReport,
};
use apollofft::{
    nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d, nufft_type1_3d_fast, nufft_type2_1d,
    nufft_type2_1d_fast, Complex64, FftBackend, FftPlan1D, FftPlan3D, Shape3D, UniformDomain1D,
    UniformGrid3D, DEFAULT_NUFFT_KERNEL_WIDTH,
};
use ndarray::{Array1, Array3};
use num_complex::Complex;
use rustfft::FftPlanner;
use std::path::PathBuf;
use std::time::Instant;

/// Run the CPU FFT validation suite.
pub fn run_fft_cpu_suite() -> Result<CpuFftReport, Box<dyn std::error::Error>> {
    let plan = FftPlan3D::new(8, 8, 8);
    let field = Array3::from_shape_fn((8, 8, 8), |(i, j, k)| {
        (i as f64 * 0.3 + j as f64 * 0.5 + k as f64 * 0.7).sin()
    });
    let spectrum = plan.forward(&field);
    let recovered = plan.inverse(&spectrum);
    let roundtrip_max_abs_error = field
        .iter()
        .zip(recovered.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);

    let signal =
        Array1::from_shape_fn(32, |i| (i as f64 * 0.23).sin() + (i as f64 * 0.11).cos());
    let signal_spectrum = FftPlan1D::new(signal.len()).forward(&signal);
    let spatial_energy: f64 = signal.iter().map(|value| value * value).sum();
    let spectral_energy: f64 =
        signal_spectrum.iter().map(|value| value.norm_sqr()).sum::<f64>() / signal.len() as f64;
    let parseval_relative_error = ((spatial_energy - spectral_energy) / spatial_energy.max(1e-12)).abs();

    Ok(CpuFftReport {
        roundtrip_max_abs_error,
        parseval_relative_error,
        passed: roundtrip_max_abs_error < 1e-12 && parseval_relative_error < 1e-12,
    })
}

/// Run the GPU FFT validation suite.
pub fn run_fft_gpu_suite() -> Result<GpuFftReport, Box<dyn std::error::Error>> {
    let cpu_plan = FftPlan3D::new(8, 8, 8);
    let field = Array3::from_shape_fn((8, 8, 8), |(i, j, k)| {
        (i as f64 * 0.3 + j as f64 * 0.5 + k as f64 * 0.7).sin()
    });
    let cpu_spectrum = cpu_plan.forward(&field);
    let cpu_inverse = cpu_plan.inverse(&cpu_spectrum);
    let surface_reported_available = apollofft_wgpu::gpu_fft_available();

    let report = match apollofft_wgpu::WgpuBackend::try_default() {
        Ok(backend) => {
            let gpu_plan = backend.plan_3d(Shape3D::new(8, 8, 8)?)?;
            let gpu_spectrum = gpu_plan.forward(&field);
            let gpu_complex = Array3::from_shape_fn((8, 8, 8), |(i, j, k)| {
                let flat = ((i * 8 + j) * 8 + k) * 2;
                Complex64::new(gpu_spectrum[flat] as f64, gpu_spectrum[flat + 1] as f64)
            });
            let forward_max_abs_error = cpu_spectrum
                .iter()
                .zip(gpu_complex.iter())
                .map(|(lhs, rhs)| (lhs - rhs).norm())
                .fold(0.0_f64, f64::max);

            let mut gpu_inverse = Array3::<f64>::zeros((8, 8, 8));
            gpu_plan.inverse(&gpu_spectrum, &mut gpu_inverse);
            let inverse_max_abs_error = cpu_inverse
                .iter()
                .zip(gpu_inverse.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0_f64, f64::max);

            GpuFftReport {
                surface_reported_available,
                attempted: true,
                passed: forward_max_abs_error < 2e-5 && inverse_max_abs_error < 1e-5,
                forward_max_abs_error: Some(forward_max_abs_error),
                inverse_max_abs_error: Some(inverse_max_abs_error),
                note: None,
            }
        }
        Err(error) => GpuFftReport {
            surface_reported_available,
            attempted: false,
            passed: false,
            forward_max_abs_error: None,
            inverse_max_abs_error: None,
            note: Some(error.to_string()),
        },
    };

    Ok(report)
}

/// Run the NUFFT validation suite.
pub fn run_nufft_suite() -> Result<NufftReport, Box<dyn std::error::Error>> {
    let domain = UniformDomain1D::new(32, 0.05)?;
    let positions: Vec<f64> = (0..20)
        .map(|i| ((i as f64 * 0.137 + 0.03) % domain.length()).abs())
        .collect();
    let values: Vec<Complex64> = (0..20)
        .map(|i| Complex64::new((i as f64 * 0.3).cos(), (i as f64 * 0.17).sin()))
        .collect();

    let exact_type1_1d = nufft_type1_1d(&positions, &values, domain);
    let fast_type1_1d =
        nufft_type1_1d_fast(&positions, &values, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
    let type1_1d_scale = exact_type1_1d
        .iter()
        .map(|value| value.norm())
        .fold(1.0_f64, f64::max);
    let type1_1d_max_relative_error = exact_type1_1d
        .iter()
        .zip(fast_type1_1d.iter())
        .map(|(lhs, rhs)| (lhs - rhs).norm())
        .fold(0.0_f64, f64::max)
        / type1_1d_scale;

    let exact_type2_1d = nufft_type2_1d(&exact_type1_1d, &positions, domain);
    let fast_type2_1d =
        nufft_type2_1d_fast(&exact_type1_1d, &positions, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
    let type2_1d_scale = exact_type2_1d
        .iter()
        .map(|value| value.norm())
        .fold(1.0_f64, f64::max);
    let type2_1d_max_relative_error = exact_type2_1d
        .iter()
        .zip(fast_type2_1d.iter())
        .map(|(lhs, rhs)| (lhs - rhs).norm())
        .fold(0.0_f64, f64::max)
        / type2_1d_scale;

    let grid = UniformGrid3D::new(8, 8, 8, 0.1, 0.1, 0.1)?;
    let positions_3d: Vec<(f64, f64, f64)> = (0..10)
        .map(|i| {
            let f = i as f64;
            (
                (f * 0.17).rem_euclid(grid.nx as f64 * grid.dx),
                (f * 0.11).rem_euclid(grid.ny as f64 * grid.dy),
                (f * 0.07).rem_euclid(grid.nz as f64 * grid.dz),
            )
        })
        .collect();
    let values_3d: Vec<Complex64> = (0..10)
        .map(|i| Complex64::new((i as f64 * 0.4).cos(), (i as f64 * 0.3).sin()))
        .collect();
    let exact_type1_3d = nufft_type1_3d(&positions_3d, &values_3d, grid);
    let fast_type1_3d =
        nufft_type1_3d_fast(&positions_3d, &values_3d, grid, DEFAULT_NUFFT_KERNEL_WIDTH);
    let type1_3d_scale = exact_type1_3d
        .iter()
        .map(|value| value.norm())
        .fold(1.0_f64, f64::max);
    let type1_3d_max_relative_error = exact_type1_3d
        .iter()
        .zip(fast_type1_3d.iter())
        .map(|(lhs, rhs)| (lhs - rhs).norm())
        .fold(0.0_f64, f64::max)
        / type1_3d_scale;

    Ok(NufftReport {
        type1_1d_max_relative_error,
        type2_1d_max_relative_error,
        type1_3d_max_relative_error,
        passed: type1_1d_max_relative_error < 1e-6
            && type2_1d_max_relative_error < 1e-6
            && type1_3d_max_relative_error < 1e-6,
    })
}

/// Run optional external comparison probes.
pub fn run_external_comparison_suite(
) -> Result<ExternalComparisonReport, Box<dyn std::error::Error>> {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf();
    let rustfft_checkout_present = workspace_root.join("external").join("rustfft").exists();
    let pyfftw_checkout_present = workspace_root.join("external").join("pyfftw").exists();

    let comparison_signal: Vec<Complex<f64>> = (0..16)
        .map(|i| Complex::new((i as f64 * 0.31).sin(), 0.0))
        .collect();
    let apollo_input = Array1::from_iter(comparison_signal.iter().map(|value| value.re));
    let apollo_output = FftPlan1D::new(apollo_input.len()).forward(&apollo_input);
    let mut rustfft_output = comparison_signal.clone();
    FftPlanner::<f64>::new()
        .plan_fft_forward(rustfft_output.len())
        .process(&mut rustfft_output);
    let rustfft_max_abs_error = apollo_output
        .iter()
        .zip(rustfft_output.iter())
        .map(|(lhs, rhs)| (lhs.re - rhs.re).hypot(lhs.im - rhs.im))
        .fold(0.0_f64, f64::max);

    let note = if rustfft_checkout_present || pyfftw_checkout_present {
        None
    } else {
        Some("external checkouts not present; only in-process rustfft comparison was executed".to_string())
    };

    Ok(ExternalComparisonReport {
        passed: rustfft_max_abs_error < 1e-12,
        rustfft_max_abs_error,
        rustfft_checkout_present,
        pyfftw_checkout_present,
        note,
    })
}

/// Run benchmark timing probes for representative Apollo workloads.
pub fn run_benchmark_suite() -> Result<BenchmarkReport, Box<dyn std::error::Error>> {
    let plan = FftPlan3D::new(8, 8, 8);
    let field = Array3::from_shape_fn((8, 8, 8), |(i, j, k)| {
        (i as f64 * 0.3 + j as f64 * 0.5 + k as f64 * 0.7).sin()
    });

    let cpu_forward_start = Instant::now();
    let spectrum = plan.forward(&field);
    let cpu_fft_forward_ms = cpu_forward_start.elapsed().as_secs_f64() * 1_000.0;

    let cpu_inverse_start = Instant::now();
    let _ = plan.inverse(&spectrum);
    let cpu_fft_inverse_ms = cpu_inverse_start.elapsed().as_secs_f64() * 1_000.0;

    let domain = UniformDomain1D::new(32, 0.05)?;
    let positions: Vec<f64> = (0..20)
        .map(|i| ((i as f64 * 0.137 + 0.03) % domain.length()).abs())
        .collect();
    let values: Vec<Complex64> = (0..20)
        .map(|i| Complex64::new((i as f64 * 0.3).cos(), (i as f64 * 0.17).sin()))
        .collect();

    let exact_start = Instant::now();
    let _ = nufft_type1_1d(&positions, &values, domain);
    let nufft_exact_type1_1d_ms = exact_start.elapsed().as_secs_f64() * 1_000.0;

    let fast_1d_start = Instant::now();
    let _ = nufft_type1_1d_fast(&positions, &values, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
    let nufft_fast_type1_1d_ms = fast_1d_start.elapsed().as_secs_f64() * 1_000.0;

    let grid = UniformGrid3D::new(8, 8, 8, 0.1, 0.1, 0.1)?;
    let positions_3d: Vec<(f64, f64, f64)> = (0..10)
        .map(|i| {
            let f = i as f64;
            (
                (f * 0.17).rem_euclid(grid.nx as f64 * grid.dx),
                (f * 0.11).rem_euclid(grid.ny as f64 * grid.dy),
                (f * 0.07).rem_euclid(grid.nz as f64 * grid.dz),
            )
        })
        .collect();
    let values_3d: Vec<Complex64> = (0..10)
        .map(|i| Complex64::new((i as f64 * 0.4).cos(), (i as f64 * 0.3).sin()))
        .collect();

    let fast_3d_start = Instant::now();
    let _ = nufft_type1_3d_fast(&positions_3d, &values_3d, grid, DEFAULT_NUFFT_KERNEL_WIDTH);
    let nufft_fast_type1_3d_ms = fast_3d_start.elapsed().as_secs_f64() * 1_000.0;

    let (gpu_fft_forward_ms, gpu_fft_inverse_ms) = match apollofft_wgpu::WgpuBackend::try_default()
    {
        Ok(backend) => {
            let gpu_plan = backend.plan_3d(Shape3D::new(8, 8, 8)?)?;
            let gpu_forward_start = Instant::now();
            let gpu_spectrum = gpu_plan.forward(&field);
            let gpu_fft_forward_ms = gpu_forward_start.elapsed().as_secs_f64() * 1_000.0;

            let gpu_inverse_start = Instant::now();
            let mut gpu_recovered = Array3::<f64>::zeros((8, 8, 8));
            gpu_plan.inverse(&gpu_spectrum, &mut gpu_recovered);
            let gpu_fft_inverse_ms = gpu_inverse_start.elapsed().as_secs_f64() * 1_000.0;
            (Some(gpu_fft_forward_ms), Some(gpu_fft_inverse_ms))
        }
        Err(_) => (None, None),
    };

    Ok(BenchmarkReport {
        cpu_fft_forward_ms,
        cpu_fft_inverse_ms,
        nufft_exact_type1_1d_ms,
        nufft_fast_type1_1d_ms,
        nufft_fast_type1_3d_ms,
        gpu_fft_forward_ms,
        gpu_fft_inverse_ms,
    })
}

/// Run the full Apollo validation matrix.
pub fn run_full_suite() -> Result<ValidationReport, Box<dyn std::error::Error>> {
    Ok(ValidationReport {
        fft_cpu: run_fft_cpu_suite()?,
        fft_gpu: run_fft_gpu_suite()?,
        nufft: run_nufft_suite()?,
        external: run_external_comparison_suite()?,
        benchmarks: run_benchmark_suite()?,
        environment: EnvironmentReport {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            debug_build: cfg!(debug_assertions),
        },
    })
}

/// Backward-compatible alias for the old validation entrypoint.
pub fn run_validation_suite() -> Result<ValidationReport, Box<dyn std::error::Error>> {
    run_full_suite()
}

/// Backward-compatible alias for the old smoke suite entrypoint.
pub fn run_smoke_suite() -> Result<ValidationReport, Box<dyn std::error::Error>> {
    run_full_suite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_suite_passes_reference_thresholds() {
        let report = run_fft_cpu_suite().expect("cpu suite must run");
        assert!(report.passed);
        assert!(report.roundtrip_max_abs_error < 1e-12);
    }

    #[test]
    fn nufft_suite_passes_reference_thresholds() {
        let report = run_nufft_suite().expect("nufft suite must run");
        assert!(report.passed);
    }

    #[test]
    fn external_suite_reports_note_when_checkouts_are_missing() {
        let report = run_external_comparison_suite().expect("external suite must run");
        if !report.rustfft_checkout_present && !report.pyfftw_checkout_present {
            assert!(report.note.is_some());
        }
    }
}
