//! Validation suite orchestration.

use crate::domain::report::{
    BenchmarkReport, CpuFftReport, EnvironmentReport, ExternalBackendReport,
    ExternalComparisonReport, GpuFftReport, NufftReport, ValidationReport,
};
use crate::infrastructure::{numpy, rustfft_reference};
use apollofft::{
    nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d, nufft_type1_3d_fast, nufft_type2_1d,
    nufft_type2_1d_fast, Complex64, FftBackend, FftPlan1D, FftPlan3D, Shape3D, UniformDomain1D,
    UniformGrid3D, DEFAULT_NUFFT_KERNEL_WIDTH,
};
use ndarray::{arr1, Array1, Array3};
use std::path::PathBuf;
use std::time::Instant;

const EXTERNAL_STABILITY_REPEATS: usize = 4;
const BENCHMARK_ITERATIONS: usize = 24;

fn representative_signal(len: usize) -> Array1<f64> {
    Array1::from_shape_fn(len, |i| (i as f64 * 0.23).sin() + (i as f64 * 0.11).cos())
}

fn representative_field(nx: usize, ny: usize, nz: usize) -> Array3<f64> {
    Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        (i as f64 * 0.3 + j as f64 * 0.5 + k as f64 * 0.7).sin() + (i as f64 * 0.13).cos() * 0.25
            - (k as f64 * 0.17).sin() * 0.5
    })
}

fn complex_max_abs_error<I, J>(lhs: I, rhs: J) -> f64
where
    I: Iterator<Item = Complex64>,
    J: Iterator<Item = Complex64>,
{
    lhs.zip(rhs)
        .map(|(left, right)| (left - right).norm())
        .fold(0.0_f64, f64::max)
}

fn benchmark_ms<F>(iterations: usize, mut f: F) -> f64
where
    F: FnMut(),
{
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed().as_secs_f64() * 1_000.0 / iterations as f64
}

fn python_environment_probe() -> EnvironmentReport {
    let probe = numpy::probe_python_environment().ok();
    EnvironmentReport {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        debug_build: cfg!(debug_assertions),
        python_version: probe.as_ref().map(|value| value.python_version.clone()),
        numpy_version: probe.as_ref().and_then(|value| value.numpy_version.clone()),
        pyfftw_version: probe.and_then(|value| value.pyfftw_version),
    }
}

/// Run the CPU FFT validation suite.
pub fn run_fft_cpu_suite() -> Result<CpuFftReport, Box<dyn std::error::Error>> {
    let plan = FftPlan3D::new(8, 8, 8);
    let field = representative_field(8, 8, 8);
    let spectrum = plan.forward(&field);
    let recovered = plan.inverse(&spectrum);
    let roundtrip_max_abs_error = field
        .iter()
        .zip(recovered.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);

    let signal = representative_signal(32);
    let signal_spectrum = FftPlan1D::new(signal.len()).forward(&signal);
    let spatial_energy: f64 = signal.iter().map(|value| value * value).sum();
    let spectral_energy: f64 = signal_spectrum
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        / signal.len() as f64;
    let parseval_relative_error =
        ((spatial_energy - spectral_energy) / spatial_energy.max(1e-12)).abs();

    let reference_spectrum = FftPlan1D::new(signal.len()).forward(&signal);
    let stability_max_abs_delta = (0..EXTERNAL_STABILITY_REPEATS)
        .map(|_| FftPlan1D::new(signal.len()).forward(&signal))
        .map(|candidate| {
            complex_max_abs_error(
                reference_spectrum.iter().copied(),
                candidate.iter().copied(),
            )
        })
        .fold(0.0_f64, f64::max);

    let non_finite_signal = arr1(&[0.0, f64::NAN, 1.0, f64::INFINITY]);
    let non_finite_input_propagates = FftPlan1D::new(non_finite_signal.len())
        .forward(&non_finite_signal)
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite());

    Ok(CpuFftReport {
        roundtrip_max_abs_error,
        parseval_relative_error,
        stability_max_abs_delta,
        non_finite_input_propagates,
        passed: roundtrip_max_abs_error < 1e-12
            && parseval_relative_error < 1e-12
            && stability_max_abs_delta < 1e-12
            && non_finite_input_propagates,
    })
}

/// Run the GPU FFT validation suite.
pub fn run_fft_gpu_suite() -> Result<GpuFftReport, Box<dyn std::error::Error>> {
    let cpu_plan = FftPlan3D::new(8, 8, 8);
    let field = representative_field(8, 8, 8);
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
                .map(|(lhs, rhs)| (*lhs - *rhs).norm())
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
                passed: forward_max_abs_error < 5e-5 && inverse_max_abs_error < 1e-5,
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
    let fast_type2_1d = nufft_type2_1d_fast(
        &exact_type1_1d,
        &positions,
        domain,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
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

    let irrational_positions = vec![
        std::f64::consts::FRAC_1_PI * domain.length() * 0.5,
        std::f64::consts::SQRT_2.fract() * domain.length() * 0.75,
        std::f64::consts::E.fract() * domain.length() * 0.9,
        std::f64::consts::PI.fract() * domain.length() * 0.3,
    ];
    let irrational_values = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(-0.3, 0.4),
        Complex64::new(0.2, -0.7),
        Complex64::new(-0.6, -0.1),
    ];
    let exact_irrational = nufft_type1_1d(&irrational_positions, &irrational_values, domain);
    let fast_irrational = nufft_type1_1d_fast(
        &irrational_positions,
        &irrational_values,
        domain,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let irrational_scale = exact_irrational
        .iter()
        .map(|value| value.norm())
        .fold(1.0_f64, f64::max);
    let irrational_positions_max_relative_error = exact_irrational
        .iter()
        .zip(fast_irrational.iter())
        .map(|(lhs, rhs)| (lhs - rhs).norm())
        .fold(0.0_f64, f64::max)
        / irrational_scale;

    let clustered_positions = vec![
        domain.length() - 1.0e-8,
        domain.length() - 5.0e-9,
        2.5e-9,
        7.5e-9,
        1.25e-8,
    ];
    let clustered_values = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(-0.4, 0.25),
        Complex64::new(0.3, -0.6),
        Complex64::new(-0.15, 0.1),
        Complex64::new(0.5, 0.2),
    ];
    let exact_clustered = nufft_type1_1d(&clustered_positions, &clustered_values, domain);
    let fast_clustered = nufft_type1_1d_fast(
        &clustered_positions,
        &clustered_values,
        domain,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let clustered_scale = exact_clustered
        .iter()
        .map(|value| value.norm())
        .fold(1.0_f64, f64::max);
    let clustered_positions_max_relative_error = exact_clustered
        .iter()
        .zip(fast_clustered.iter())
        .map(|(lhs, rhs)| (lhs - rhs).norm())
        .fold(0.0_f64, f64::max)
        / clustered_scale;

    Ok(NufftReport {
        type1_1d_max_relative_error,
        type2_1d_max_relative_error,
        type1_3d_max_relative_error,
        irrational_positions_max_relative_error,
        clustered_positions_max_relative_error,
        passed: type1_1d_max_relative_error < 1e-6
            && type2_1d_max_relative_error < 1e-6
            && type1_3d_max_relative_error < 1e-6
            && irrational_positions_max_relative_error < 1e-6
            && clustered_positions_max_relative_error < 1e-6,
    })
}

fn rustfft_external_report(
    signal: &Array1<f64>,
    prime_signal: &Array1<f64>,
    field: &Array3<f64>,
) -> ExternalBackendReport {
    let apollo_fft1 = FftPlan1D::new(signal.len()).forward(signal);
    let rustfft_fft1 = rustfft_reference::fft1_real(signal);
    let fft1_max_abs_error =
        complex_max_abs_error(apollo_fft1.iter().copied(), rustfft_fft1.into_iter());

    let apollo_fft1_prime = FftPlan1D::new(prime_signal.len()).forward(prime_signal);
    let rustfft_fft1_prime = rustfft_reference::fft1_real(prime_signal);
    let fft1_prime_max_abs_error = complex_max_abs_error(
        apollo_fft1_prime.iter().copied(),
        rustfft_fft1_prime.into_iter(),
    );

    let apollo_fft3 = FftPlan3D::new(field.dim().0, field.dim().1, field.dim().2).forward(field);
    let rustfft_fft3 = rustfft_reference::fft3_real(field);
    let fft3_max_abs_error =
        complex_max_abs_error(apollo_fft3.iter().copied(), rustfft_fft3.iter().copied());

    let reference = rustfft_reference::fft1_real(signal);
    let stability_max_abs_delta = (0..EXTERNAL_STABILITY_REPEATS)
        .map(|_| rustfft_reference::fft1_real(signal))
        .map(|candidate| complex_max_abs_error(reference.iter().copied(), candidate.into_iter()))
        .fold(0.0_f64, f64::max);

    ExternalBackendReport {
        backend: "rustfft".to_string(),
        available: true,
        attempted: true,
        fft1_max_abs_error: Some(fft1_max_abs_error),
        fft1_prime_max_abs_error: Some(fft1_prime_max_abs_error),
        fft3_max_abs_error: Some(fft3_max_abs_error),
        stability_max_abs_delta: Some(stability_max_abs_delta),
        version: None,
        note: None,
    }
}

fn python_external_report(
    backend: &str,
    available: bool,
    pairs_1d: Option<Vec<[f64; 2]>>,
    pairs_prime: Option<Vec<[f64; 2]>>,
    pairs_3d: Option<Vec<[f64; 2]>>,
    stability: Option<f64>,
    version: Option<String>,
    apollo_fft1: &[Complex64],
    apollo_fft1_prime: &[Complex64],
    apollo_fft3: &[Complex64],
    unavailable_note: &str,
) -> ExternalBackendReport {
    let decode = |pairs: Vec<[f64; 2]>| {
        pairs
            .into_iter()
            .map(|value| Complex64::new(value[0], value[1]))
            .collect::<Vec<_>>()
    };

    let fft1_max_abs_error = pairs_1d
        .map(|pairs| complex_max_abs_error(apollo_fft1.iter().copied(), decode(pairs).into_iter()));
    let fft1_prime_max_abs_error = pairs_prime.map(|pairs| {
        complex_max_abs_error(apollo_fft1_prime.iter().copied(), decode(pairs).into_iter())
    });
    let fft3_max_abs_error = pairs_3d
        .map(|pairs| complex_max_abs_error(apollo_fft3.iter().copied(), decode(pairs).into_iter()));

    ExternalBackendReport {
        backend: backend.to_string(),
        available,
        attempted: available,
        fft1_max_abs_error,
        fft1_prime_max_abs_error,
        fft3_max_abs_error,
        stability_max_abs_delta: stability,
        version,
        note: if available {
            None
        } else {
            Some(unavailable_note.to_string())
        },
    }
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

    let signal = representative_signal(16);
    let prime_signal = representative_signal(17);
    let field = representative_field(4, 5, 6);
    let rustfft = rustfft_external_report(&signal, &prime_signal, &field);

    let apollo_fft1 = FftPlan1D::new(signal.len()).forward(&signal).to_vec();
    let apollo_fft1_prime = FftPlan1D::new(prime_signal.len())
        .forward(&prime_signal)
        .to_vec();
    let apollo_fft3 = FftPlan3D::new(4, 5, 6)
        .forward(&field)
        .iter()
        .copied()
        .collect::<Vec<_>>();

    let numpy_1d = numpy::compare_fft(
        &[signal.len()],
        signal.as_slice().ok_or("1d slice failed")?,
        EXTERNAL_STABILITY_REPEATS,
    )
    .ok();
    let numpy_prime = numpy::compare_fft(
        &[prime_signal.len()],
        prime_signal.as_slice().ok_or("prime slice failed")?,
        EXTERNAL_STABILITY_REPEATS,
    )
    .ok();
    let numpy_3d = numpy::compare_fft(
        &[4, 5, 6],
        field.as_slice().ok_or("3d slice failed")?,
        EXTERNAL_STABILITY_REPEATS,
    )
    .ok();

    let numpy_available = numpy_1d
        .as_ref()
        .map(|report| report.numpy_available)
        .unwrap_or(false);
    let pyfftw_available = numpy_1d
        .as_ref()
        .map(|report| report.pyfftw_available)
        .unwrap_or(false);

    let numpy = python_external_report(
        "numpy",
        numpy_available,
        numpy_1d
            .as_ref()
            .and_then(|value| value.numpy_pairs.clone()),
        numpy_prime
            .as_ref()
            .and_then(|value| value.numpy_pairs.clone()),
        numpy_3d
            .as_ref()
            .and_then(|value| value.numpy_pairs.clone()),
        numpy_1d
            .as_ref()
            .and_then(|value| value.numpy_stability_max_abs_delta),
        numpy_1d
            .as_ref()
            .and_then(|value| value.numpy_version.clone()),
        &apollo_fft1,
        &apollo_fft1_prime,
        &apollo_fft3,
        "python or numpy was unavailable",
    );
    let pyfftw = python_external_report(
        "pyfftw",
        pyfftw_available,
        numpy_1d
            .as_ref()
            .and_then(|value| value.pyfftw_pairs.clone()),
        numpy_prime
            .as_ref()
            .and_then(|value| value.pyfftw_pairs.clone()),
        numpy_3d
            .as_ref()
            .and_then(|value| value.pyfftw_pairs.clone()),
        numpy_1d
            .as_ref()
            .and_then(|value| value.pyfftw_stability_max_abs_delta),
        numpy_1d
            .as_ref()
            .and_then(|value| value.pyfftw_version.clone()),
        &apollo_fft1,
        &apollo_fft1_prime,
        &apollo_fft3,
        "pyfftw was unavailable",
    );

    let rustfft_passed = rustfft.fft1_max_abs_error.unwrap_or(f64::INFINITY) < 1e-12
        && rustfft.fft1_prime_max_abs_error.unwrap_or(f64::INFINITY) < 1e-12
        && rustfft.fft3_max_abs_error.unwrap_or(f64::INFINITY) < 1e-12
        && rustfft.stability_max_abs_delta.unwrap_or(f64::INFINITY) < 1e-12;
    let numpy_passed = !numpy.available
        || (numpy.fft1_max_abs_error.unwrap_or(f64::INFINITY) < 1e-12
            && numpy.fft1_prime_max_abs_error.unwrap_or(f64::INFINITY) < 1e-12
            && numpy.fft3_max_abs_error.unwrap_or(f64::INFINITY) < 1e-12
            && numpy.stability_max_abs_delta.unwrap_or(f64::INFINITY) < 1e-12);
    let pyfftw_passed = !pyfftw.available
        || (pyfftw.fft1_max_abs_error.unwrap_or(f64::INFINITY) < 1e-12
            && pyfftw.fft1_prime_max_abs_error.unwrap_or(f64::INFINITY) < 1e-12
            && pyfftw.fft3_max_abs_error.unwrap_or(f64::INFINITY) < 1e-12
            && pyfftw.stability_max_abs_delta.unwrap_or(f64::INFINITY) < 1e-12);
    let robustness_passed = rustfft_passed && numpy_passed && pyfftw_passed;

    let note = if numpy.available
        || pyfftw.available
        || rustfft_checkout_present
        || pyfftw_checkout_present
    {
        None
    } else {
        Some("only the built-in rustfft reference path was available; python comparators were skipped".to_string())
    };

    Ok(ExternalComparisonReport {
        passed: robustness_passed,
        rustfft_checkout_present,
        pyfftw_checkout_present,
        rustfft,
        numpy,
        pyfftw,
        robustness_passed,
        note,
    })
}

/// Run benchmark timing probes for representative Apollo workloads.
pub fn run_benchmark_suite() -> Result<BenchmarkReport, Box<dyn std::error::Error>> {
    let signal = representative_signal(256);
    let field = representative_field(16, 16, 16);

    let fft1_plan = FftPlan1D::new(signal.len());
    let apollo_fft1_ms = benchmark_ms(BENCHMARK_ITERATIONS, || {
        let _ = fft1_plan.forward(&signal);
    });

    let fft3_plan = FftPlan3D::new(16, 16, 16);
    let apollo_fft3_forward_ms = benchmark_ms(BENCHMARK_ITERATIONS, || {
        let _ = fft3_plan.forward(&field);
    });
    let baseline_spectrum = fft3_plan.forward(&field);
    let apollo_fft3_inverse_ms = benchmark_ms(BENCHMARK_ITERATIONS, || {
        let _ = fft3_plan.inverse(&baseline_spectrum);
    });

    let rustfft_fft1_ms = benchmark_ms(BENCHMARK_ITERATIONS, || {
        let _ = rustfft_reference::fft1_real(&signal);
    });
    let rustfft_fft3_ms = benchmark_ms(BENCHMARK_ITERATIONS, || {
        let _ = rustfft_reference::fft3_real(&field);
    });

    let numpy_1d = numpy::benchmark_fft(
        &[signal.len()],
        signal.as_slice().ok_or("1d slice failed")?,
        BENCHMARK_ITERATIONS,
    )
    .ok();
    let numpy_3d = numpy::benchmark_fft(
        &[16, 16, 16],
        field.as_slice().ok_or("3d slice failed")?,
        BENCHMARK_ITERATIONS,
    )
    .ok();

    let domain = UniformDomain1D::new(32, 0.05)?;
    let positions: Vec<f64> = (0..20)
        .map(|i| ((i as f64 * 0.137 + 0.03) % domain.length()).abs())
        .collect();
    let values: Vec<Complex64> = (0..20)
        .map(|i| Complex64::new((i as f64 * 0.3).cos(), (i as f64 * 0.17).sin()))
        .collect();

    let nufft_exact_type1_1d_ms = benchmark_ms(BENCHMARK_ITERATIONS, || {
        let _ = nufft_type1_1d(&positions, &values, domain);
    });
    let nufft_fast_type1_1d_ms = benchmark_ms(BENCHMARK_ITERATIONS, || {
        let _ = nufft_type1_1d_fast(&positions, &values, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
    });

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
    let nufft_fast_type1_3d_ms = benchmark_ms(BENCHMARK_ITERATIONS, || {
        let _ = nufft_type1_3d_fast(&positions_3d, &values_3d, grid, DEFAULT_NUFFT_KERNEL_WIDTH);
    });

    let (gpu_fft_forward_ms, gpu_fft_inverse_ms) = match apollofft_wgpu::WgpuBackend::try_default()
    {
        Ok(backend) => {
            let gpu_plan = backend.plan_3d(Shape3D::new(16, 16, 16)?)?;
            let gpu_fft_forward_ms = benchmark_ms(BENCHMARK_ITERATIONS, || {
                let _ = gpu_plan.forward(&field);
            });
            let gpu_spectrum = gpu_plan.forward(&field);
            let gpu_fft_inverse_ms = benchmark_ms(BENCHMARK_ITERATIONS, || {
                let mut gpu_recovered = Array3::<f64>::zeros((16, 16, 16));
                gpu_plan.inverse(&gpu_spectrum, &mut gpu_recovered);
            });
            (Some(gpu_fft_forward_ms), Some(gpu_fft_inverse_ms))
        }
        Err(_) => (None, None),
    };

    Ok(BenchmarkReport {
        apollo_fft1_ms,
        apollo_fft3_forward_ms,
        apollo_fft3_inverse_ms,
        rustfft_fft1_ms,
        rustfft_fft3_ms,
        numpy_fft1_ms: numpy_1d.as_ref().and_then(|value| value.numpy_ms),
        numpy_fft3_ms: numpy_3d.as_ref().and_then(|value| value.numpy_ms),
        pyfftw_fft1_ms: numpy_1d.as_ref().and_then(|value| value.pyfftw_ms),
        pyfftw_fft3_ms: numpy_3d.as_ref().and_then(|value| value.pyfftw_ms),
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
        environment: python_environment_probe(),
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
        assert!(report.stability_max_abs_delta < 1e-12);
    }

    #[test]
    fn nufft_suite_passes_reference_thresholds() {
        let report = run_nufft_suite().expect("nufft suite must run");
        assert!(report.passed);
        assert!(report.irrational_positions_max_relative_error < 1e-6);
        assert!(report.clustered_positions_max_relative_error < 1e-6);
    }

    #[test]
    fn external_suite_reports_rustfft_parity() {
        let report = run_external_comparison_suite().expect("external suite must run");
        assert!(report.rustfft.attempted);
        assert!(report.rustfft.fft1_max_abs_error.expect("rustfft error") < 1e-12);
    }

    #[test]
    fn benchmark_suite_exposes_rustfft_timings() {
        let report = run_benchmark_suite().expect("benchmark suite must run");
        assert!(report.apollo_fft1_ms >= 0.0);
        assert!(report.rustfft_fft1_ms >= 0.0);
    }
}
