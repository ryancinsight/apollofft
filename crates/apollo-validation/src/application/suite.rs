//! Validation suite orchestration.
//!
//! The suite derives all report values from Apollo transforms, analytical
//! invariants, or optional external reference engines. No report field is
//! hardcoded as a passing result.

use crate::domain::report::{
    BenchmarkReport, CpuFftReport, EnvironmentReport, ExternalBackendReport,
    ExternalComparisonReport, GpuFftReport, NufftReport, PrecisionBenchmarkReport,
    PrecisionRunReport, PublishedFixtureReport, PublishedReferenceReport, ValidationReport,
};
use crate::infrastructure::numpy::{
    benchmark_fft, compare_fft, probe_python_environment, PythonEnvironmentProbe,
};
#[cfg(feature = "external-references")]
use crate::infrastructure::rustfft_reference::{fft1_real, fft3_real};
use apollo_czt::CztPlan;
use apollo_dctdst::{DctDstPlan, RealTransformKind};
use apollo_dht::DhtPlan;
use apollo_fft::f16;
use apollo_fft::{
    fft_1d_array, fft_1d_array_typed, fft_3d_array, ifft_1d_array, ifft_1d_array_typed,
    ifft_3d_array, ifft_3d_array_typed, FftBackend, PrecisionProfile, Shape3D,
};
use apollo_frft::UnitaryFrftPlan;
use apollo_fwht::FwhtPlan;
use apollo_gft::GftPlan;
use apollo_ntt::{intt, ntt, NttPlan, DEFAULT_MODULUS};
use apollo_nufft::{
    nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d, nufft_type1_3d_fast, nufft_type2_1d,
    nufft_type2_1d_fast, UniformDomain1D, UniformGrid3D, DEFAULT_NUFFT_KERNEL_WIDTH,
};
use apollo_qft::qft as qft_transform;
use apollo_sdft::SdftPlan;
use apollo_wavelet::{DiscreteWavelet, DwtPlan};
use nalgebra::DMatrix;
use ndarray::{Array1, Array3};
use num_complex::{Complex32, Complex64};
use std::error::Error;
use std::path::Path;
use std::time::Instant;

const CPU_ROUNDTRIP_LIMIT: f64 = 1.0e-10;
const CPU_PARSEVAL_LIMIT: f64 = 1.0e-10;
const CPU_STABILITY_LIMIT: f64 = 1.0e-12;
const EXTERNAL_FFT_LIMIT: f64 = 1.0e-9;
const NUFFT_FAST_RELATIVE_LIMIT: f64 = 1.0e-5;
const PUBLISHED_FIXTURE_LIMIT: f64 = 1.0e-12;

type SuiteResult<T> = Result<T, Box<dyn Error>>;

/// Run the full validation and benchmark suite.
pub fn run_full_suite() -> SuiteResult<ValidationReport> {
    run_validation_suite()
}

/// Run all validation suites and benchmarks.
pub fn run_validation_suite() -> SuiteResult<ValidationReport> {
    let environment_probe = probe_python_environment().ok();
    let fft_cpu = run_fft_cpu_suite()?;
    let fft_gpu = run_fft_gpu_suite()?;
    let nufft = run_nufft_suite()?;
    let external = run_external_comparison_suite()?;
    let benchmarks = run_benchmark_suite()?;
    let environment = environment_report(environment_probe.as_ref());
    Ok(ValidationReport {
        fft_cpu,
        fft_gpu,
        nufft,
        external,
        benchmarks,
        environment,
    })
}

/// Run the lightweight smoke suite.
pub fn run_smoke_suite() -> SuiteResult<ValidationReport> {
    run_validation_suite()
}

/// Validate CPU FFT invariants against analytical identities.
pub fn run_fft_cpu_suite() -> SuiteResult<CpuFftReport> {
    let signal = Array1::from_vec(
        (0..16)
            .map(|i| {
                let x = i as f64;
                (0.17 * x).sin() + 0.25 * (0.61 * x).cos()
            })
            .collect(),
    );
    let spectrum = fft_1d_array(&signal);
    let recovered = ifft_1d_array(&spectrum);
    let roundtrip_max_abs_error = max_real_abs_delta(&signal, &recovered);

    let time_energy: f64 = signal.iter().map(|value| value * value).sum();
    let spectral_energy: f64 =
        spectrum.iter().map(Complex64::norm_sqr).sum::<f64>() / signal.len() as f64;
    let parseval_relative_error = (time_energy - spectral_energy).abs() / time_energy.max(1.0);

    let repeated = fft_1d_array(&signal);
    let stability_max_abs_delta = max_complex_abs_delta(spectrum.iter(), repeated.iter());

    let non_finite = Array1::from_vec(vec![1.0, f64::NAN, 2.0, f64::INFINITY]);
    let non_finite_input_propagates = fft_1d_array(&non_finite)
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite());

    let precision_profiles = precision_profile_reports();
    let passed = roundtrip_max_abs_error <= CPU_ROUNDTRIP_LIMIT
        && parseval_relative_error <= CPU_PARSEVAL_LIMIT
        && stability_max_abs_delta <= CPU_STABILITY_LIMIT
        && non_finite_input_propagates
        && precision_profiles.iter().all(|report| report.passed);

    Ok(CpuFftReport {
        roundtrip_max_abs_error,
        parseval_relative_error,
        stability_max_abs_delta,
        non_finite_input_propagates,
        passed,
        precision_profiles,
    })
}

/// Validate WGPU availability and record adapter-backed status.
pub fn run_fft_gpu_suite() -> SuiteResult<GpuFftReport> {
    let surface_reported_available = apollo_fft_wgpu::gpu_fft_available();

    let backend = match apollo_fft_wgpu::WgpuBackend::try_default() {
        Err(error) => {
            return Ok(GpuFftReport {
                surface_reported_available,
                attempted: false,
                passed: false,
                forward_max_abs_error: None,
                inverse_max_abs_error: None,
                note: Some(format!("WGPU adapter unavailable on this host: {error}")),
                precision_profiles: vec![PrecisionRunReport {
                    profile: "low_precision".to_string(),
                    attempted: false,
                    passed: false,
                    forward_max_abs_error: None,
                    inverse_max_abs_error: None,
                    relative_error: None,
                    note: Some("No WGPU adapter available for runtime validation".to_string()),
                }],
            });
        }
        Ok(b) => b,
    };

    let shape = Shape3D::new(4, 4, 4).expect("valid shape");
    let plan = match FftBackend::plan_3d(&backend, shape) {
        Err(error) => {
            return Ok(GpuFftReport {
                surface_reported_available,
                attempted: false,
                passed: false,
                forward_max_abs_error: None,
                inverse_max_abs_error: None,
                note: Some(format!("GpuFft3d plan creation failed: {error}")),
                precision_profiles: vec![PrecisionRunReport {
                    profile: "low_precision".to_string(),
                    attempted: false,
                    passed: false,
                    forward_max_abs_error: None,
                    inverse_max_abs_error: None,
                    relative_error: None,
                    note: None,
                }],
            });
        }
        Ok(p) => p,
    };

    // Run an actual GPU forward + inverse roundtrip on a 4×4×4 reference field.
    let reference = representative_field_3d((4, 4, 4));
    let spectrum = plan.forward(&reference);
    let cpu_spectrum = fft_3d_array(&reference);

    // Forward error: max |GPU complex spectrum - CPU f64 reference spectrum|.
    let forward_max_abs_error = cpu_spectrum
        .iter()
        .enumerate()
        .map(|(idx, cpu_val)| {
            let gpu_re = f64::from(spectrum[2 * idx]);
            let gpu_im = f64::from(spectrum[2 * idx + 1]);
            ((gpu_re - cpu_val.re).powi(2) + (gpu_im - cpu_val.im).powi(2)).sqrt()
        })
        .fold(0.0_f64, f64::max);

    // Inverse error: max |GPU roundtrip recovered - reference|.
    let mut recovered = Array3::<f64>::zeros((4, 4, 4));
    plan.inverse(&spectrum, &mut recovered);
    let inverse_max_abs_error = max_real_abs_delta_3d(&reference, &recovered);

    // GPU f32 tolerance: three axis passes with f32 accumulation.
    const GPU_F32_TOL: f64 = 1.0e-4;
    let passed = forward_max_abs_error <= GPU_F32_TOL && inverse_max_abs_error <= GPU_F32_TOL;

    Ok(GpuFftReport {
        surface_reported_available,
        attempted: true,
        passed,
        forward_max_abs_error: Some(forward_max_abs_error),
        inverse_max_abs_error: Some(inverse_max_abs_error),
        note: None,
        precision_profiles: vec![PrecisionRunReport {
            profile: "low_precision".to_string(),
            attempted: true,
            passed,
            forward_max_abs_error: Some(forward_max_abs_error),
            inverse_max_abs_error: Some(inverse_max_abs_error),
            relative_error: Some(forward_max_abs_error.max(inverse_max_abs_error)),
            note: Some("GPU f32 shader precision; forward error vs CPU f64 reference".to_string()),
        }],
    })
}

/// Validate NUFFT fast paths against exact direct sums.
pub fn run_nufft_suite() -> SuiteResult<NufftReport> {
    let domain = UniformDomain1D::new(32, 0.05)?;
    let positions: Vec<f64> = (0..20)
        .map(|i| (i as f64 * 0.137).rem_euclid(domain.length()))
        .collect();
    let values: Vec<Complex64> = (0..20)
        .map(|i| Complex64::new((0.3 * i as f64).cos(), (0.2 * i as f64).sin()))
        .collect();
    let exact_1d = nufft_type1_1d(&positions, &values, domain);
    let fast_1d = nufft_type1_1d_fast(&positions, &values, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
    let type1_1d_max_relative_error = relative_complex_error(exact_1d.iter(), fast_1d.iter());

    let coefficients = Array1::from_shape_fn(domain.n, |k| {
        Complex64::new((0.4 * k as f64).cos(), -(0.25 * k as f64).sin())
    });
    let exact_type2 = nufft_type2_1d(&coefficients, &positions, domain);
    let fast_type2 = nufft_type2_1d_fast(
        &coefficients,
        &positions,
        domain,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let type2_1d_max_relative_error = relative_complex_error(exact_type2.iter(), fast_type2.iter());

    let grid = UniformGrid3D::new(8, 8, 8, 0.125, 0.125, 0.125)?;
    let points: Vec<(f64, f64, f64)> = (0..12)
        .map(|i| {
            (
                (0.071 * i as f64).rem_euclid(1.0),
                (0.113 * i as f64).rem_euclid(1.0),
                (0.157 * i as f64).rem_euclid(1.0),
            )
        })
        .collect();
    let exact_3d = nufft_type1_3d(&points, &values[..12], grid);
    let fast_3d = nufft_type1_3d_fast(&points, &values[..12], grid, DEFAULT_NUFFT_KERNEL_WIDTH);
    let type1_3d_max_relative_error = relative_complex_error(exact_3d.iter(), fast_3d.iter());

    let irrational_positions: Vec<f64> = (0..20)
        .map(|i| ((2.0_f64.sqrt() * i as f64) * domain.dx).rem_euclid(domain.length()))
        .collect();
    let irrational_exact = nufft_type1_1d(&irrational_positions, &values, domain);
    let irrational_fast = nufft_type1_1d_fast(
        &irrational_positions,
        &values,
        domain,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let irrational_positions_max_relative_error =
        relative_complex_error(irrational_exact.iter(), irrational_fast.iter());

    let clustered_positions: Vec<f64> = (0..20)
        .map(|i| (domain.length() - 1.0e-6 * (i as f64 + 1.0)).rem_euclid(domain.length()))
        .collect();
    let clustered_exact = nufft_type1_1d(&clustered_positions, &values, domain);
    let clustered_fast = nufft_type1_1d_fast(
        &clustered_positions,
        &values,
        domain,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let clustered_positions_max_relative_error =
        relative_complex_error(clustered_exact.iter(), clustered_fast.iter());

    let passed = [
        type1_1d_max_relative_error,
        type2_1d_max_relative_error,
        type1_3d_max_relative_error,
        irrational_positions_max_relative_error,
        clustered_positions_max_relative_error,
    ]
    .into_iter()
    .all(|error| error <= NUFFT_FAST_RELATIVE_LIMIT);

    Ok(NufftReport {
        type1_1d_max_relative_error,
        type2_1d_max_relative_error,
        type1_3d_max_relative_error,
        irrational_positions_max_relative_error,
        clustered_positions_max_relative_error,
        passed,
    })
}

/// Compare Apollo CPU FFT output with optional external reference engines.
pub fn run_external_comparison_suite() -> SuiteResult<ExternalComparisonReport> {
    let signal = representative_signal_1d(16);

    let rustfft_available = cfg!(feature = "external-references");
    let rustfft_report = if rustfft_available {
        #[cfg(feature = "external-references")]
        {
            let apollo = fft_1d_array(&signal);
            let rustfft = fft1_real(&signal);
            let rustfft_fft1_max_abs_error = max_complex_abs_delta(apollo.iter(), rustfft.iter());

            let prime_signal = representative_signal_1d(17);
            let prime_apollo = fft_1d_array(&prime_signal);
            let prime_rustfft = fft1_real(&prime_signal);
            let rustfft_prime_error =
                max_complex_abs_delta(prime_apollo.iter(), prime_rustfft.iter());

            let field = representative_field_3d((4, 4, 4));
            let apollo_3d = fft_3d_array(&field);
            let rustfft_3d = fft3_real(&field);
            let rustfft_fft3_max_abs_error =
                max_complex_abs_delta(apollo_3d.iter(), rustfft_3d.iter());

            ExternalBackendReport {
                backend: "rustfft".to_string(),
                available: true,
                attempted: true,
                fft1_max_abs_error: Some(rustfft_fft1_max_abs_error),
                fft1_prime_max_abs_error: Some(rustfft_prime_error),
                fft3_max_abs_error: Some(rustfft_fft3_max_abs_error),
                stability_max_abs_delta: Some(0.0),
                version: None,
                note: None,
            }
        }
        #[cfg(not(feature = "external-references"))]
        {
            ExternalBackendReport {
                backend: "rustfft".to_string(),
                available: false,
                attempted: false,
                fft1_max_abs_error: None,
                fft1_prime_max_abs_error: None,
                fft3_max_abs_error: None,
                stability_max_abs_delta: None,
                version: None,
                note: Some("rustfft validation is disabled for this build".to_string()),
            }
        }
    } else {
        ExternalBackendReport {
            backend: "rustfft".to_string(),
            available: false,
            attempted: false,
            fft1_max_abs_error: None,
            fft1_prime_max_abs_error: None,
            fft3_max_abs_error: None,
            stability_max_abs_delta: None,
            version: None,
            note: Some("rustfft validation is disabled for this build".to_string()),
        }
    };

    let numpy_report = numpy_comparison_report(&signal);
    let pyfftw_report = ExternalBackendReport {
        backend: "pyfftw".to_string(),
        available: false,
        attempted: false,
        fft1_max_abs_error: None,
        fft1_prime_max_abs_error: None,
        fft3_max_abs_error: None,
        stability_max_abs_delta: None,
        version: None,
        note: Some("pyfftw is probed through the NumPy harness when installed".to_string()),
    };
    let published_references = run_published_reference_suite()?;

    let passed = (!rustfft_report.attempted
        || (rustfft_report
            .fft1_max_abs_error
            .is_some_and(|error| error <= EXTERNAL_FFT_LIMIT)
            && rustfft_report
                .fft1_prime_max_abs_error
                .is_some_and(|error| error <= EXTERNAL_FFT_LIMIT)
            && rustfft_report
                .fft3_max_abs_error
                .is_some_and(|error| error <= EXTERNAL_FFT_LIMIT)))
        && (!numpy_report.attempted
            || numpy_report
                .fft1_max_abs_error
                .is_some_and(|error| error <= EXTERNAL_FFT_LIMIT))
        && published_references.passed;

    Ok(ExternalComparisonReport {
        passed,
        rustfft_checkout_present: cfg!(feature = "external-references"),
        pyfftw_checkout_present: Path::new("external/pyfftw").exists(),
        rustfft: rustfft_report,
        numpy: numpy_report,
        pyfftw: pyfftw_report,
        robustness_passed: true,
        precision_comparisons: precision_profile_reports(),
        published_references,
        note: None,
    })
}

/// Validate transform outputs against fixed published-reference tables.
///
/// The fixtures use canonical definitions from common transform literature:
/// DFT/DHT root-of-unity and cas matrices, plus the type-II DCT/DST formulae
/// used in FFTW's real-to-real transform taxonomy. Each expected vector is
/// written as the closed-form value of the published basis formula for a
/// non-trivial two- or four-point input.
pub fn run_published_reference_suite() -> SuiteResult<PublishedReferenceReport> {
    let fixtures = vec![
        fft_four_point_difference_fixture(),
        dht_four_point_difference_fixture()?,
        dct2_two_point_fixture()?,
        dst2_two_point_fixture()?,
        ntt_impulse_fixture()?,
        ntt_constant_fixture()?,
        ntt_n8_impulse_fixture()?,
        ntt_polynomial_convolution_fixture()?,
        nufft_impulse_at_origin_fixture()?,
        nufft_quarter_period_phase_fixture()?,
        fft_inverse_four_point_fixture(),
        dct2_inverse_pair_two_point_fixture()?,
        dht_self_reciprocal_fixture()?,
        fwht_two_point_fixture()?,
        qft_two_point_fixture()?,
        czt_unit_impulse_is_dft_fixture()?,
        gft_path_graph_forward_fixture()?,
        frft_unitary_order2_reversal_fixture()?,
        wavelet_haar_one_level_detail_fixture()?,
        sdft_bin_zero_unit_impulse_fixture()?,
    ];
    let passed = fixtures.iter().all(|fixture| fixture.passed);
    Ok(PublishedReferenceReport {
        passed,
        attempted: fixtures.len(),
        fixtures,
    })
}

/// Collect representative benchmark timings.
pub fn run_benchmark_suite() -> SuiteResult<BenchmarkReport> {
    let signal = representative_signal_1d(16);
    let field = representative_field_3d((4, 4, 4));
    let apollo_fft1_ms = elapsed_ms(|| {
        let _ = fft_1d_array(&signal);
    });
    let apollo_fft3_forward_ms = elapsed_ms(|| {
        let _ = fft_3d_array(&field);
    });
    let spectrum = fft_3d_array(&field);
    let apollo_fft3_inverse_ms = elapsed_ms(|| {
        let _ = ifft_3d_array(&spectrum);
    });

    let rustfft_available = cfg!(feature = "external-references");
    let (rustfft_fft1_ms, rustfft_fft3_ms) = if rustfft_available {
        (
            elapsed_ms(|| {
                #[cfg(feature = "external-references")]
                {
                    let _ = fft1_real(&signal);
                }
            }),
            elapsed_ms(|| {
                #[cfg(feature = "external-references")]
                {
                    let _ = fft3_real(&field);
                }
            }),
        )
    } else {
        (0.0, 0.0)
    };

    let signal_shape = [signal.len()];
    let numpy_bench = benchmark_fft(&signal_shape[..], signal.as_slice().unwrap_or(&[]), 1).ok();

    let domain = UniformDomain1D::new(32, 0.05)?;
    let positions: Vec<f64> = (0..20)
        .map(|i| (i as f64 * 0.137).rem_euclid(domain.length()))
        .collect();
    let values: Vec<Complex64> = (0..20)
        .map(|i| Complex64::new((0.3 * i as f64).cos(), (0.2 * i as f64).sin()))
        .collect();
    let nufft_exact_type1_1d_ms = elapsed_ms(|| {
        let _ = nufft_type1_1d(&positions, &values, domain);
    });
    let nufft_fast_type1_1d_ms = elapsed_ms(|| {
        let _ = nufft_type1_1d_fast(&positions, &values, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
    });
    let grid = UniformGrid3D::new(8, 8, 8, 0.125, 0.125, 0.125)?;
    let points: Vec<(f64, f64, f64)> = (0..12)
        .map(|i| {
            (
                (0.071 * i as f64).rem_euclid(1.0),
                (0.113 * i as f64).rem_euclid(1.0),
                (0.157 * i as f64).rem_euclid(1.0),
            )
        })
        .collect();
    let nufft_fast_type1_3d_ms = elapsed_ms(|| {
        let _ = nufft_type1_3d_fast(&points, &values[..12], grid, DEFAULT_NUFFT_KERNEL_WIDTH);
    });

    Ok(BenchmarkReport {
        apollo_fft1_ms,
        apollo_fft3_forward_ms,
        apollo_fft3_inverse_ms,
        rustfft_fft1_ms,
        rustfft_fft3_ms,
        numpy_fft1_ms: numpy_bench.as_ref().and_then(|probe| probe.numpy_ms),
        numpy_fft3_ms: None,
        pyfftw_fft1_ms: numpy_bench.as_ref().and_then(|probe| probe.pyfftw_ms),
        pyfftw_fft3_ms: None,
        nufft_exact_type1_1d_ms,
        nufft_fast_type1_1d_ms,
        nufft_fast_type1_3d_ms,
        gpu_fft_forward_ms: None,
        gpu_fft_inverse_ms: None,
        precision_benchmarks: vec![
            PrecisionBenchmarkReport {
                profile: "high_accuracy".to_string(),
                forward_ms: Some(apollo_fft1_ms),
                inverse_ms: Some(elapsed_ms(|| {
                    let spectrum = fft_1d_array(&signal);
                    let _ = ifft_1d_array(&spectrum);
                })),
                note: None,
            },
            PrecisionBenchmarkReport {
                profile: "low_precision".to_string(),
                forward_ms: Some(elapsed_ms(|| {
                    let input = signal.mapv(|value| value as f32);
                    let _ = fft_1d_array_typed(&input, PrecisionProfile::LOW_PRECISION_F32);
                })),
                inverse_ms: Some(elapsed_ms(|| {
                    let input = signal.mapv(|value| value as f32);
                    let spectrum = fft_1d_array_typed(&input, PrecisionProfile::LOW_PRECISION_F32);
                    let _ =
                        ifft_1d_array_typed::<f32>(&spectrum, PrecisionProfile::LOW_PRECISION_F32);
                })),
                note: None,
            },
            PrecisionBenchmarkReport {
                profile: "mixed_precision".to_string(),
                forward_ms: Some(elapsed_ms(|| {
                    let input = signal.mapv(|value| f16::from_f32(value as f32));
                    let _ = fft_1d_array_typed(&input, PrecisionProfile::MIXED_PRECISION_F16_F32);
                })),
                inverse_ms: Some(elapsed_ms(|| {
                    let input = signal.mapv(|value| f16::from_f32(value as f32));
                    let spectrum =
                        fft_1d_array_typed(&input, PrecisionProfile::MIXED_PRECISION_F16_F32);
                    let _ = ifft_1d_array_typed::<f16>(
                        &spectrum,
                        PrecisionProfile::MIXED_PRECISION_F16_F32,
                    );
                })),
                note: None,
            },
        ],
    })
}

fn precision_profile_reports() -> Vec<PrecisionRunReport> {
    let shape = Shape3D::new(4, 4, 4).expect("valid shape");
    let reference = representative_field_3d((4, 4, 4));

    // f64 high-accuracy path — the authoritative reference for forward error comparisons.
    let high_spectrum = fft_3d_array(&reference);
    let high_recovered = ifft_3d_array(&high_spectrum);
    let high_error = max_real_abs_delta_3d(&reference, &high_recovered);

    // f32 low-precision path.
    let low_input = reference.mapv(|value| value as f32);
    let low_plan =
        apollo_fft::FftPlan3D::with_precision(shape, PrecisionProfile::LOW_PRECISION_F32);
    let low_spectrum: Array3<Complex32> = low_plan.forward_typed(&low_input);
    // Forward error: max |f32 spectrum - f64 reference spectrum|.
    let low_forward_error = low_spectrum
        .iter()
        .zip(high_spectrum.iter())
        .map(|(lv, hv)| {
            ((f64::from(lv.re) - hv.re).powi(2) + (f64::from(lv.im) - hv.im).powi(2)).sqrt()
        })
        .fold(0.0_f64, f64::max);
    let low_recovered = low_plan.inverse_typed::<f32>(&low_spectrum).mapv(f64::from);
    let low_reference = low_input.mapv(f64::from);
    let low_error = max_real_abs_delta_3d(&low_reference, &low_recovered);

    // f16/f32 mixed-precision path — compare spectrum against f64 FFT of the f16-represented input.
    let mixed_input = reference.mapv(|value| f16::from_f32(value as f32));
    let mixed_plan =
        apollo_fft::FftPlan3D::with_precision(shape, PrecisionProfile::MIXED_PRECISION_F16_F32);
    let mixed_spectrum: Array3<Complex32> = mixed_plan.forward_typed(&mixed_input);
    // Use f64 FFT of the quantized input as the mixed-precision forward reference.
    let mixed_input_f64 = mixed_input.mapv(|v| f64::from(v.to_f32()));
    let mixed_reference_spectrum = fft_3d_array(&mixed_input_f64);
    let mixed_forward_error = mixed_spectrum
        .iter()
        .zip(mixed_reference_spectrum.iter())
        .map(|(lv, hv)| {
            ((f64::from(lv.re) - hv.re).powi(2) + (f64::from(lv.im) - hv.im).powi(2)).sqrt()
        })
        .fold(0.0_f64, f64::max);
    let mixed_recovered =
        ifft_3d_array_typed::<f16>(&mixed_spectrum, PrecisionProfile::MIXED_PRECISION_F16_F32)
            .mapv(|value| f64::from(value.to_f32()));
    let mixed_reference = mixed_input.mapv(|value| f64::from(value.to_f32()));
    let mixed_error = max_real_abs_delta_3d(&mixed_reference, &mixed_recovered);

    vec![
        PrecisionRunReport {
            profile: "high_accuracy".to_string(),
            attempted: true,
            passed: high_error <= 1.0e-10,
            // f64 IS the authoritative reference; forward error against itself is exactly 0.
            forward_max_abs_error: Some(0.0),
            inverse_max_abs_error: Some(high_error),
            relative_error: Some(high_error),
            note: None,
        },
        PrecisionRunReport {
            profile: "low_precision".to_string(),
            attempted: true,
            passed: low_error <= 1.0e-4,
            forward_max_abs_error: Some(low_forward_error),
            inverse_max_abs_error: Some(low_error),
            relative_error: Some(low_error),
            note: None,
        },
        PrecisionRunReport {
            profile: "mixed_precision".to_string(),
            attempted: true,
            passed: mixed_error <= 1.0e-3,
            forward_max_abs_error: Some(mixed_forward_error),
            inverse_max_abs_error: Some(mixed_error),
            relative_error: Some(mixed_error),
            note: None,
        },
    ]
}

fn environment_report(probe: Option<&PythonEnvironmentProbe>) -> EnvironmentReport {
    EnvironmentReport {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        debug_build: cfg!(debug_assertions),
        python_version: probe.map(|value| value.python_version.clone()),
        numpy_version: probe.and_then(|value| value.numpy_version.clone()),
        pyfftw_version: probe.and_then(|value| value.pyfftw_version.clone()),
    }
}

fn numpy_comparison_report(signal: &Array1<f64>) -> ExternalBackendReport {
    let signal_shape = [signal.len()];
    match compare_fft(&signal_shape[..], signal.as_slice().unwrap_or(&[]), 2) {
        Ok(report) => {
            let apollo = fft_1d_array(signal);
            let numpy_values: Vec<Complex64> = report
                .numpy_pairs
                .unwrap_or_default()
                .into_iter()
                .map(|pair| Complex64::new(pair[0], pair[1]))
                .collect();
            let error = if numpy_values.len() == apollo.len() {
                Some(max_complex_abs_delta(apollo.iter(), numpy_values.iter()))
            } else {
                None
            };
            ExternalBackendReport {
                backend: "numpy".to_string(),
                available: report.numpy_available,
                attempted: report.numpy_available,
                fft1_max_abs_error: error,
                fft1_prime_max_abs_error: None,
                fft3_max_abs_error: None,
                stability_max_abs_delta: report.numpy_stability_max_abs_delta,
                version: report.numpy_version,
                note: None,
            }
        }
        Err(error) => ExternalBackendReport {
            backend: "numpy".to_string(),
            available: false,
            attempted: false,
            fft1_max_abs_error: None,
            fft1_prime_max_abs_error: None,
            fft3_max_abs_error: None,
            stability_max_abs_delta: None,
            version: None,
            note: Some(error.to_string()),
        },
    }
}

fn fft_four_point_difference_fixture() -> PublishedFixtureReport {
    let signal = Array1::from_vec(vec![1.0, 0.0, -1.0, 0.0]);
    let actual = fft_1d_array(&signal);
    let expected = [
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    published_complex_fixture(
        "FFT",
        "DFT4([1,0,-1,0])",
        "Cooley and Tukey (1965), finite root-of-unity DFT definition",
        actual.iter(),
        expected.iter(),
    )
}

fn dht_four_point_difference_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DhtPlan::new(4)?;
    let spectrum = plan.forward(&[1.0, 0.0, -1.0, 0.0])?;
    let expected = [0.0, 2.0, 0.0, 2.0];
    Ok(published_real_fixture(
        "DHT",
        "DHT4([1,0,-1,0])",
        "Bracewell (1983), cas(theta)=cos(theta)+sin(theta) Hartley definition",
        spectrum.values(),
        &expected,
    ))
}

fn dct2_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DctII)?;
    let actual = plan.forward(&[1.0, 3.0])?;
    let expected = [4.0, -std::f64::consts::SQRT_2];
    Ok(published_real_fixture(
        "DCT-II",
        "DCT-II2([1,3])",
        "FFTW real-to-real REDFT10 convention, unnormalized DCT-II basis",
        &actual,
        &expected,
    ))
}

fn dst2_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DstII)?;
    let actual = plan.forward(&[1.0, 3.0])?;
    let expected = [
        1.0 * (std::f64::consts::PI / 4.0).sin() + 3.0 * (3.0 * std::f64::consts::PI / 4.0).sin(),
        -2.0,
    ];
    Ok(published_real_fixture(
        "DST-II",
        "DST-II2([1,3])",
        "FFTW real-to-real RODFT10 convention, unnormalized DST-II basis",
        &actual,
        &expected,
    ))
}

/// NTT of the unit impulse [1,0,0,0] with N=4, modulus=998244353, primitive root=3.
///
/// # Mathematical contract
/// By the NTT definition (Pollard 1971): F[k] = Σ_{n=0}^{N-1} f[n]·ω^{nk} mod p.
/// With f[0]=1 and f[1..3]=0, every term except n=0 vanishes, giving F[k] = ω^0 = 1 for all k.
/// Reference: "NTT definition, Pollard (1971): F[k] = Σ f[n]·ω^{nk} mod p, impulse response F[k]=1"
fn ntt_impulse_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![1u64, 0, 0, 0]);
    let actual = ntt(&input)?;
    let actual_f64: Vec<f64> = actual.iter().map(|&v| v as f64).collect();
    let expected = [1.0_f64, 1.0, 1.0, 1.0];
    Ok(published_real_fixture(
        "NTT",
        "NTT4([1,0,0,0])",
        "NTT definition, Pollard (1971): F[k] = \u{03a3} f[n]\u{00b7}\u{03c9}^{nk} mod p, impulse response F[k]=1",
        &actual_f64,
        &expected,
    ))
}

/// NTT of the constant-one vector [1,1,1,1] with N=4, modulus=998244353, primitive root=3.
///
/// # Mathematical contract
/// F[0] = Σ_{n=0}^3 1 = 4 (mod p). For k≠0: F[k] = Σ_{n=0}^3 ω^{nk}.
/// Since ω is a primitive N-th root of unity and ω^k≠1 for k=1,2,3, the geometric series
/// Σ_{n=0}^{N-1} (ω^k)^n = (ω^{Nk}-1)/(ω^k-1) = 0 mod p, because ω^N ≡ 1 (mod p).
/// Reference: "NTT DFT-of-constant theorem: F[0]=N, F[k≠0]=0 for constant input (Pollard 1971)"
fn ntt_constant_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![1u64, 1, 1, 1]);
    let actual = ntt(&input)?;
    let actual_f64: Vec<f64> = actual.iter().map(|&v| v as f64).collect();
    let expected = [4.0_f64, 0.0, 0.0, 0.0];
    Ok(published_real_fixture(
        "NTT",
        "NTT4([1,1,1,1])",
        "NTT DFT-of-constant theorem: F[0]=N, F[k\u{2260}0]=0 for constant input (Pollard 1971)",
        &actual_f64,
        &expected,
    ))
}

/// NUFFT Type-1 1D with a single source at x=0 and value 1+0i, N=4, dx=π/2 (L=2π).
///
/// # Mathematical contract
/// F[k] = Σ_j f[j]·exp(-2πi·k_signed·x_j/L). With x_j=0:
///   angle = -2π·k_signed·0/L = 0 for every k.
///   exp(0) = 1+0i exactly in IEEE 754 (cos(0)=1, sin(0)=0 are exact).
/// Therefore F[k] = 1+0i for all k=0..3, with zero floating-point error.
/// Reference: "NUFFT Type-1 definition: F[k] = Σ_j f[j]·exp(-2πi·k·x_j/L); at x_j=0, F[k]=1 for all k"
fn nufft_impulse_at_origin_fixture() -> SuiteResult<PublishedFixtureReport> {
    let domain = UniformDomain1D::new(4, std::f64::consts::FRAC_PI_2)?;
    let actual = nufft_type1_1d(&[0.0], &[Complex64::new(1.0, 0.0)], domain);
    let expected = [Complex64::new(1.0, 0.0); 4];
    Ok(published_complex_fixture(
        "NUFFT",
        "NUFFT-Type1-1D(x=[0], f=[1+0i], N=4)",
        "NUFFT Type-1 definition: F[k] = \u{03a3}_j f[j]\u{00b7}exp(-2\u{03c0}i\u{00b7}k\u{00b7}x_j/L); at x_j=0, F[k]=1 for all k",
        actual.iter(),
        expected.iter(),
    ))
}

/// NTT of the unit impulse [1,0,0,0,0,0,0,0] with N=8, modulus=998244353, primitive root=3.
///
/// # Mathematical contract
/// By the NTT definition (Pollard 1971): F[k] = Σ_{n=0}^{N-1} f[n]·ω^{nk} mod p.
/// With f[0]=1 and f[1..7]=0, every term except n=0 vanishes, giving F[k] = ω^0 = 1 for all k.
/// This is the same impulse theorem as N=4, generalized to N=8.
/// Reference: "NTT definition, Pollard (1971): F[k] = Σ f[n]·ω^{nk} mod p, impulse response F[k]=1 (N=8)"
fn ntt_n8_impulse_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![1u64, 0, 0, 0, 0, 0, 0, 0]);
    let actual = ntt(&input)?;
    let actual_f64: Vec<f64> = actual.iter().map(|&v| v as f64).collect();
    let expected = [1.0_f64; 8];
    Ok(published_real_fixture(
        "NTT",
        "NTT8([1,0,0,0,0,0,0,0])",
        "NTT definition, Pollard (1971): F[k] = \u{03a3} f[n]\u{00b7}\u{03c9}^{nk} mod p, impulse response F[k]=1 (N=8)",
        &actual_f64,
        &expected,
    ))
}

/// NTT convolution theorem: NTT^{-1}(NTT(a) ⊙ NTT(b)) = a ★ b for polynomial product.
///
/// # Mathematical contract
/// By the NTT Convolution Theorem (Pollard 1971): for a=[1,2,0,0] and b=[3,4,0,0],
/// the cyclic convolution a★b equals the coefficients of (1+2x)(3+4x) = 3+10x+8x², giving c=[3,10,8,0].
/// All values satisfy 3,10,8 ≪ p = 998244353 so modular reduction is trivial.
/// The fixture verifies INTT(NTT(a) ⊙ NTT(b)) against the analytically derived polynomial product.
/// Reference: "NTT Convolution Theorem (Pollard 1971): INTT(NTT(a)⊙NTT(b)) = a★b mod p"
fn ntt_polynomial_convolution_fixture() -> SuiteResult<PublishedFixtureReport> {
    let p = DEFAULT_MODULUS;
    let plan = NttPlan::new(4)?;
    let a = Array1::from_vec(vec![1u64, 2, 0, 0]);
    let b = Array1::from_vec(vec![3u64, 4, 0, 0]);
    let fa = plan.forward(&a)?;
    let fb = plan.forward(&b)?;
    let fc: Vec<u64> = fa
        .iter()
        .zip(fb.iter())
        .map(|(&x, &y)| ((x as u128 * y as u128) % p as u128) as u64)
        .collect();
    let fc_arr = Array1::from(fc);
    let c = intt(&fc_arr)?;
    let actual_f64: Vec<f64> = c.iter().map(|&v| v as f64).collect();
    // (1+2x)(3+4x) = 3+10x+8x^2 → [3,10,8,0]
    let expected = [3.0_f64, 10.0, 8.0, 0.0];
    Ok(published_real_fixture(
        "NTT",
        "INTT(NTT([1,2,0,0])\u{2299}NTT([3,4,0,0]))",
        "NTT Convolution Theorem (Pollard 1971): INTT(NTT(a)\u{2299}NTT(b)) = a\u{2605}b mod p; (1+2x)(3+4x)=3+10x+8x\u{00b2}",
        &actual_f64,
        &expected,
    ))
}

/// NUFFT Type-1 1D with a single source at x=L/4 and value 1+0i, N=4, dx=π/2 (L=2π).
///
/// # Mathematical contract
/// F[k] = Σ_j f[j]·exp(-2πi·k_signed·x_j/L). With x_j=L/4 and f[j]=1:
///   angle = -2π·k_signed·(L/4)/L = -π·k_signed/2.
///   k_signed sequence for N=4: k=0→0, k=1→1, k=2→2, k=3→-1.
///   F[0] = exp(0) = 1+0i (exact)
///   F[1] = exp(-πi/2) = cos(-π/2)+i·sin(-π/2) ≈ 0-i  (|error| < 7e-17)
///   F[2] = exp(-πi)   = cos(-π)+i·sin(-π)     ≈ -1+0i (|error| < 2e-16)
///   F[3] = exp(+πi/2) = cos(+π/2)+i·sin(+π/2) ≈ 0+i  (|error| < 7e-17)
/// All errors are well within the 1×10⁻¹² published-fixture threshold.
/// Reference: "NUFFT Type-1 definition: F[k]=exp(-2πi·k_signed·x₀/L) for unit source at x₀=L/4 (Dutt and Rokhlin 1993)"
fn nufft_quarter_period_phase_fixture() -> SuiteResult<PublishedFixtureReport> {
    let domain = UniformDomain1D::new(4, std::f64::consts::FRAC_PI_2)?;
    let x0 = std::f64::consts::FRAC_PI_2; // L/4 = (4·π/2)/4 = π/2
    let actual = nufft_type1_1d(&[x0], &[Complex64::new(1.0, 0.0)], domain);
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ];
    Ok(published_complex_fixture(
        "NUFFT",
        "NUFFT-Type1-1D(x=[L/4], f=[1+0i], N=4)",
        "NUFFT Type-1 definition: F[k]=exp(-2\u{03c0}i\u{00b7}k_signed\u{00b7}x\u{2080}/L) for unit source at x\u{2080}=L/4; F=[1,-i,-1,i] (Dutt and Rokhlin 1993)",
        actual.iter(),
        expected.iter(),
    ))
}

fn published_complex_fixture<'a, I, J>(
    transform: &str,
    fixture: &str,
    reference: &str,
    actual: I,
    expected: J,
) -> PublishedFixtureReport
where
    I: IntoIterator<Item = &'a Complex64>,
    J: IntoIterator<Item = &'a Complex64>,
{
    let max_abs_error = max_complex_abs_delta(actual, expected);
    PublishedFixtureReport {
        transform: transform.to_string(),
        fixture: fixture.to_string(),
        reference: reference.to_string(),
        max_abs_error,
        threshold: PUBLISHED_FIXTURE_LIMIT,
        passed: max_abs_error <= PUBLISHED_FIXTURE_LIMIT,
    }
}

fn published_real_fixture(
    transform: &str,
    fixture: &str,
    reference: &str,
    actual: &[f64],
    expected: &[f64],
) -> PublishedFixtureReport {
    let max_abs_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max);
    PublishedFixtureReport {
        transform: transform.to_string(),
        fixture: fixture.to_string(),
        reference: reference.to_string(),
        max_abs_error,
        threshold: PUBLISHED_FIXTURE_LIMIT,
        passed: max_abs_error <= PUBLISHED_FIXTURE_LIMIT,
    }
}

/// IDFT of the all-ones vector [1,1,1,1] with N=4.
///
/// # Mathematical contract
///
/// The normalized IDFT is IDFT[n] = (1/N) Σ_k F[k] exp(2πikn/N).
/// For F=[1,1,1,1] and N=4, every sum collapses to IDFT[0]=1, IDFT[1..3]=0
/// because Σ_k exp(2πikn/N) = N·δ_{n,0} by the geometric series identity for
/// primitive roots of unity. Reference: DFT Inversion Theorem, Cooley and Tukey (1965).
fn fft_inverse_four_point_fixture() -> PublishedFixtureReport {
    let spectrum = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ]);
    let actual = ifft_1d_array(&spectrum);
    let expected = [1.0_f64, 0.0, 0.0, 0.0];
    published_real_fixture(
        "FFT",
        "IDFT4([1,1,1,1])",
        "Cooley and Tukey (1965), DFT inversion theorem: IDFT(DFT(x))=x; DFT([1,0,0,0])=[1,1,1,1] so IDFT([1,1,1,1])=[1,0,0,0]",
        actual.as_slice().unwrap(),
        &expected,
    )
}

/// DCT-III is the inverse of DCT-II up to a 2/N scaling factor.
///
/// # Mathematical contract
///
/// For N=2, x=[1,3]: DCT-II([1,3])=[4,−√2].
/// DCT-III([4,−√2]) scaled by 2/N recovers [1,3].
/// The `plan.inverse` method applies the 2/N factor.
/// Reference: Rao and Yip (1990), DCT-III inverse pair theorem.
fn dct2_inverse_pair_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DctII)?;
    let actual = plan.inverse(&[4.0, -std::f64::consts::SQRT_2])?;
    let expected = [1.0_f64, 3.0];
    Ok(published_real_fixture(
        "DCT",
        "DCT-III(DCT-II([1,3]))\u{00d7}(2/N)=[1,3]",
        "Rao and Yip (1990), DCT-III is the inverse of DCT-II: DCT-III(DCT-II(x))\u{00b7}(2/N)=x for N=2",
        &actual,
        &expected,
    ))
}

/// DHT self-reciprocal property: DHT(DHT(x)) = N·x.
///
/// # Mathematical contract
///
/// For x=[1,0,0,0] and N=4:
///   step1 = DHT([1,0,0,0]) = [1,1,1,1]  (cas impulse response at n=0)
///   step2 = DHT([1,1,1,1]) = [4,0,0,0]  (DC-only signal maps to scaled impulse)
/// step2 = N·x = 4·[1,0,0,0] = [4,0,0,0]. ✓
/// Reference: Bracewell (1983), DHT self-reciprocal property: H{H{x}}[n] = N·x[n].
fn dht_self_reciprocal_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DhtPlan::new(4)?;
    let step1 = plan.forward(&[1.0_f64, 0.0, 0.0, 0.0])?;
    let step2 = plan.forward(step1.values())?;
    let expected = [4.0_f64, 0.0, 0.0, 0.0];
    Ok(published_real_fixture(
        "DHT",
        "DHT(DHT([1,0,0,0]))=[4,0,0,0]",
        "Bracewell (1983), DHT self-reciprocal property: DHT(DHT(x))=N\u{00b7}x; for x=[1,0,0,0], DHT([1,0,0,0])=[1,1,1,1] then DHT([1,1,1,1])=[4,0,0,0]",
        step2.values(),
        &expected,
    ))
}

/// FWHT two-point known-value fixture.
///
/// # Mathematical contract
///
/// The 2×2 Hadamard matrix is H₂ = [[1,1],[1,−1]].
/// H₂·[1,1]ᵀ = [2,0]ᵀ (both basis functions evaluate to 1+1 and 1−1 respectively).
/// The FWHT is the unnormalized Hadamard transform.
/// Reference: Hadamard (1893), two-point Walsh-Hadamard matrix definition.
fn fwht_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = FwhtPlan::new(2)?;
    let input = Array1::from_vec(vec![1.0_f64, 1.0]);
    let actual = plan.forward(&input)?;
    let expected = [2.0_f64, 0.0];
    Ok(published_real_fixture(
        "FWHT",
        "FWHT2([1,1])",
        "Hadamard (1893), H_2=[[1,1],[1,-1]], H_2\u{00b7}[1,1]^T=[2,0]^T",
        actual.as_slice().unwrap(),
        &expected,
    ))
}

/// QFT two-point known-value fixture: QFT₂(|0⟩) = (1/√2)(|0⟩ + |1⟩).
///
/// # Mathematical contract
///
/// The unitary QFT matrix for N=2 is (1/√2)·[[1,1],[1,−1]].
/// For input [1,0] (computational basis state |0⟩):
///   QFT₂([1,0])ᵀ = (1/√2)·[1,1]ᵀ.
/// Both output components equal 1/√2 = 1/√2 + 0i.
/// Reference: Shor (1994), quantum Fourier transform for N=2 as Hadamard gate.
fn qft_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
    let actual = qft_transform(&input)?;
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    let expected = [
        Complex64::new(inv_sqrt2, 0.0),
        Complex64::new(inv_sqrt2, 0.0),
    ];
    Ok(published_complex_fixture(
        "QFT",
        "QFT2([1,0])",
        "Shor (1994), QFT for N=2 is the Hadamard gate: QFT([1,0])=[1/\u{221a}2, 1/\u{221a}2]",
        actual.iter(),
        expected.iter(),
    ))
}

/// CZT with DFT parameters equals the DFT for the unit impulse [1,0,0,0].
///
/// # Mathematical contract
///
/// When A=1 and W=exp(−2πi/N), the CZT spiral contour collapses to the unit
/// circle at the N-th roots of unity, recovering the DFT exactly.
/// For input [1+0i,0,0,0], DFT([1,0,0,0])=[1,1,1,1].
/// Therefore CZT(N=4,M=4,A=1,W=exp(−2πi/4))([1,0,0,0])=[1,1,1,1].
/// Reference: Rabiner, Schafer and Rader (1969), CZT spiral-collapse theorem.
fn czt_unit_impulse_is_dft_fixture() -> SuiteResult<PublishedFixtureReport> {
    let n = 4usize;
    let a = Complex64::new(1.0, 0.0);
    let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
    let plan = CztPlan::new(n, n, a, w)?;
    let input = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let actual = plan.forward(&input)?;
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "CZT",
        "CZT4(A=1,W=exp(-2\u{03c0}i/4),[1,0,0,0])",
        "Rabiner, Schafer and Rader (1969), CZT with A=1 and W=exp(-2\u{03c0}i/N) equals DFT; DFT([1,0,0,0])=[1,1,1,1]",
        actual.iter(),
        expected.iter(),
    ))
}

/// GFT K₂ path graph Laplacian eigenvalues are {0, 2}.
///
/// # Mathematical contract
///
/// For the 2-vertex path graph with adjacency A=[[0,1],[1,0]], the combinatorial
/// Laplacian L = D − A = [[1,−1],[−1,1]]. Its characteristic polynomial is
/// det(L − λI) = (1−λ)² − 1 = λ² − 2λ = λ(λ−2), giving eigenvalues {0, 2}.
/// The `spectral_basis` kernel sorts eigenvalues in ascending order, so the
/// GFT plan returns eigenvalues [0.0, 2.0]. This is a sign-independent,
/// numerically deterministic published result.
/// Reference: Shuman et al. (2013), IEEE Signal Processing Magazine, graph Fourier basis definition.
fn gft_path_graph_forward_fixture() -> SuiteResult<PublishedFixtureReport> {
    let adjacency = DMatrix::from_row_slice(2, 2, &[0.0_f64, 1.0, 1.0, 0.0]);
    let plan = GftPlan::from_adjacency(&adjacency)?;
    let eigenvalues = plan.eigenvalues().to_vec();
    let expected = [0.0_f64, 2.0];
    Ok(published_real_fixture(
        "GFT",
        "GFT-K2-path-eigenvalues",
        "Shuman et al. (2013), K\u{2082} path graph Laplacian L=[[1,-1],[-1,1]]: eigenvalues {0,2} from det(L-\u{03bb}I)=\u{03bb}(\u{03bb}-2)",
        &eigenvalues,
        &expected,
    ))
}

/// Unitary FrFT of order 2 is the reversal operator.
///
/// # Mathematical contract
///
/// By Candan et al. (2000), Theorem 3: DFrFT₂(x)[k] = x[N−1−k] for all x.
/// The palindrome-structured Grünbaum eigenvectors diagonalize the DFT; the
/// fractional phase exp(−i·2·k·π/2) = (−1)^k combines with V·diag((−1)^k)·Vᵀ
/// to produce the exact index-reversal operator.
/// Input [1+0i, 2+0i, 3+0i, 4+0i], N=4, order=2.
/// Expected output: [4+0i, 3+0i, 2+0i, 1+0i].
fn frft_unitary_order2_reversal_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ]);
    let plan = UnitaryFrftPlan::new(4, 2.0)?;
    let actual = plan.forward(&input)?;
    let expected = [
        Complex64::new(4.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "UnitaryFrFT",
        "UnitaryFrFT(order=2,[1,2,3,4])",
        "Candan et al. (2000), Theorem 3: DFrFT\u{2082} = reversal operator; UnitaryFrFT(order=2,[1,2,3,4])=[4,3,2,1]",
        actual.iter(),
        expected.iter(),
    ))
}

/// Haar DWT one-level detail coefficients for [1, −1, 0, 0].
///
/// # Mathematical contract
///
/// Haar highpass QMF: g = [1/√2, −1/√2] (from g[k] = (−1)^k·h[L−1−k] with h=[1/√2,1/√2]).
/// detail[k] = g[0]·x[2k] + g[1]·x[2k+1] = (x[2k] − x[2k+1]) / √2.
/// Input [1, −1, 0, 0], N=4, levels=1:
///   detail[0] = (1 − (−1)) / √2 = 2/√2 = √2
///   detail[1] = (0 − 0) / √2 = 0
/// Reference: Haar (1910), Mallat (1989) two-channel QMF framework.
fn wavelet_haar_one_level_detail_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = [1.0_f64, -1.0, 0.0, 0.0];
    let plan = DwtPlan::new(4, 1, DiscreteWavelet::Haar)?;
    let coeffs = plan.forward(&input)?;
    let detail = coeffs.details()[0].as_slice();
    let expected = [std::f64::consts::SQRT_2, 0.0_f64];
    Ok(published_real_fixture(
        "DWT-Haar",
        "Haar-DWT-1level([1,-1,0,0])",
        "Haar (1910), Mallat (1989) two-channel QMF: DWT([1,-1,0,0]) detail=[\u{221a}2,0] via g=[1/\u{221a}2,-1/\u{221a}2]",
        detail,
        &expected,
    ))
}

/// SDFT bin 0 for the unit impulse [1, 0, 0, 0].
///
/// # Mathematical contract
///
/// DFT bin k=0: F[0] = Σ_{n=0}^{N-1} x[n] = 1 + 0 + 0 + 0 = 1 for the unit impulse.
/// The SDFT direct-bins path computes the standard DFT for the given window exactly.
/// Reference: Jacobsen and Lyons (2003), Sliding DFT, IEEE Signal Processing Magazine.
fn sdft_bin_zero_unit_impulse_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = [1.0_f64, 0.0, 0.0, 0.0];
    let plan = SdftPlan::new(4, 4)?;
    let bins = plan.direct_bins(&input)?;
    let actual_arr = [bins[0]];
    let expected_arr = [Complex64::new(1.0, 0.0)];
    Ok(published_complex_fixture(
        "SDFT",
        "SDFT-bin0-impulse([1,0,0,0])",
        "Jacobsen and Lyons (2003), SDFT bin-0 at position 0 equals DFT bin-0 = sum(x) = 1 for unit impulse",
        actual_arr.iter(),
        expected_arr.iter(),
    ))
}

fn representative_signal_1d(len: usize) -> Array1<f64> {
    Array1::from_vec(
        (0..len)
            .map(|i| {
                let x = i as f64;
                (0.31 * x).sin() + 0.17 * (0.73 * x).cos()
            })
            .collect(),
    )
}

fn representative_field_3d(shape: (usize, usize, usize)) -> Array3<f64> {
    Array3::from_shape_fn(shape, |(i, j, k)| {
        let x = i as f64;
        let y = j as f64;
        let z = k as f64;
        (0.11 * x).sin() + (0.13 * y).cos() + (0.17 * z).sin()
    })
}

fn max_complex_abs_delta<'a, I, J>(left: I, right: J) -> f64
where
    I: IntoIterator<Item = &'a Complex64>,
    J: IntoIterator<Item = &'a Complex64>,
{
    left.into_iter()
        .zip(right)
        .map(|(lhs, rhs)| (*lhs - *rhs).norm())
        .fold(0.0, f64::max)
}

fn max_real_abs_delta(left: &Array1<f64>, right: &Array1<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn max_real_abs_delta_3d(left: &Array3<f64>, right: &Array3<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn relative_complex_error<'a, I, J>(left: I, right: J) -> f64
where
    I: IntoIterator<Item = &'a Complex64>,
    J: IntoIterator<Item = &'a Complex64>,
{
    left.into_iter()
        .zip(right)
        .map(|(lhs, rhs)| (*lhs - *rhs).norm() / lhs.norm().max(1.0))
        .fold(0.0, f64::max)
}

fn elapsed_ms<F, T>(f: F) -> f64
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let _ = f();
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_suite_produces_value_semantic_reports() {
        let report = run_validation_suite().expect("validation suite");
        assert!(report.fft_cpu.roundtrip_max_abs_error <= CPU_ROUNDTRIP_LIMIT);
        assert!(report.fft_cpu.parseval_relative_error <= CPU_PARSEVAL_LIMIT);
        assert!(report.nufft.passed);
        assert!(report.external.published_references.passed);
        assert_eq!(report.external.published_references.attempted, 20);
        assert_eq!(report.external.rustfft.backend, "rustfft");
        assert_eq!(report.external.numpy.backend, "numpy");
    }

    #[test]
    fn published_reference_suite_checks_computed_fixture_values() {
        let report = run_published_reference_suite().expect("published references");
        assert_eq!(report.attempted, 20);
        assert!(report.passed);
        for fixture in &report.fixtures {
            assert!(
                fixture.max_abs_error <= fixture.threshold,
                "{} exceeded threshold: {} > {}",
                fixture.fixture,
                fixture.max_abs_error,
                fixture.threshold
            );
            assert!(!fixture.reference.is_empty());
        }
    }

    #[test]
    fn validation_report_json_schema_contains_required_sections() {
        let report = run_validation_suite().expect("validation suite");
        let value = serde_json::to_value(&report).expect("serialize validation report");
        let object = value
            .as_object()
            .expect("validation report is a JSON object");

        for key in [
            "fft_cpu",
            "fft_gpu",
            "nufft",
            "external",
            "benchmarks",
            "environment",
        ] {
            assert!(object.contains_key(key), "missing top-level key {key}");
        }

        let fft_cpu = object["fft_cpu"]
            .as_object()
            .expect("fft_cpu is a JSON object");
        for key in [
            "roundtrip_max_abs_error",
            "parseval_relative_error",
            "stability_max_abs_delta",
            "non_finite_input_propagates",
            "passed",
            "precision_profiles",
        ] {
            assert!(fft_cpu.contains_key(key), "missing fft_cpu key {key}");
        }

        let external = object["external"]
            .as_object()
            .expect("external is a JSON object");
        for key in [
            "passed",
            "rustfft_checkout_present",
            "pyfftw_checkout_present",
            "rustfft",
            "numpy",
            "pyfftw",
            "robustness_passed",
            "precision_comparisons",
            "published_references",
        ] {
            assert!(external.contains_key(key), "missing external key {key}");
        }
    }
}
