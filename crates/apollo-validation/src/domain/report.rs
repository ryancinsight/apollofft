//! Validation report structures.

use serde::{Deserialize, Serialize};

/// Structured summary of the Apollo validation matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// CPU FFT validation results.
    pub fft_cpu: CpuFftReport,
    /// GPU FFT validation results.
    pub fft_gpu: GpuFftReport,
    /// NUFFT validation results.
    pub nufft: NufftReport,
    /// External comparison results.
    pub external: ExternalComparisonReport,
    /// Benchmark timings collected during validation.
    pub benchmarks: BenchmarkReport,
    /// Execution environment metadata.
    pub environment: EnvironmentReport,
}

/// CPU FFT validation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuFftReport {
    /// Maximum absolute error for a 3D roundtrip.
    pub roundtrip_max_abs_error: f64,
    /// Relative Parseval error for a representative 1D signal.
    pub parseval_relative_error: f64,
    /// Whether the CPU validation thresholds passed.
    pub passed: bool,
}

/// GPU FFT validation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuFftReport {
    /// Whether the GPU surface crate is linked.
    pub surface_reported_available: bool,
    /// Whether an adapter/device-backed validation run was attempted.
    pub attempted: bool,
    /// Whether the attempted validation run succeeded.
    pub passed: bool,
    /// Forward comparison max absolute error against Apollo CPU.
    pub forward_max_abs_error: Option<f64>,
    /// Inverse comparison max absolute error against Apollo CPU.
    pub inverse_max_abs_error: Option<f64>,
    /// Optional note explaining skip/failure details.
    pub note: Option<String>,
}

/// NUFFT validation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NufftReport {
    /// Relative error for fast 1D type-1 against exact direct evaluation.
    pub type1_1d_max_relative_error: f64,
    /// Relative error for fast 1D type-2 against exact direct evaluation.
    pub type2_1d_max_relative_error: f64,
    /// Relative error for fast 3D type-1 against exact direct evaluation.
    pub type1_3d_max_relative_error: f64,
    /// Whether the NUFFT validation thresholds passed.
    pub passed: bool,
}

/// Comparison with external FFT implementations or assets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalComparisonReport {
    /// Whether the external comparison thresholds passed.
    pub passed: bool,
    /// Direct comparison against the `rustfft` crate on a representative 1D signal.
    pub rustfft_max_abs_error: f64,
    /// Whether an external `rustfft` checkout exists under the workspace.
    pub rustfft_checkout_present: bool,
    /// Whether an external `pyfftw` checkout exists under the workspace.
    pub pyfftw_checkout_present: bool,
    /// Optional note explaining skipped comparisons.
    pub note: Option<String>,
}

/// Benchmark timings collected during validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// CPU 3D FFT forward wall time in milliseconds.
    pub cpu_fft_forward_ms: f64,
    /// CPU 3D FFT inverse wall time in milliseconds.
    pub cpu_fft_inverse_ms: f64,
    /// Exact direct 1D type-1 NUFFT wall time in milliseconds.
    pub nufft_exact_type1_1d_ms: f64,
    /// Fast 1D type-1 NUFFT wall time in milliseconds.
    pub nufft_fast_type1_1d_ms: f64,
    /// Fast 3D type-1 NUFFT wall time in milliseconds.
    pub nufft_fast_type1_3d_ms: f64,
    /// GPU FFT forward wall time in milliseconds when attempted.
    pub gpu_fft_forward_ms: Option<f64>,
    /// GPU FFT inverse wall time in milliseconds when attempted.
    pub gpu_fft_inverse_ms: Option<f64>,
}

/// Environment metadata captured during a validation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentReport {
    /// Host operating system.
    pub os: String,
    /// Host architecture.
    pub arch: String,
    /// Whether debug assertions were enabled for the build.
    pub debug_build: bool,
}
