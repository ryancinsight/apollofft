//! Python and NumPy probing helpers used by the validation suites.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Environment details reported by the NumPy harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonEnvironmentProbe {
    /// Python version string.
    pub python_version: String,
    /// NumPy version when import succeeded.
    pub numpy_version: Option<String>,
    /// `pyfftw` version when import succeeded.
    pub pyfftw_version: Option<String>,
}

/// Result of a NumPy-backed FFT parity run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonFftComparison {
    /// Whether NumPy was available.
    pub numpy_available: bool,
    /// Whether `pyfftw` was available.
    pub pyfftw_available: bool,
    /// Flattened complex spectrum pairs from NumPy.
    pub numpy_pairs: Option<Vec<[f64; 2]>>,
    /// Flattened complex spectrum pairs from `pyfftw`.
    pub pyfftw_pairs: Option<Vec<[f64; 2]>>,
    /// Repeated-run stability delta from NumPy.
    pub numpy_stability_max_abs_delta: Option<f64>,
    /// Repeated-run stability delta from `pyfftw`.
    pub pyfftw_stability_max_abs_delta: Option<f64>,
    /// NumPy version when available.
    pub numpy_version: Option<String>,
    /// `pyfftw` version when available.
    pub pyfftw_version: Option<String>,
}

/// Result of a Python-side FFT benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonBenchmarkProbe {
    /// Whether NumPy benchmarking was available.
    pub numpy_available: bool,
    /// Whether `pyfftw` benchmarking was available.
    pub pyfftw_available: bool,
    /// NumPy timing in milliseconds.
    pub numpy_ms: Option<f64>,
    /// `pyfftw` timing in milliseconds.
    pub pyfftw_ms: Option<f64>,
}

#[derive(Debug, Serialize)]
struct HarnessRequest<'a> {
    mode: &'a str,
    shape: Vec<usize>,
    real_input: Option<&'a [f64]>,
    iterations: Option<usize>,
    repeats: Option<usize>,
}

fn harness_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("numpy_fft_probe.py")
}

fn run_harness<T: for<'de> Deserialize<'de>>(
    request: &HarnessRequest<'_>,
) -> Result<T, Box<dyn std::error::Error>> {
    let payload = serde_json::to_vec(request)?;
    let mut child = Command::new("python")
        .arg(harness_script())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .ok_or("python harness stdin not available")?
        .write_all(&payload)?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!("python harness failed: {stderr}").into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// Probe Python/NumPy environment metadata.
pub fn probe_python_environment() -> Result<PythonEnvironmentProbe, Box<dyn std::error::Error>> {
    run_harness(&HarnessRequest {
        mode: "environment",
        shape: Vec::new(),
        real_input: None,
        iterations: None,
        repeats: None,
    })
}

/// Compare Apollo output against NumPy and optional `pyfftw`.
pub fn compare_fft(
    shape: &[usize],
    real_input: &[f64],
    repeats: usize,
) -> Result<PythonFftComparison, Box<dyn std::error::Error>> {
    run_harness(&HarnessRequest {
        mode: "compare_fft",
        shape: shape.to_vec(),
        real_input: Some(real_input),
        iterations: None,
        repeats: Some(repeats),
    })
}

/// Benchmark NumPy and optional `pyfftw`.
pub fn benchmark_fft(
    shape: &[usize],
    real_input: &[f64],
    iterations: usize,
) -> Result<PythonBenchmarkProbe, Box<dyn std::error::Error>> {
    run_harness(&HarnessRequest {
        mode: "benchmark_fft",
        shape: shape.to_vec(),
        real_input: Some(real_input),
        iterations: Some(iterations),
        repeats: None,
    })
}
