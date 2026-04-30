//! Criterion benchmarks for STFT-WGPU forward and inverse FFT-accelerated paths.
//!
//! Measures wall-clock cost (GPU dispatch + PCIe readback) for the GPU forward
//! STFT (O(N log N) per frame, Radix-2 DIT) and GPU inverse STFT (O(N log N) per
//! frame, Radix-2 DIT) across three `(frame_len, hop_len, signal_len)` parameter sets.
//!
//! # Mathematical justification
//!
//! STFT forward: X[m, k] = Σ_{n=0}^{N−1} w_a[n] · x[m·hop − N/2 + n] · exp(−2πi·k·n/N)
//!   N = `frame_len` (must be a power of two for the Radix-2 path).
//! STFT inverse WOLA: y = OLA(IDFT(X) · w_s) / Σ_m w_s²  (COLA normalisation).
//! Parameter sets satisfy hop = frame_len / 2 (Hann COLA condition).
//!
//! Signal is a sum of two bin-aligned sinusoids (k₁=16, k₂=64) so that the
//! DFT spectrum is analytically exact (no spectral leakage), giving a
//! meaningful and stable benchmark workload.

#![allow(missing_docs)]

use apollo_stft_wgpu::{StftWgpuBackend, StftWgpuPlan};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Parameter sets: `(frame_len, hop_len, signal_len)`.
/// Each hop = frame_len / 2 satisfies the Hann COLA condition.
const PARAMS: &[(usize, usize, usize)] = &[(256, 128, 4096), (512, 256, 8192), (1024, 512, 16384)];

/// Acquire a WGPU backend, or return `None` if no adapter is available.
fn try_backend() -> Option<StftWgpuBackend> {
    StftWgpuBackend::try_default().ok()
}

/// Analytical signal: sum of two bin-aligned sinusoids.
///
/// `s[n] = sin(2π · k₁ · n / N) + 0.5 · sin(2π · k₂ · n / N)`, k₁=16, k₂=64.
/// Bin alignment guarantees exact DFT spectrum (zero spectral leakage).
fn analytical_signal(signal_len: usize, frame_len: usize) -> Vec<f32> {
    (0..signal_len)
        .map(|n| {
            let t = n as f32;
            (2.0 * std::f32::consts::PI * 16.0 * t / frame_len as f32).sin()
                + 0.5 * (2.0 * std::f32::consts::PI * 64.0 * t / frame_len as f32).sin()
        })
        .collect()
}

/// Benchmark the GPU forward STFT FFT-accelerated path across three frame sizes.
///
/// "allocating" refers to per-call GPU buffer allocation (no persistent buffers
/// across iterations). Measures end-to-end cost: host→GPU upload, N·log₂N
/// butterfly stages, interleave, GPU→host readback.
fn bench_forward_fft(c: &mut Criterion) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device; skipping bench_forward_fft");
        return;
    };

    let mut group = c.benchmark_group("stft_forward_fft");

    for &(frame_len, hop_len, signal_len) in PARAMS {
        let plan = StftWgpuPlan::new(frame_len, hop_len);
        let signal = analytical_signal(signal_len, frame_len);

        group.bench_with_input(
            BenchmarkId::new("frame_len", frame_len),
            &frame_len,
            |b, _| {
                b.iter(|| {
                    backend
                        .execute_forward(black_box(&plan), black_box(&signal))
                        .expect("GPU forward STFT")
                });
            },
        );
    }

    group.finish();
}

/// Benchmark the GPU inverse STFT FFT-accelerated path across three frame sizes.
///
/// Spectrum is pre-computed once (outside the benchmark loop) so only the
/// inverse dispatch is measured: deinterleave → bitrev → butterfly×log₂N →
/// scale+window → OLA → readback.
fn bench_inverse_fft(c: &mut Criterion) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device; skipping bench_inverse_fft");
        return;
    };

    let mut group = c.benchmark_group("stft_inverse_fft");

    for &(frame_len, hop_len, signal_len) in PARAMS {
        let plan = StftWgpuPlan::new(frame_len, hop_len);
        let signal = analytical_signal(signal_len, frame_len);
        // Pre-compute spectrum once; benchmark measures only the inverse path.
        let spectrum = backend
            .execute_forward(&plan, &signal)
            .expect("forward pass for inverse benchmark setup");

        group.bench_with_input(
            BenchmarkId::new("frame_len", frame_len),
            &frame_len,
            |b, _| {
                b.iter(|| {
                    backend
                        .execute_inverse(
                            black_box(&plan),
                            black_box(&spectrum),
                            black_box(signal_len),
                        )
                        .expect("GPU inverse STFT")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_forward_fft, bench_inverse_fft);
criterion_main!(benches);
