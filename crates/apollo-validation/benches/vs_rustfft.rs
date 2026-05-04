//! Criterion benchmarks comparing Apollo FFT against RustFFT.
//!
//! Gated on the `external-references` feature of `apollo-validation`, which
//! makes `rustfft` an explicit optional dependency.  Run with:
//!
//! ```text
//! cargo bench -p apollo-validation --features external-references --bench vs_rustfft
//! ```
//!
//! Both implementations operate on `Complex<f64>` (f64) and `Complex<f32>` (f32)
//! in-place buffers.  RustFFT planning is performed once outside the measured
//! loop — matching production usage where a plan is reused across many transforms.
//! Apollo's `fft_forward_64`/`fft_forward_32` auto-selectors are measured
//! including their O(1) dispatch overhead (no dynamic allocation inside the loop).
//!
//! Sizes benchmarked:
//! - Power-of-two (radix path): 64, 256, 1024, 4096, 16384, 65536, 262144
//! - Arbitrary/non-power-of-two (Bluestein path): 100, 1000, 10000

#![cfg(feature = "external-references")]
#![allow(missing_docs)]

use apollo_fft::application::execution::kernel::{fft_forward_32, fft_forward_64};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::{Complex32, Complex64};
use rustfft::FftPlanner;

// ── helpers ───────────────────────────────────────────────────────────────────

fn signal_f64(len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|i| {
            let x = i as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

fn signal_f32(len: usize) -> Vec<Complex32> {
    (0..len)
        .map(|i| {
            let x = i as f32;
            Complex32::new((0.017_f32 * x).sin(), 0.25_f32 * (0.031_f32 * x).cos())
        })
        .collect()
}

// ── f64 benchmark group ───────────────────────────────────────────────────────

fn bench_f64(c: &mut Criterion) {
    const POT: &[usize] = &[64, 256, 1024, 4096, 16_384, 65_536, 262_144];
    const ARB: &[usize] = &[100, 1_000, 10_000];

    let mut group = c.benchmark_group("apollo_vs_rustfft_f64");
    group.sample_size(50);

    for &len in POT.iter().chain(ARB.iter()) {
        let input = signal_f64(len);

        // Apollo — auto-selector (radix path for PoT, Bluestein for arbitrary).
        group.bench_with_input(
            BenchmarkId::new("apollo", len),
            &input,
            |bench, input| {
                bench.iter(|| {
                    let mut buf = input.clone();
                    fft_forward_64(black_box(&mut buf));
                    black_box(buf);
                });
            },
        );

        // RustFFT — plan once, process many.  Scratch buffer reused per iter.
        {
            let mut planner: FftPlanner<f64> = FftPlanner::new();
            let fft = planner.plan_fft_forward(len);
            let mut scratch = vec![
                rustfft::num_complex::Complex::new(0.0_f64, 0.0_f64);
                fft.get_inplace_scratch_len()
            ];
            // rustfft::num_complex::Complex<f64> has the same ABI as num_complex::Complex64.
            let input_rft: Vec<rustfft::num_complex::Complex<f64>> = input
                .iter()
                .map(|c| rustfft::num_complex::Complex::new(c.re, c.im))
                .collect();

            group.bench_with_input(
                BenchmarkId::new("rustfft", len),
                &input_rft,
                |bench, input| {
                    bench.iter(|| {
                        let mut buf = input.clone();
                        fft.process_with_scratch(black_box(&mut buf), &mut scratch);
                        black_box(&buf);
                    });
                },
            );
        }
    }

    group.finish();
}

// ── f32 benchmark group ───────────────────────────────────────────────────────

fn bench_f32(c: &mut Criterion) {
    const POT: &[usize] = &[64, 256, 1024, 4096, 16_384, 65_536, 262_144];
    const ARB: &[usize] = &[100, 1_000, 10_000];

    let mut group = c.benchmark_group("apollo_vs_rustfft_f32");
    group.sample_size(50);

    for &len in POT.iter().chain(ARB.iter()) {
        let input = signal_f32(len);

        // Apollo f32 auto-selector.
        group.bench_with_input(
            BenchmarkId::new("apollo", len),
            &input,
            |bench, input| {
                bench.iter(|| {
                    let mut buf = input.clone();
                    fft_forward_32(black_box(&mut buf));
                    black_box(buf);
                });
            },
        );

        // RustFFT f32.
        {
            let mut planner: FftPlanner<f32> = FftPlanner::new();
            let fft = planner.plan_fft_forward(len);
            let mut scratch = vec![
                rustfft::num_complex::Complex::new(0.0_f32, 0.0_f32);
                fft.get_inplace_scratch_len()
            ];
            let input_rft: Vec<rustfft::num_complex::Complex<f32>> = input
                .iter()
                .map(|c| rustfft::num_complex::Complex::new(c.re, c.im))
                .collect();

            group.bench_with_input(
                BenchmarkId::new("rustfft", len),
                &input_rft,
                |bench, input| {
                    bench.iter(|| {
                        let mut buf = input.clone();
                        fft.process_with_scratch(black_box(&mut buf), &mut scratch);
                        black_box(&buf);
                    });
                },
            );
        }
    }

    group.finish();
}

// ── criterion entry points ────────────────────────────────────────────────────

criterion_group!(benches, bench_f64, bench_f32);
criterion_main!(benches);
