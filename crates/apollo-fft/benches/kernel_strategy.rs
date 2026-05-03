//! Criterion benchmarks for Apollo FFT kernel strategies.

#![allow(missing_docs)]

use apollo_fft::application::execution::kernel::{
    bluestein, direct, fft_forward_64, mixed_radix, radix16, radix2, radix32, radix4, radix64,
    radix8,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64;

/// Generate a deterministic complex sinusoidal test signal of the given length.
fn signal(len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|index| {
            let x = index as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

/// Benchmark direct-DFT, radix strategies, mixed-radix, auto-selector and Bluestein.
fn bench_fft_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_kernel_strategy");

    for len in [16usize, 32, 64, 128, 256] {
        let input = signal(len);
        if len <= 128 {
            group.bench_with_input(
                BenchmarkId::new("direct_dft", len),
                &input,
                |bench, input| {
                    bench.iter(|| {
                        let output = direct::dft_forward_64(black_box(input));
                        black_box(output);
                    });
                },
            );
        }

        group.bench_with_input(
            BenchmarkId::new("radix2_inplace", len),
            &input,
            |bench, input| {
                bench.iter(|| {
                    let mut data = input.clone();
                    radix2::forward_inplace_64(black_box(&mut data));
                    black_box(data);
                });
            },
        );

        if len.is_power_of_two() && (len.trailing_zeros() % 2 == 0) {
            group.bench_with_input(
                BenchmarkId::new("radix4_inplace", len),
                &input,
                |bench, input| {
                    bench.iter(|| {
                        let mut data = input.clone();
                        radix4::forward_inplace_64(black_box(&mut data));
                        black_box(data);
                    });
                },
            );
        }
        if len.is_power_of_two() && (len.trailing_zeros() % 3 == 0) {
            group.bench_with_input(
                BenchmarkId::new("radix8_inplace", len),
                &input,
                |bench, input| {
                    bench.iter(|| {
                        let mut data = input.clone();
                        radix8::forward_inplace_64(black_box(&mut data));
                        black_box(data);
                    });
                },
            );
        }
        if len.is_power_of_two() && (len.trailing_zeros() % 4 == 0) {
            group.bench_with_input(
                BenchmarkId::new("radix16_inplace", len),
                &input,
                |bench, input| {
                    bench.iter(|| {
                        let mut data = input.clone();
                        radix16::forward_inplace_64(black_box(&mut data));
                        black_box(data);
                    });
                },
            );
        }
        if len.is_power_of_two() && (len.trailing_zeros() % 5 == 0) {
            group.bench_with_input(
                BenchmarkId::new("radix32_inplace", len),
                &input,
                |bench, input| {
                    bench.iter(|| {
                        let mut data = input.clone();
                        radix32::forward_inplace_64(black_box(&mut data));
                        black_box(data);
                    });
                },
            );
        }
        if len.is_power_of_two() && (len.trailing_zeros() % 6 == 0) {
            group.bench_with_input(
                BenchmarkId::new("radix64_inplace", len),
                &input,
                |bench, input| {
                    bench.iter(|| {
                        let mut data = input.clone();
                        radix64::forward_inplace_64(black_box(&mut data));
                        black_box(data);
                    });
                },
            );
        }

        group.bench_with_input(
            BenchmarkId::new("mixed_radix_inplace", len),
            &input,
            |bench, input| {
                bench.iter(|| {
                    let mut data = input.clone();
                    mixed_radix::forward_inplace_64(black_box(&mut data));
                    black_box(data);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("auto_selector", len),
            &input,
            |bench, input| {
                bench.iter(|| {
                    let mut data = input.clone();
                    fft_forward_64(black_box(&mut data));
                    black_box(data);
                });
            },
        );
    }

    for len in [31usize, 63, 127] {
        let input = signal(len);
        group.bench_with_input(
            BenchmarkId::new("bluestein_inplace", len),
            &input,
            |bench, input| {
                bench.iter(|| {
                    let mut data = input.clone();
                    bluestein::forward_inplace_64(black_box(&mut data));
                    black_box(data);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_fft_kernels);
criterion_main!(benches);
