//! Criterion benchmarks for Apollo FFT kernel strategies.

#![allow(missing_docs)]

use apollo_fft::application::execution::kernel::{
    bluestein, direct, fft_forward_64, fft_forward_f16, mixed_radix, radix16, radix2, radix32,
    radix4, radix64, radix8, Cf16,
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

/// Deterministic f16-complex signal used by mixed-precision selector benchmarks.
fn signal_f16(len: usize) -> Vec<Cf16> {
    (0..len)
        .map(|index| {
            let x = index as f32;
            Cf16::from_f32_pair((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

fn max_abs_err_f16_vs_f64(input: &[Cf16], f16_kernel: fn(&mut [Cf16])) -> f64 {
    let mut f16_buf = input.to_vec();
    f16_kernel(&mut f16_buf);
    let mut f64_buf: Vec<Complex64> = input
        .iter()
        .map(|v| Complex64::new(v.re.to_f32() as f64, v.im.to_f32() as f64))
        .collect();
    fft_forward_64(&mut f64_buf);
    f16_buf
        .iter()
        .zip(f64_buf.iter())
        .map(|(a, b)| {
            let ar = a.re.to_f32() as f64;
            let ai = a.im.to_f32() as f64;
            ((ar - b.re).powi(2) + (ai - b.im).powi(2)).sqrt()
        })
        .fold(0.0f64, f64::max)
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

    for len in [64usize, 96] {
        let input = signal_f16(len);
        group.bench_with_input(
            BenchmarkId::new("mixed_precision_f16_auto", len),
            &input,
            |bench, input| {
                bench.iter(|| {
                    let mut data = input.clone();
                    fft_forward_f16(black_box(&mut data));
                    black_box(data);
                });
            },
        );
    }

    // f16 radix family benchmark coverage.
    for len in [64usize, 256, 1024] {
        let input = signal_f16(len);
        if len.is_power_of_two() && (len.trailing_zeros() % 2 == 0) {
            group.bench_with_input(
                BenchmarkId::new("radix4_inplace_f16", len),
                &input,
                |bench, input| {
                    bench.iter(|| {
                        let mut data = input.clone();
                        radix4::forward_inplace_f16(black_box(&mut data));
                        black_box(data);
                    });
                },
            );
        }
        if len.is_power_of_two() && (len.trailing_zeros() % 3 == 0) {
            group.bench_with_input(
                BenchmarkId::new("radix8_inplace_f16", len),
                &input,
                |bench, input| {
                    bench.iter(|| {
                        let mut data = input.clone();
                        radix8::forward_inplace_f16(black_box(&mut data));
                        black_box(data);
                    });
                },
            );
        }
        if len.is_power_of_two() && (len.trailing_zeros() % 4 == 0) {
            group.bench_with_input(
                BenchmarkId::new("radix16_inplace_f16", len),
                &input,
                |bench, input| {
                    bench.iter(|| {
                        let mut data = input.clone();
                        radix16::forward_inplace_f16(black_box(&mut data));
                        black_box(data);
                    });
                },
            );
        }
        if len.is_power_of_two() && (len.trailing_zeros() % 5 == 0) {
            group.bench_with_input(
                BenchmarkId::new("radix32_inplace_f16", len),
                &input,
                |bench, input| {
                    bench.iter(|| {
                        let mut data = input.clone();
                        radix32::forward_inplace_f16(black_box(&mut data));
                        black_box(data);
                    });
                },
            );
        }
        if len.is_power_of_two() && (len.trailing_zeros() % 6 == 0) {
            group.bench_with_input(
                BenchmarkId::new("radix64_inplace_f16", len),
                &input,
                |bench, input| {
                    bench.iter(|| {
                        let mut data = input.clone();
                        radix64::forward_inplace_f16(black_box(&mut data));
                        black_box(data);
                    });
                },
            );
        }
    }

    // Accuracy comparisons for new f16 radix entrypoints against f64 selector reference.
    for len in [64usize, 256, 1024] {
        let input = signal_f16(len);
        if len.is_power_of_two() && (len.trailing_zeros() % 3 == 0) {
            group.bench_with_input(
                BenchmarkId::new("accuracy_f16_radix8_vs_f64", len),
                &input,
                |bench, input| {
                    bench.iter(|| black_box(max_abs_err_f16_vs_f64(input, radix8::forward_inplace_f16)));
                },
            );
        }
        if len.is_power_of_two() && (len.trailing_zeros() % 4 == 0) {
            group.bench_with_input(
                BenchmarkId::new("accuracy_f16_radix16_vs_f64", len),
                &input,
                |bench, input| {
                    bench.iter(|| black_box(max_abs_err_f16_vs_f64(input, radix16::forward_inplace_f16)));
                },
            );
        }
        if len.is_power_of_two() && (len.trailing_zeros() % 5 == 0) {
            group.bench_with_input(
                BenchmarkId::new("accuracy_f16_radix32_vs_f64", len),
                &input,
                |bench, input| {
                    bench.iter(|| black_box(max_abs_err_f16_vs_f64(input, radix32::forward_inplace_f16)));
                },
            );
        }
        if len.is_power_of_two() && (len.trailing_zeros() % 6 == 0) {
            group.bench_with_input(
                BenchmarkId::new("accuracy_f16_radix64_vs_f64", len),
                &input,
                |bench, input| {
                    bench.iter(|| black_box(max_abs_err_f16_vs_f64(input, radix64::forward_inplace_f16)));
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_fft_kernels);
criterion_main!(benches);
