use apollo_fft::application::execution::kernel::{bluestein, direct, radix2};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64;

fn signal(len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|index| {
            let x = index as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

fn bench_fft_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_kernel_strategy");

    for len in [32usize, 64, 128] {
        let input = signal(len);
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
