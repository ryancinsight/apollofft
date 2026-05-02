"""
Performance benchmark: Apollo FFT vs numpy.fft

Measures median wall-clock time over N_TRIALS repetitions for 1D, 2D, and 3D FFTs
at various sizes, then prints a formatted comparison table with speedup ratios.

Usage:
    python tests/benchmark_vs_numpy.py

Requirements: numpy, pyapollofft (built with `maturin develop --release`)
"""

import time

import numpy as np

import pyapollofft as afft

# ── Configuration ─────────────────────────────────────────────────────────────

SIZES_1D = [64, 256, 1024, 4096, 16384, 65536]
SIZES_2D = [32, 64, 128, 256, 512, 1024]
SIZES_3D = [8, 16, 32, 64, 128]

N_WARMUP = 5
N_TRIALS = 20


# ── Timing helper ─────────────────────────────────────────────────────────────

def median_time_ms(fn, *args) -> float:
    """Return the median execution time in milliseconds over N_TRIALS calls."""
    for _ in range(N_WARMUP):
        fn(*args)
    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return sorted(times)[N_TRIALS // 2] * 1_000.0


# ── 1D benchmarks ─────────────────────────────────────────────────────────────

print("=" * 64)
print("  1D FFT: Apollo (real→complex) vs numpy.fft.fft (float64)")
print(f"  warmup={N_WARMUP}, trials={N_TRIALS}, median latency")
print("=" * 64)
print(f"{'N':>8}  {'Apollo (ms)':>12}  {'numpy (ms)':>12}  {'Speedup':>9}")
print("-" * 64)

for n in SIZES_1D:
    x = np.random.default_rng(0).standard_normal(n).astype(np.float64)
    x_c = np.ascontiguousarray(x)
    t_apollo = median_time_ms(afft.fft1, x_c)
    t_numpy = median_time_ms(np.fft.fft, x_c)
    ratio = t_numpy / t_apollo if t_apollo > 0 else float("inf")
    print(f"{n:>8}  {t_apollo:>12.4f}  {t_numpy:>12.4f}  {ratio:>8.2f}x")

print()

# ── 2D benchmarks ─────────────────────────────────────────────────────────────

print("=" * 64)
print("  2D FFT: Apollo (real→complex) vs numpy.fft.fft2 (float64)")
print(f"  warmup={N_WARMUP}, trials={N_TRIALS}, median latency")
print("=" * 64)
print(f"{'N×N':>8}  {'Apollo (ms)':>12}  {'numpy (ms)':>12}  {'Speedup':>9}")
print("-" * 64)

for n in SIZES_2D:
    x = np.random.default_rng(1).standard_normal((n, n)).astype(np.float64)
    x_c = np.ascontiguousarray(x)
    t_apollo = median_time_ms(afft.fft2, x_c)
    t_numpy = median_time_ms(np.fft.fft2, x_c)
    ratio = t_numpy / t_apollo if t_apollo > 0 else float("inf")
    print(f"{f'{n}x{n}':>8}  {t_apollo:>12.4f}  {t_numpy:>12.4f}  {ratio:>8.2f}x")

print()

# ── 3D benchmarks ─────────────────────────────────────────────────────────────

print("=" * 64)
print("  3D FFT: Apollo (real→complex) vs numpy.fft.fftn (float64)")
print(f"  warmup={N_WARMUP}, trials={N_TRIALS}, median latency")
print("=" * 64)
print(f"{'N³':>8}  {'Apollo (ms)':>12}  {'numpy (ms)':>12}  {'Speedup':>9}")
print("-" * 64)

for n in SIZES_3D:
    x = np.random.default_rng(2).standard_normal((n, n, n)).astype(np.float64)
    x_c = np.ascontiguousarray(x)
    t_apollo = median_time_ms(afft.fft3, x_c)
    t_numpy = median_time_ms(np.fft.fftn, x_c)
    ratio = t_numpy / t_apollo if t_apollo > 0 else float("inf")
    print(f"{f'{n}³':>8}  {t_apollo:>12.4f}  {t_numpy:>12.4f}  {ratio:>8.2f}x")

print()

# ── Complex FFT (fft_complex1 vs numpy) ───────────────────────────────────────

print("=" * 64)
print("  1D FFT (complex128): Apollo fft_complex1 vs numpy.fft.fft")
print(f"  warmup={N_WARMUP}, trials={N_TRIALS}, median latency")
print("=" * 64)
print(f"{'N':>8}  {'Apollo (ms)':>12}  {'numpy (ms)':>12}  {'Speedup':>9}")
print("-" * 64)

for n in SIZES_1D:
    rng = np.random.default_rng(3)
    z = np.ascontiguousarray(
        rng.standard_normal(n) + 1j * rng.standard_normal(n), dtype=np.complex128
    )
    t_apollo = median_time_ms(afft.fft_complex1, z)
    t_numpy = median_time_ms(np.fft.fft, z)
    ratio = t_numpy / t_apollo if t_apollo > 0 else float("inf")
    print(f"{n:>8}  {t_apollo:>12.4f}  {t_numpy:>12.4f}  {ratio:>8.2f}x")

print()
print("Benchmark complete.")
