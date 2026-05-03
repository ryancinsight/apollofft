"""
Performance benchmark + output validation: Apollo FFT vs numpy.fft

Section 1 (Output Validation): For every tested size and function, computes
Apollo and NumPy outputs and asserts max absolute error < ATOL. Fails fast
with a descriptive message on the first mismatch.

Section 2 (Timing): Measures median wall-clock time over N_TRIALS repetitions
for 1D, 2D, and 3D FFTs at various sizes, then prints a formatted comparison
table with speedup ratios.

Usage:
    python tests/benchmark_vs_numpy.py

Requirements: numpy, pyapollofft (built with `maturin develop --release`)
"""

import sys
import time

import numpy as np

import pyapollofft as afft

# ── Configuration ─────────────────────────────────────────────────────────────

SIZES_1D = [64, 256, 1024, 4096, 16384, 65536]
SIZES_2D = [32, 64, 128, 256, 512, 1024]
SIZES_3D = [8, 16, 32, 64, 128]

N_WARMUP = 5
N_TRIALS = 20

# Maximum absolute error tolerated vs NumPy (double-precision FFT).
# Worst-case accumulated error for N-point FFT ≈ O(sqrt(N) * 2.2e-16).
# For N=65536: ~5.6e-14. Threshold 1e-9 provides >4 decades of margin.
ATOL = 1e-9

# ── Section 1: Output validation ──────────────────────────────────────────────

print("=" * 64)
print("  OUTPUT VALIDATION: Apollo vs NumPy (atol={:.0e})".format(ATOL))
print("=" * 64)

_validation_failed = False
_val_rows = []

def _check(label, apollo_out, numpy_out):
    global _validation_failed
    diff = np.abs(np.asarray(apollo_out) - np.asarray(numpy_out))
    max_abs = float(np.max(diff))
    status = "PASS" if max_abs < ATOL else "FAIL"
    if status == "FAIL":
        _validation_failed = True
    _val_rows.append((label, max_abs, status))

# 1D real FFT
for n in SIZES_1D:
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.standard_normal(n))
    _check(f"fft1  N={n:>6}", afft.fft1(x), np.fft.fft(x))

# 1D complex FFT
for n in SIZES_1D:
    rng = np.random.default_rng(3)
    z = np.ascontiguousarray(
        rng.standard_normal(n) + 1j * rng.standard_normal(n), dtype=np.complex128
    )
    _check(f"cpx1  N={n:>6}", afft.fft_complex1(z), np.fft.fft(z))

# 2D real FFT
for n in SIZES_2D:
    rng = np.random.default_rng(1)
    x = np.ascontiguousarray(rng.standard_normal((n, n)))
    _check(f"fft2  N={n:>4}x{n:<4}", afft.fft2(x), np.fft.fft2(x))

# 3D real FFT
for n in SIZES_3D:
    rng = np.random.default_rng(2)
    x = np.ascontiguousarray(rng.standard_normal((n, n, n)))
    _check(f"fft3  N={n:>3}^3", afft.fft3(x), np.fft.fftn(x))

# Print validation table
print(f"  {'Transform':<18}  {'Max |err|':>12}  {'Status':>6}")
print("-" * 46)
for label, max_abs, status in _val_rows:
    print(f"  {label:<18}  {max_abs:>12.3e}  {status:>6}")
print()

if _validation_failed:
    print("ERROR: one or more output comparisons FAILED. Aborting benchmark.")
    sys.exit(1)

print("  All output comparisons PASSED.\n")


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
