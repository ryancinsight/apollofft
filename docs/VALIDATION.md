# Apollo Validation

Apollo validation is structured as named suites that can run independently or as a single report via
`cargo run -p apollo-validation`. For performance comparisons, use
`cargo run -p apollo-validation --release`; debug builds are useful for correctness but not for timing claims.

## Suites

- `run_fft_cpu_suite()`: CPU roundtrip, Parseval, and reference-threshold checks.
- `run_fft_cpu_suite()`: CPU roundtrip, Parseval, repeated-run stability, and non-finite robustness checks.
- `run_fft_gpu_suite()`: WGPU vs Apollo CPU parity, with explicit skip/failure notes when no usable adapter exists.
- `run_nufft_suite()`: exact-vs-fast 1D and 3D NUFFT comparisons, including irrational-position and clustered near-boundary stress cases.
- `run_external_comparison_suite()`: direct Apollo parity against `rustfft`, NumPy FFT, and optional `pyfftw`, plus repeated-run stability checks.
- `run_benchmark_suite()`: representative Apollo CPU FFT, `rustfft`, NumPy, GPU FFT, and NUFFT timing probes.
- `run_full_suite()`: top-level aggregation used by the CLI.

## Report Shape

The validation CLI emits JSON with these top-level sections:

- `fft_cpu`
- `fft_gpu`
- `nufft`
- `external`
- `benchmarks`
- `environment`

The `external` section now contains per-backend subreports (`rustfft`, `numpy`, and `pyfftw`) with
1D, prime-sized 1D, and 3D max-absolute-error measurements together with repeated-run stability
metrics.

## Adversarial Coverage

- Degenerate dimensions.
- Buffer/device limit rejection in the WGPU crate.
- Shape mismatches.
- Non-contiguous Python buffers.
- Repeated-run stability against `rustfft` and NumPy.
- Non-finite signal propagation checks in Apollo CPU FFT.
- Prime-sized and mixed-shape external parity checks.
- Clustered and wrapped NUFFT point distributions.
- Backend-unavailable paths with explicit notes instead of silent fallbacks.
- Cache reuse under contention.

## Optional External Assets

`apollo/external/rustfft` and `apollo/external/pyfftw` remain intentionally untracked. Apollo validation must still run without them; the report records their presence and includes skipped/comparison notes rather than failing by omission. Direct NumPy comparison is attempted through the host Python interpreter, and optional `pyfftw` comparison is enabled only when the module is importable.

## Current Audit Direction

Recent Apollo optimization work follows the same themes emphasized by current high-performance FFT and NUFFT literature:

- Keep NUFFT spreading cache-aware by reordering points and improving locality.
- Reduce transient allocation in hot loops so bandwidth is spent on useful arithmetic.
- Treat GPU FFT throughput as a kernel-orchestration problem: host-device traffic and per-stage launches dominate small and medium transforms, so future Apollo WGPU work should keep axis data resident on device and fuse more stages.

Primary references:

- FINUFFT documentation highlights point bin-sorting, SIMD spreading, and lower RAM use as major CPU NUFFT performance drivers.
- Barnett et al., “A parallel non-uniform fast Fourier transform library based on an exponential of semicircle kernel” motivates cache-aware spreading and memory-efficient kernels.
- Shih et al., “cuFINUFFT” emphasizes cache-aware point reordering and blocked spreading in shared memory for GPU NUFFT.
- Wu et al., “TurboFFT” shows that kernel fusion is central to competitive GPU FFT performance.
