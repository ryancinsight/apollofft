# Apollo Validation

Apollo validation is structured as named suites that can run independently or as a single report via
`cargo run -p apollo-validation`.

## Suites

- `run_fft_cpu_suite()`: CPU roundtrip, Parseval, and reference-threshold checks.
- `run_fft_gpu_suite()`: WGPU vs Apollo CPU parity, with explicit skip/failure notes when no usable adapter exists.
- `run_nufft_suite()`: exact-vs-fast 1D and 3D NUFFT comparisons.
- `run_external_comparison_suite()`: in-process `rustfft` parity plus explicit presence tracking for optional `external/rustfft` and `external/pyfftw` checkouts.
- `run_benchmark_suite()`: representative CPU FFT, GPU FFT, and NUFFT timing probes.
- `run_full_suite()`: top-level aggregation used by the CLI.

## Report Shape

The validation CLI emits JSON with these top-level sections:

- `fft_cpu`
- `fft_gpu`
- `nufft`
- `external`
- `benchmarks`
- `environment`

## Adversarial Coverage

- Degenerate dimensions.
- Buffer/device limit rejection in the WGPU crate.
- Shape mismatches.
- Non-contiguous Python buffers.
- Backend-unavailable paths with explicit notes instead of silent fallbacks.
- Cache reuse under contention.

## Optional External Assets

`apollo/external/rustfft` and `apollo/external/pyfftw` remain intentionally untracked. Apollo validation must still run without them; the report records their presence and includes skipped/comparison notes rather than failing by omission.
