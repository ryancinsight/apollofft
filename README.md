# Apollo Workspace

Apollo is a standalone workspace for reusable Fourier transform planning, execution, validation, backend adapters, and Python bindings.

Stage 2 moves Apollo beyond the initial compatibility cut:

- `apollo-fft` owns reusable dense CPU FFT plans, cache orchestration, shared contracts, and backend abstractions.
- `apollo-dctdst` owns DCT/DST real-to-real transform plan metadata,
  verified direct kernels, inverse scaling, and caller-owned output execution.
- `apollo-dht` owns real-to-real Discrete Hartley Transform plans, coefficient storage, and self-inverse kernels.
- `apollo-hilbert` owns Hilbert transform plans, analytic-signal storage, envelope extraction, and phase extraction.
- `apollo-nufft` owns non-uniform FFT plans and direct-reference validation surfaces.
- `apollo-radon` owns parallel-beam Radon projections, adjoint backprojection, and filtered backprojection.
- `apollo-sht` owns spherical harmonic transform grid metadata, coefficient storage, Gauss-Legendre quadrature, and value-computing SHT plans.
- `apollo-sdft` owns sliding DFT streaming-window metadata, O(bin_count) update recurrence kernels, and streaming state.
- `apollo-mellin` owns Mellin scale-domain metadata and validation.
- `apollo-sft` owns the sparse Fourier transform single source of truth.
- `apollo-fft-wgpu` owns the real shader-backed 3D WGPU FFT path with radix-2 and Bluestein/Chirp-Z axis strategies.
- `apollo-nufft-wgpu` owns NUFFT GPU execution and now ships exact direct Type-1 and Type-2 WGPU kernels for 1D and 3D; fast gridding paths remain unsupported until GPU spreading/interpolation, oversampled FFT dispatch, and deconvolution land there.
- Per-transform WGPU crates own GPU backend boundaries for their respective mathematical domains. `apollo-fwht-wgpu` and `apollo-dht-wgpu` ship real 1D `f32` kernels, `apollo-dctdst-wgpu` ships the full real 1D `f32` DCT-II/DCT-III/DST-II/DST-III family, `apollo-czt-wgpu` ships a direct complex forward CZT kernel, `apollo-gft-wgpu` ships forward and inverse graph-basis execution, `apollo-hilbert-wgpu` ships forward analytic/quadrature Hilbert execution, `apollo-mellin-wgpu` ships a forward Mellin log-frequency spectrum kernel, `apollo-ntt-wgpu` ships forward and inverse NTT execution on its supported modulus surface, `apollo-qft-wgpu` ships forward and inverse dense unitary QFT execution, `apollo-radon-wgpu` ships forward parallel-beam projection execution, `apollo-sdft-wgpu` ships forward direct-bin sliding DFT execution, `apollo-sft-wgpu` ships direct dense DFT sparse top-k execution and inverse reconstruction, `apollo-sht-wgpu` ships direct complex SHT forward/inverse execution using owner-derived basis and quadrature buffers, `apollo-stft-wgpu` ships forward Hann-windowed STFT execution, and `apollo-wavelet-wgpu` ships forward and inverse Haar DWT execution.
- `apollo-wavelet` owns discrete and continuous wavelet transform plans for multiresolution analysis.
- `apollo-validation` emits structured CPU, GPU, NUFFT, benchmark, and external-comparison reports.
- `apollo-python` exposes FFT, NUFFT, precision selection, and backend capability introspection for Python callers.

Mixed precision is now a first-class Apollo concept:

- CPU defaults to `high_accuracy` (`f64` storage and `f64` compute).
- CPU also supports opt-in `low_precision` (`f32` storage and `f32` compute).
- CPU also supports opt-in `mixed_precision` (`half::f16` storage with `f32` compute in the current FFT path).
- WGPU currently exposes only the truthful `low_precision` profile because the shipped shaders execute in `f32`; Apollo does not advertise mixed precision there until a real mixed arithmetic path exists.

## Crates

- `apollo-fft`: CPU FFT plans, cache management, shared types, and backend abstractions.
- `apollo-czt`: chirp z-transform plans, direct reference execution,
  Bluestein convolution execution, and caller-owned output paths.
- `apollo-dctdst`: DCT/DST real-to-real transform plans, verified direct
  kernels, inverse scaling, and caller-owned output execution.
- `apollo-dht`: real-to-real Discrete Hartley Transform plans with forward/inverse kernel reuse.
- `apollo-frft`: fractional Fourier transform reference plans with finite
  integer-rotation state and caller-owned output execution.
- `apollo-fwht`: fast Walsh-Hadamard transform plans, in-place kernels, and
  caller-owned output execution.
- `apollo-gft`: graph Fourier transform domain validation, Laplacian spectral basis construction, reusable plans, and verification.
- `apollo-hilbert`: Hilbert transform plans with analytic-signal, envelope, and phase extraction.
- `apollo-mellin`: Mellin scale-domain moments, log-resampling, and log-frequency spectra.
- `apollo-ntt`: radix-2 number theoretic transform plans, modular residue
  normalization, in-place kernels, and caller-owned output execution.
- `apollo-nufft`: non-uniform FFT plans and exact direct references.
- `apollo-qft`: quantum state-dimension validation, dense unitary QFT kernels, reusable plans, and verification.
- `apollo-radon`: parallel-beam Radon transform plans with sinogram storage and filtered backprojection.
- `apollo-sdft`: sliding DFT streaming plans with direct initialization and recurrence updates.
- `apollo-sft`: sparse Fourier transform domain model, plan execution, direct recovery kernel, and verification.
- `apollo-sht`: spherical harmonic transform plans with Gauss-Legendre latitude quadrature and complex coefficient storage.
- `apollo-stft`: short-time Fourier transform plans with centered-frame
  reconstruction, overlap-add normalization, and caller-owned output buffers.
- `apollo-fft-wgpu`: Real dense FFT WGPU backend and GPU parity surface.
- `apollo-nufft-wgpu`: exact direct Type-1/Type-2 NUFFT WGPU execution for 1D and 3D plus capability contract surface.
- `apollo-czt-wgpu`, `apollo-dctdst-wgpu`, `apollo-dht-wgpu`,
  `apollo-frft-wgpu`, `apollo-gft-wgpu`,
  `apollo-hilbert-wgpu`, `apollo-mellin-wgpu`, `apollo-ntt-wgpu`,
  `apollo-qft-wgpu`, `apollo-radon-wgpu`, `apollo-sdft-wgpu`,
  `apollo-sft-wgpu`, `apollo-sht-wgpu`, `apollo-stft-wgpu`, and
  `apollo-wavelet-wgpu`: per-transform WGPU backend boundaries with
  capability and plan descriptor contracts.
- `apollo-czt-wgpu`: direct complex forward CZT WGPU execution with CPU parity tests; inverse remains unsupported.
- `apollo-dctdst-wgpu`: 1D real `f32` DCT-II/DCT-III/DST-II/DST-III WGPU execution with CPU parity tests.
- `apollo-dht-wgpu`: 1D real `f32` DHT WGPU execution with CPU parity tests.
- `apollo-fwht-wgpu`: 1D real `f32` FWHT WGPU execution with CPU parity tests.
- `apollo-gft-wgpu`: forward and inverse real graph Fourier basis WGPU execution with CPU parity tests on the implemented `f32` surface.
- `apollo-hilbert-wgpu`: forward analytic-signal and quadrature Hilbert WGPU execution with CPU parity tests; inverse remains unsupported.
- `apollo-mellin-wgpu`: forward Mellin log-frequency spectrum WGPU execution with CPU parity tests; inverse remains unsupported.
- `apollo-ntt-wgpu`: forward and inverse modular NTT WGPU execution with CPU parity tests on the supported 32-bit modulus surface.
- `apollo-qft-wgpu`: forward and inverse dense unitary QFT WGPU execution with CPU parity tests on the implemented `f32` complex surface.
- `apollo-radon-wgpu`: forward parallel-beam Radon projection WGPU execution with CPU parity tests; inverse and reconstruction remain unsupported.
- `apollo-sdft-wgpu`: forward direct-bin sliding DFT WGPU execution with CPU parity tests; inverse remains unsupported.
- `apollo-sft-wgpu`: forward dense direct DFT with deterministic top-k sparse projection and inverse sparse reconstruction with CPU parity tests.
- `apollo-sht-wgpu`: forward and inverse complex spherical harmonic transform WGPU execution with CPU parity tests on owner-derived quadrature and basis buffers.
- `apollo-stft-wgpu`: forward Hann-windowed STFT WGPU execution with CPU parity tests; inverse remains unsupported.
- `apollo-wavelet-wgpu`: forward and inverse Haar DWT WGPU execution with CPU parity tests on the implemented `f32` surface.
- `apollo-wavelet`: DWT/CWT multiresolution transforms with Haar, Daubechies-4, Ricker, and DC-corrected real Morlet support.
- `apollo-cudatile`: trait-compatible cudatile adapter surface.
- `apollo-validation`: parity, adversarial, benchmark, and external-reference runners.
- `apollo-python`: PyO3 bindings, NumPy interop, and backend introspection.

## Architecture

Apollo uses a vertical hierarchy per crate:

```text
src/
  domain/            contracts, metadata, validated configuration
  application/       plans, orchestration, transform execution
  infrastructure/    concrete kernels, backend transport, external probes
  verification/      value-semantic tests where crate-local test modules are not enough
```

Dependency direction is one-way:

```text
lib.rs -> application -> domain
lib.rs -> infrastructure -> application -> domain
```

`domain` never depends on `application` or `infrastructure`. Backend-specific code stays behind infrastructure or adapter crates. Public APIs are exposed through crate roots and narrow compatibility modules.

### Transform Ownership

- Dense FFT: `apollo-fft`. The 1D, 2D, and 3D CPU plans use Apollo-owned
  radix-2 and Bluestein FFT kernels with FFTW-compatible public inverse
  normalization. The direct DFT kernel remains a reference surface.
- NUFFT: `apollo-nufft`. NUFFT domain descriptors, Kaiser-Bessel kernels,
  exact direct references, and fast gridding plans live outside `apollo-fft`.
- Sparse FFT: `apollo-sft`.
- Real-to-real DCT/DST: `apollo-dctdst`.
- Discrete Hartley Transform: `apollo-dht`.
- Hilbert Transform: `apollo-hilbert`.
- Radon Transform: `apollo-radon`.
- Spherical harmonic transform: `apollo-sht`.
- Sliding DFT: `apollo-sdft`.
- Mellin transform: `apollo-mellin`.
- Wavelet transforms: `apollo-wavelet`.
- GPU FFT: `apollo-fft-wgpu`. Radix-2 GPU execution stages bit reversal,
  butterfly stages, and inverse scaling as separate compute passes over
  linearized workgroup domains.
- GPU NUFFT: `apollo-nufft-wgpu`. NUFFT WGPU support is a separate transform
  domain boundary and does not live in the dense FFT backend crate.
- Other GPU transform domains: each transform has an owning `*-wgpu` crate.
  `apollo-czt-wgpu` now executes the direct complex forward CZT on WGPU for
  the implemented `f32` kernel surface. Inverse remains unsupported there
  until the owning CZT crate defines and verifies an authoritative inverse or
  adjoint contract.
  `apollo-hilbert-wgpu` now executes the analytic-signal Hilbert path on WGPU
  by direct DFT, analytic-spectrum masking, and inverse DFT over the
  implemented `f32` surface. Inverse remains unsupported there until the
  owning Hilbert crate defines and verifies an authoritative inverse or
  adjoint contract.
  `apollo-mellin-wgpu` now executes the forward Mellin log-frequency spectrum
  on WGPU by log-resampling onto the plan scale grid and applying the direct
  spectrum sum over the implemented `f32` surface. Inverse remains unsupported
  there until the owning Mellin crate defines and verifies an authoritative
  inverse or adjoint contract.
  `apollo-ntt-wgpu` now executes the forward and inverse NTT on WGPU by direct
  modular summation over the supported 32-bit modulus surface and validates
  parity against the owning CPU crate.
  `apollo-gft-wgpu` now executes the forward and inverse graph Fourier
  transform on WGPU by direct multiplication against the orthonormal spectral
  basis carried by the plan and validates parity against the owning CPU crate.
  `apollo-qft-wgpu` now executes the forward and inverse dense unitary QFT on
  WGPU by direct complex summation with `1 / sqrt(n)` normalization and
  validates parity against the owning CPU crate.
  `apollo-radon-wgpu` now executes the forward parallel-beam Radon projection
  on WGPU by direct image-space accumulation into sinogram detector bins and
  validates parity against the owning CPU crate. Inverse and filtered
  backprojection remain unsupported there until the owning Radon crate defines
  and verifies an authoritative GPU reconstruction contract.
  `apollo-sdft-wgpu` now executes forward direct-bin sliding DFT evaluation on
  WGPU by direct `O(N*K)` summation over the current real-valued window and
  validates parity against `apollo-sdft::SdftPlan::direct_bins`. Inverse
  remains unsupported there because the crate exposes streaming bin evaluation
  rather than a reconstruction contract.
  `apollo-sht-wgpu` now executes forward and inverse complex SHT on WGPU by
  direct matrix summation over owner-derived Gauss-Legendre quadrature weights
  and spherical harmonic basis values, preserving `apollo-sht` as the SSOT for
  basis and quadrature definitions.
  `apollo-stft-wgpu` now executes forward Hann-windowed STFT on WGPU by direct
  per-frame DFT over centered frames with zero-padding semantics and validates
  parity against `apollo-stft::StftPlan::forward`. Inverse WOLA reconstruction
  remains unsupported on the GPU surface.
  `apollo-wavelet-wgpu` now executes forward and inverse multi-level Haar DWT
  on WGPU using a Mallat decomposition and validates forward coefficients and
  inverse reconstruction against `apollo-wavelet::DwtPlan` for the Haar
  surface. Daubechies4 and CWT remain CPU-only until their GPU contracts are
  implemented and verified.
  `apollo-dctdst-wgpu` now executes real 1D DCT-II, DCT-III, DST-II, and
  DST-III on WGPU and computes their inverses by the paired transform with
  `2 / n` normalization.
  `apollo-dht-wgpu` now executes the 1D real DHT on WGPU with direct Hartley
  summation and inverse normalization by `1 / n`.
  `apollo-fwht-wgpu` now executes the 1D real FWHT on WGPU with the same
  butterfly network as the CPU plan and inverse normalization by `1 / n`.
- Python bindings: `apollo-python`.
- Validation and external parity: `apollo-validation`.

SFT is consolidated into `apollo-sft`; `apollo-fft` does not contain an SFT implementation or SFT export path. This preserves SSOT and prevents duplicated sparse transform logic.

## Precision Profiles

Apollo exposes these precision descriptors through Rust and Python:

- `high_accuracy`
- `low_precision`
- `mixed_precision`

In Rust, they are represented by `PrecisionMode`, `StoragePrecision`, `ComputePrecision`, and
`PrecisionProfile`. Existing APIs keep their current default behavior; lower-precision paths are
opt-in via `with_precision(...)` constructors or the generic `*_typed(...)` helpers that dispatch on
`RealFftData`.

Apollo now also exposes explicit `*_f16` helpers for real-domain FFT storage. The maintainable
Rust surface is the generic typed API.

## Design Rules

- `apollo/` is intentionally **not** a member of the root `d:\\kwavers\\Cargo.toml` workspace.
- Shared transform invariants live in the owning crate and are re-exported instead of duplicated.
- `kwavers` consumes Apollo FFT/NUFFT through compatibility re-exports instead of owning reusable transform implementations.
- Solver-specific spectral helpers remain in `kwavers` until they prove broadly reusable.
- Bounded variation belongs in traits, newtypes, configuration structs, strategy types, or backend abstractions, not cloned public APIs.
- Validation assertions inspect computed values and compare them against analytical invariants or independent references.

## References

- [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- [`docs/THEORY.md`](./docs/THEORY.md)
- [`docs/VALIDATION.md`](./docs/VALIDATION.md)
- [`docs/MIGRATION_KWAVERS.md`](./docs/MIGRATION_KWAVERS.md)
