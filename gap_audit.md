# Apollo Gap Audit

## Closed Gaps

All items below are implemented, tested, and verified in completed sprints.

### Workspace and Infrastructure

- Registered every `crates/apollo-*` crate in the root workspace.
- Replaced incomplete `apollo-validation` orchestration with computed CPU, GPU-surface, NUFFT, external-reference, benchmark, and environment reports.
- Added real crate roots for `apollo-frft`, `apollo-gft`, and `apollo-stft`.
- Split `apollo-validation` external references behind an optional validation-only feature so `rustfft` is validation-only; audited that `realfft` is absent from the workspace dependency graph.
- Completed `apollo-validation` with the new multi-crate API surface and conditional external-backend wiring.
- Aligned `apollo-python` with current crate names, shape newtypes, and full-spectrum FFT plan APIs.
- Added crate-local architecture README files for all `crates/apollo-*` crates.

### Core Algorithm Correctness

- Corrected CZT Bluestein convolution lag construction against the direct CZT definition.
- Corrected SFT expected coefficients against the analytical DFT of the test signal.
- Corrected STFT boundary coverage by using centered analysis frames with overlap-add normalization.
- Fixed `FftPlan1D` and `FftPlan2D` missing `forward_complex`/`inverse_complex` allocating wrappers (parity with `FftPlan3D`).

### FFT O(N log N) Kernel Strategy

- Replaced O(N^2) direct DFT kernels with O(N log N) strategy: iterative Cooley-Tukey radix-2 for power-of-2 sizes and Bluestein chirp-Z for arbitrary sizes; `rustfft` removed from production `apollo-fft` dependency.
- Implemented `kernel::radix2` (iterative Cooley-Tukey DIT, power-of-2) with value-semantic tests.
- Implemented `kernel::bluestein` (chirp-Z, arbitrary N, verified for N=3,5,6,7,11) with value-semantic tests.
- Added `fft_forward_64`, `fft_inverse_64`, `fft_inverse_unnorm_64`, `fft_forward_32`, `fft_inverse_32`, `fft_inverse_unnorm_32` auto-selecting wrappers to `kernel::mod`.
- Updated `FftPlan1D`, `FftPlan2D`, `FftPlan3D` axis-pass methods to use new O(N log N) kernel.
- Corrected stale FFT architecture docs from direct-kernel execution to radix-2/Bluestein auto-selection.

### Memory and Performance Optimizations

- Eliminated per-stage `Vec<Complex>` twiddle allocations in radix-2 (f32/f64 forward/inverse) by replacing with a single N/2-entry stride-indexed table (Unified Twiddle Table theorem proved in module doc).
- Cached Bluestein scratch buffer in `FftPlan1D` via `Mutex<Vec<Complex64>>` to eliminate per-call heap allocation on the non-power-of-two hot path.
- Precomputed DWT highpass QMF coefficients once per `analysis_stage_into`/`synthesis_stage_into` call; QMF identity g[k] = (-1)^k h[L-1-k] proved from Smith-Barnwell PR condition.
- Removed duplicate transformed-lane collections from FFT 2D/3D axis passes.
- Reduced NUFFT interpolation and 3D separable-pass allocation by borrowing type-2 grids and reusing per-axis lane buffers.
- Reduced Radon filtered-backprojection allocation by adding caller-owned ramp filtering.
- Implemented `FftPlan1D` zero-allocation `forward_complex_slice_inplace` and `inverse_complex_slice_inplace` methods to execute dense kernels directly from caller slices.
- Eliminated O(M) nested `Array1` heap allocations in STFT `forward_with_window_inner` and `inverse_into` by using `FftPlan1D` slice execution and flattened arrays.
- Eliminated dynamic `Array1::from_shape_vec` conversions in NUFFT 1D Type-1 and Type-2 evaluation kernels utilizing `FftPlan1D` slice execution.

### New Transform Crates

- Added `apollo-hilbert` with Hilbert transform plans, analytic-signal storage, envelope/phase extraction, and analytical/property tests.
- Added `apollo-radon` with parallel-beam forward projections, adjoint backprojection, ramp-filtered backprojection, sinogram storage, and analytical/property tests.
- Completed `apollo-mellin` with Mellin moments, log-frequency spectra, execution contracts, and analytical tests.

### Theorem Documentation and Proofs

- Added Parseval/Plancherel energy-invariance theorem with proof to `radix2.rs` module doc; added Unified Twiddle Table theorem proving stride-index equivalence.
- Added I_0 convergence theorem (geometric tail bound, K=256 sufficiency corollary) to `kaiser_bessel.rs`.
- Replaced stale skeleton documentation in completed transform crates and added DCT/DST value-semantic tests.
- Removed incorrect unverified DCT/DST fast branch and added large-plan parity tests against analytical kernels.
- Added CZT README, Bluestein theorem docs, caller-owned forward path, and in-place convolution workspace multiplication.
- Added FWHT README, Hadamard involution theorem docs, caller-owned real/complex output paths, and parity tests.
- Added NTT README, root-of-unity theorem docs, true in-place execution, caller-owned output paths, residue normalization, and overflow-safe modular addition.
- Added FrFT README, FrFT rotation theorem docs, finite singular integer-order plan state, inverse APIs, and inverse parity tests.
- Added STFT README, overlap-add theorem docs, cleaned module comments, actionable buffer diagnostics, and inverse caller-owned parity tests.
- Added DCT/DST README, inverse-pair theorem docs, caller-owned inverse output, and inverse parity tests.

### Bug Fixes and Repairs

- Consolidated SFT ownership into `apollo-sft` and split it into domain, application, infrastructure, and verification modules.
- Cleaned `apollo-sft` Rustdoc encoding, removed deprecated ndarray raw-vector extraction, and reused the crate-local direct DFT reference in verification.
- Repaired SHT source encoding so Rust tooling parses theorem/reference docs.
- Repaired SDFT result propagation and QFT property-test plan construction.
- Removed duplicated NUFFT 3D module tail, restored sorted type-2 interpolation, and replaced approximate `I_0` with the defining convergent series.
- Restored `NttPlan` after truncation and verified modular arithmetic, convolution, caller-owned, and property tests.
- Repaired CZT test placement, enabled `Complex64` metadata serialization, and rejected zero-magnitude CZT step parameters.
- Corrected Wavelet Morlet admissibility documentation and kernel by applying the DC correction with a zero-mean numerical proof test.

### Testing and Validation

- Added Python `rfft3`/`irfft3` value-semantic tests documenting the full-spectrum contract and asserting computed output values.
- Added validation report JSON schema-shape tests for required top-level and nested sections.
- Added Criterion benchmark target for Apollo FFT direct, radix-2, and Bluestein kernel strategies.
- Verified zero test failures after each sprint increment.
- Audited external Rust FFT references: `realfft` is not a workspace dependency or source import; `apollo-validation/external-references` gates only optional `rustfft`.

### WGPU Backend Architecture

- Renamed dense FFT WGPU crate to `apollo-fft-wgpu` and updated validation/Python dependencies.
- Added `apollo-nufft-wgpu` with capability, plan, and unsupported-execution contracts.
- Added per-transform WGPU backend crates for CZT, DCT/DST, DHT, FrFT, FWHT, GFT, Hilbert, Mellin, NTT, QFT, Radon, SDFT, SFT, SHT, STFT, and Wavelet.
- Verified each new WGPU crate has domain, application, infrastructure, verification, and README artifacts.

### WGPU Numerical Kernels (First Wave)

- Added direct forward CZT WGPU kernels with CPU parity validation.
- Added forward Hilbert WGPU kernels with CPU parity validation.
- Added forward Mellin WGPU kernels with CPU parity validation.
- Added forward and inverse NTT WGPU kernels with CPU parity validation.
- Added forward and inverse QFT WGPU kernels with CPU parity validation.
- Added forward Radon WGPU kernels with CPU parity validation.
- Added numerical DCT-II/DCT-III/DST-II/DST-III WGPU kernels with CPU parity validation.
- Added numerical DHT WGPU kernels with CPU parity validation.
- Added numerical FWHT WGPU kernels with CPU parity validation.

### WGPU Numerical Kernels (Sprint Completions)

- **QFT WGPU**: `apollo-qft-wgpu` executes forward/inverse unitary QFT by direct O(N^2) summation with 1/sqrt(N) normalization; CPU parity tested.
- **FrFT WGPU**: `apollo-frft-wgpu` executes forward/inverse FrFT via 5-mode dispatch (identity, centred DFT, reversal, centred IDFT, general chirp); `FrftWgpuPlan` carries `order_bits: u32`; CPU parity tested.
- **SDFT WGPU**: `apollo-sdft-wgpu` executes forward direct-bins DFT matching `SdftPlan::direct_bins`; `SdftWgpuPlan` carries `window_len` and `bin_count`; CPU parity tested.
- **GFT WGPU**: `apollo-gft-wgpu` executes forward U^T x and inverse U X by direct matrix-vector product; basis passed at call time; CPU parity tested.
- **STFT WGPU**: `apollo-stft-wgpu` executes forward Hann-windowed STFT per frame; `StftWgpuPlan` carries `frame_len` and `hop_len`; CPU parity tested.
- **Wavelet WGPU**: `apollo-wavelet-wgpu` executes forward/inverse multi-level Haar DWT via two-buffer Mallat decomposition; `WaveletWgpuPlan` carries `len` and `levels`; roundtrip error < 1e-5.
- **SFT WGPU**: `apollo-sft-wgpu` executes dense direct DFT on WGPU, projects top-k support through the `apollo-sft` sparse spectrum contract, and reconstructs by normalized inverse direct DFT; CPU parity tested.
- **NUFFT WGPU**: `apollo-nufft-wgpu` executes exact direct Type-1 and Type-2 summations for 1D and 3D on WGPU; CPU exact-reference parity tested.
- **NUFFT WGPU Fast 1D**: `apollo-nufft-wgpu` executes fast Kaiser-Bessel Type-1 and Type-2 1D paths with GPU spreading/interpolation, `apollo-fft-wgpu` oversampled FFT dispatch, and GPU deconvolution; CPU gridded-reference parity tested.
- **NUFFT WGPU Fast 3D**: `apollo-nufft-wgpu` executes fast Kaiser-Bessel Type-1 and Type-2 3D paths with GPU separable spreading/interpolation, `apollo-fft-wgpu` oversampled 3D FFT dispatch, radix-2 support-safe oversampled dimensions, and GPU separable deconvolution; CPU gridded-reference parity tested.
- **SHT WGPU**: `apollo-sht-wgpu` executes direct complex forward/inverse SHT on WGPU using `apollo-sht` quadrature samples and GPU-generated associated-Legendre/spherical-harmonic basis values; CPU parity tested.
- **SHT WGPU Basis Generation**: moved associated Legendre recurrence, Condon-Shortley negative-order handling, spherical harmonic normalization, conjugation, and quadrature weighting into the WGPU basis-generation pass while preserving `apollo-sht` as the quadrature SSOT.

---

## Remaining Gaps

The following gap is deferred to a future increment. It requires reference-source selection and validation infrastructure work before implementation.


### 1. Published-Reference Audit

Transform-specific analytical tests exist for all crates, but external published-reference fixtures are still pending for a separate validation increment. No external reference data set has been ingested and no cross-library comparison has been executed. Completion requires selecting reference data from published transform tables or a reference implementation and wiring those into the `apollo-validation` external-reference report path.
