# Apollo Gap Audit

## Open Gaps

- Mixed-precision CPU storage contracts remain incomplete for NUFFT and SHT.
- Mixed-precision GPU/cudatile capability contracts remain incomplete for FFT-WGPU, CZT-WGPU, DCTDST-WGPU, DHT-WGPU, FrFT-WGPU, FWHT-WGPU, GFT-WGPU, Hilbert-WGPU, Mellin-WGPU, NTT-WGPU, NUFFT-WGPU, QFT-WGPU, Radon-WGPU, SDFT-WGPU, SFT-WGPU, SHT-WGPU, STFT-WGPU, Wavelet-WGPU, and cudatile.

## Closed Gaps

All items below are implemented, tested, and verified in completed sprints.

### Closure Phase

- Fixed `[workspace.lints.clippy]` priority: assigned `all` and `pedantic` groups `priority = -1` so individual overrides at default priority 0 take precedence; eliminated 22 clippy compilation failures across all transform crates.
- Propagated workspace lints to all 39 crates via `[lints] workspace = true` in every `Cargo.toml`; added comprehensive DSP-appropriate pedantic suppressions (cast truncation/precision/loss, needless_range_loop, too_many_arguments, manual_is_multiple_of, manual_div_ceil, etc.).
- Fixed `apollo-fft` doc-lint warnings: replaced `- ` list markers with `* ` in `direct.rs` module doc; replaced `for k in 0..n { output[k] = }` with `iter_mut().enumerate()` in `dft_forward` and `dft_inverse`.
- Replaced `CpuBackend::default()` with `CpuBackend` (unit-struct literal) in `apollo-fft` transport tests to satisfy `clippy::default_constructed_unit_structs`.
- Added `#![allow(missing_docs)]` and doc comments to `apollo-fft/benches/kernel_strategy.rs`.
- Added `fast_type2_1d_normalization_invariance_when_device_exists` test to `apollo-nufft-wgpu` verification: single non-zero coefficient at k=0, verifies GPU output matches CPU gridded reference and that output is constant across positions (detects 1/m rescaling regressions).
- Added normalization convention documentation to `nufft_fast_1d.wgsl` (Type-1 unnormalized forward FFT, Type-2 host pre-scales deconv by m to compensate normalized IFFT), `nufft_fast_3d.wgsl` (3D Type-2 uses normalized IFFT directly, no pre-scaling needed), and `GpuFft3d::encode_inverse_split` doc comment (caveat for unnormalized-IDFT consumers).
- Removed 22 scratch/temporary files from repository root and `scratch/` directory.
- Added scratch-file gitignore patterns to `.gitignore`.
- Verified zero clippy errors, zero clippy warnings, zero test failures across full workspace.

### Workspace and Infrastructure

- Registered every `crates/apollo-*` crate in the root workspace.
- Replaced incomplete `apollo-validation` orchestration with computed CPU, GPU-surface, NUFFT, external-reference, benchmark, and environment reports.
- Added real crate roots for `apollo-frft`, `apollo-gft`, and `apollo-stft`.
- Split `apollo-validation` external references behind an optional validation-only feature so `rustfft` is validation-only; audited that `realfft` is absent from the workspace dependency graph.
- Completed `apollo-validation` with the new multi-crate API surface and conditional external-backend wiring.
- Aligned `apollo-python` with current crate names, shape newtypes, and full-spectrum FFT plan APIs.
- Added crate-local architecture README files for all `crates/apollo-*` crates.
- Re-audited all 39 workspace crates for manifest, README, and library-root presence; added missing `apollo-python` architecture, mathematical contract, precision contract, and verification README sections.

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
- Removed host-side zero-vector initialization for `apollo-sht-wgpu` generated basis storage; GPU basis generation now writes directly into device-allocated storage before reduction.
- Removed host-side zero-vector uploads for inactive `apollo-nufft-wgpu` fast-path placeholder bindings; shared layouts now bind device-only storage where shader entry points do not read that binding.
- Removed full-field `Vec<Vec<Complex>>` lane copies for contiguous `apollo-fft` 2D row passes and 3D innermost-axis passes; Rayon now transforms backing-slice chunks in place, preserving parallelism while reducing peak pass memory and scatter traffic.
- Added caller-owned 3D typed FFT forward/inverse paths for `f64`, `f32`, and mixed `f16` storage profiles, allowing repeated memory-bound 3D workloads to reuse output and scratch spectra.
- Extended validation precision benchmarks so forward and inverse timing reports cover high-accuracy `f64`, low-precision `f32`, and mixed `f16` storage profiles.
- Added typed caller-owned DHT and DCT/DST execution paths for `f64`, `f32`, and mixed `f16` storage profiles without duplicating transform kernels.
- Added typed caller-owned FWHT execution paths for `f64`, `f32`, and mixed `f16` storage profiles without duplicating the Hadamard butterfly schedule.
- Added typed caller-owned CZT execution paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles without duplicating the Bluestein transform path.
- Added typed caller-owned FrFT execution paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles without duplicating the direct fractional-kernel path.
- Added typed caller-owned GFT execution paths for `f64`, `f32`, and mixed `f16` storage profiles without duplicating the graph-basis multiply path.
- Added typed caller-owned Hilbert quadrature paths for `f64`, `f32`, and mixed `f16` storage profiles without duplicating the analytic-mask path.
- Added typed caller-owned Mellin log-resample paths for `f64`, `f32`, and mixed `f16` storage profiles without duplicating the log-scale interpolation, moment, or spectrum paths.
- Added typed caller-owned QFT execution paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles without duplicating the dense unitary QFT path.
- Added typed caller-owned Radon forward/backprojection paths for `f64`, `f32`, and mixed `f16` storage profiles without duplicating the discrete projection or adjoint paths.
- Added typed caller-owned SDFT direct-bin paths for `f64`/`Complex64`, `f32`/`Complex32`, and mixed `f16`/`[f16; 2]` storage profiles without duplicating the direct DFT bin kernel.
- Added typed caller-owned STFT forward/inverse paths for `f64`/`Complex64`, `f32`/`Complex32`, and mixed `f16`/`[f16; 2]` storage profiles without duplicating the frame/window/FFT execution path.
- Added typed caller-owned Wavelet DWT/CWT paths for `f64`, `f32`, and mixed `f16` storage profiles without duplicating the orthogonal filter-bank or continuous wavelet kernels.
- Added typed caller-owned SFT sparse forward/inverse paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles without duplicating the dense FFT, top-K selection, or sparse inverse path.

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
- Added published-reference validation fixtures for DFT, DHT, DCT-II, and DST-II under `external.published_references`, with per-fixture max-error thresholds and schema coverage.

### Published-Reference Audit

- Added independent CZT–DFT cross-check in `apollo-czt`: spiral-collapse theorem verified against `apollo_fft::fft_1d_complex` (independent Cooley-Tukey/Bluestein path).
- Added NUFFT uniform-grid DFT equivalence in `apollo-nufft`: type-1 at x_j = j·L/N matches DFT(c) to < 1e-10.
- Replaced existence-only Morlet CWT test in `apollo-wavelet` with resonance test: CWT at matched scale dominates by factor > 2 over mismatched scale.
- Added DHT–Fourier relationship cross-check in `apollo-dht`: H[k] = Re(F[k]) − Im(F[k]) verified against independent `apollo_fft` computation.
- Fixed hardcoded `type2_1d_max_relative_error = 0.0` mock in `apollo-validation`: replaced with computed fast vs. exact type-2 NUFFT relative error.

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

- **NUFFT WGPU Fast Type-2 1D Normalization Bug (fixed)**: `execute_fast_type2_1d` in `kernel.rs` was producing results a factor of `oversampled_len` (= m) too small. Root cause: the CPU `type2_into` path calls a normalized IFFT (divides by m) and then explicitly multiplies by m to recover the unnormalized IDFT required by the KB interpolation kernel; the GPU path called `encode_inverse_split` (which also divides by m) but omitted the compensating ×m scale. Fix: in `execute_fast_type2_1d`, deconv values are packed into `ComplexPod` with `oversampled_len as f32` scaling before the GPU grid-load pass, so the normalized IFFT output equals the unnormalized IDFT without adding a second host-side deconv vector. The 3D path is unaffected: both CPU and GPU 3D type-2 paths use the normalized IDFT directly without rescaling, so they agree.

---

## Remaining Gaps

No open gaps are recorded after the current sprint. Future increments should expand published-reference fixture breadth where a transform has a compact stable table or a documented external reference implementation that can be vendored or feature-gated without affecting production crates.
