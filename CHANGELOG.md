# Changelog

All notable changes to the Apollo workspace are documented in this file.
Versioning follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
Change-class tags: [patch] backward-compatible fix, [minor] additive non-breaking, [major] breaking, [arch] cross-cutting redesign.

---

## [Unreleased]
*(no unreleased changes)*

---

## [0.12.2] — Closure XXII

### Closure XXII — GPU Benchmark Runner Workflow + Root README Correction [patch]

#### Added
- `.github/workflows/gpu-benchmarks.yml`: manual `workflow_dispatch` workflow targeting a
  labeled self-hosted runner (`[self-hosted, gpu, apollo]`) to execute the WGPU Criterion
  benchmark suites and upload the output bundle as a workflow artifact.
- `scripts/run_gpu_benchmarks.ps1`: reproducible GPU benchmark runner script that stages
  logs, `manifest.json`, `summary.md`, and `target/criterion` under `.benchmarks/gpu-runner/`.
- `.benchmarks/gpu-runner/.gitkeep`: tracked anchor for benchmark artifact staging.

#### Changed
- Root `README.md`: corrected stale CZT/Mellin/STFT/Radon WGPU capability prose and added a
  dedicated "GPU Benchmark Runner" section documenting labels, workflow entry point, executed
  benches, and output staging path.
- `.gitignore`: ignores generated `.benchmarks/gpu-runner/*` outputs while keeping the tracked
  directory anchor.

---

## [0.12.1] — Closure XXI

### Closure XXI — README Documentation Sync for v0.2.0 Inverse Additions [patch]

#### Changed
- `apollo-czt/README.md`: added "Inverse Transform" section documenting
  `CztPlan::inverse`, Björck-Pereyra Vandermonde solve, and `CztError::NotInvertible`
  conditions (non-square plan, node collision). Updated "Verification" section to
  list inverse roundtrip and rejection tests.
- `apollo-mellin/README.md`: added "Inverse Transform" section documenting
  `MellinPlan::inverse_spectrum`, IDFT + exp-resample algorithm, and
  `MellinError::SpectrumLengthMismatch`. Updated "Verification" section.
- `apollo-czt-wgpu/README.md`: updated "Execution Contract" to reflect forward+inverse
  support; added adjoint formula description; removed stale "inverse unsupported" prose.
  Updated "Verification" section to mention GPU inverse roundtrip test.
- `apollo-mellin-wgpu/README.md`: updated "Execution Contract" to document two-pass
  GPU inverse (IDFT + exp-resample); updated capability and verification prose.
- `checklist.md`: added completed Closure XX entry (22 checklist items).

---

## [0.12.0] — Closure XX

### Closure XX — CPU + GPU Inverse Transforms: CZT and Mellin [minor]

#### Added — apollo-czt v0.2.0
- `CztPlan::inverse(spectrum)` — exact Vandermonde solve via Björck-Pereyra algorithm
  (`O(N²)` in-place Newton evaluation). Returns `CztError::NotInvertible` when
  `M ≠ N` or when Vandermonde nodes collide (denominator below `f64::EPSILON * 1024`).
- `CztStorage::inverse_into` — default method adapting `inverse` to in-place storage API.
- `CztError::NotInvertible { reason: &'static str }` variant.
- 5 value-semantic tests: roundtrip at DFT parameters, general `A` offset, non-unit `W`
  spacing, rejection of rectangular plans, rejection of wrong spectrum length.

#### Added — apollo-mellin v0.2.0
- `MellinPlan::inverse_spectrum(spectrum, out_min, out_max, output)` — IDFT of
  log-domain spectrum then exp-resample from log-grid to linear domain.
  Rayon-parallel IDFT for `N ≥ 256`.
- `inverse_log_frequency_spectrum` and `exp_resample` exported from `lib.rs`.
- `MellinError::SpectrumLengthMismatch` variant.
- 4 value-semantic tests: constant-signal roundtrip (`ε < 1e-10`), linear-signal
  roundtrip (interpolation error `< 0.1` for `N = 64`), wrong-length rejection,
  invalid-bounds rejection.

#### Added — apollo-czt-wgpu v0.2.0
- `czt_inverse` WGSL entry point: adjoint formula `x[n] = (A^n/N)·∑_k X[k]·W^{-nk}`,
  exact for unitary DFT parameters.
- `MellinGpuKernel::inverse_pipeline` field; `execute_inverse` dispatches over `N` threads.
- `CztWgpuBackend::execute_inverse(plan, spectrum)` — validates `M == N`, delegates to kernel.
- `WgpuCapabilities::forward_inverse` constructor.
- 2 new GPU tests: roundtrip at DFT parameters, rejection of non-square plan.

#### Added — apollo-mellin-wgpu v0.2.0
- `mellin_inverse_spectrum` WGSL kernel: IDFT pass, spectrum → log-domain samples.
- `mellin_exp_resample` WGSL kernel: exp-resample pass, log-domain → linear output.
- `InverseMellinParamsPod` uniform struct (32 bytes, reuses params buffer slot).
- `MellinGpuKernel::inverse_spectrum_pipeline`, `exp_resample_pipeline`,
  `inv_params_buffer` fields; `execute_inverse` dispatches two GPU passes + readback.
- `MellinWgpuBackend::execute_inverse(plan, spectrum, out_min, out_max, out_len)`.
- `WgpuCapabilities::forward_inverse` constructor.
- 2 new GPU tests: constant-signal roundtrip (`ε < 5e-4`), invalid-domain rejection.

---

## [0.10.0] — Closure XIX

### Closure XIX — StftGpuBuffers Non-PoT Scratch Sizing [minor]

#### Changed
- `StftGpuBuffers::new` now accepts arbitrary `frame_len` (not just power-of-two).
  Scratch buffers (`re_scratch_buf`, `im_scratch_buf`, `frame_data_buf`) are automatically
  sized to `frame_count × M` where `M = chirp_padded_len(frame_len)` when `!frame_len.is_power_of_two()`,
  and `frame_count × frame_len` when PoT.
- `StftWgpuBackend::make_buffers`, `execute_forward_with_buffers`, `execute_inverse_with_buffers`
  no longer return `WgpuError::FrameLenNotPowerOfTwo`; non-PoT `frame_len` now dispatches
  via the buffer-reuse path (kernel auto-selects Chirp-Z for non-PoT at dispatch time).
- Module docstring in `buffers.rs` updated: PoT constraint removed; Closure XIX note added.

#### Added (Tests)
- `make_buffers_accepts_non_power_of_two_frame_len_structurally`: structural verification
  that `make_buffers` accepts non-PoT without returning `FrameLenNotPowerOfTwo`.
- `forward_buffers_non_pot_frame_len_400_when_device_exists` (GPU-gated): buffer-reuse
  forward at `frame_len=400`, GPU output matched against CPU `apollo-stft` reference.
- `inverse_buffers_non_pot_frame_len_400_when_device_exists` (GPU-gated): buffer-reuse
  inverse at `frame_len=400`, WOLA interior-sample recovery comparison.

#### Removed
- Panic condition `assert!(frame_len.is_power_of_two())` in `StftGpuBuffers::new`.

---

## [0.9.0] — Closure XVIII

### Closure XVIII — Non-Power-of-Two STFT GPU Path (Bluestein/Chirp-Z) [minor]

#### Added
- `crates/apollo-stft-wgpu/src/infrastructure/shaders/stft_chirp.wgsl`: Five-pass WGSL
  compute shader implementing the Bluestein Chirp-Z mapping for the STFT: `premul_fwd`,
  `premul_inv`, `pointmul`, `postmul_fwd`, `postmul_inv`. Hann analysis/synthesis windows
  and exp(±πi·n²/N) chirp twiddles are applied on-GPU.
- `crates/apollo-stft-wgpu/src/infrastructure/shaders/stft_chirp_fft.wgsl`: Radix-2
  sub-FFT shader operating on chirp working buffers: `chirp_fft_bitrev`,
  `chirp_fft_butterfly_fwd`, `chirp_fft_butterfly_inv`, `chirp_fft_scale`.
- `crates/apollo-stft-wgpu/src/infrastructure/chirp.rs`: `StftChirpData` struct —
  pre-allocated GPU resources (chirp kernel H, working buffers, bind groups, pipelines)
  for the Bluestein path. `chirp_padded_len(n)` returns `(2n−1).next_power_of_two()`.
- `design_history_file/adr_stft_wgpu_non_pot_chirpz.md`: ADR for the Chirp-Z dispatch
  strategy (Bluestein 1970, Rabiner 1969).
- `ndarray = "0.16"` added as a regular (non-dev) dependency to `apollo-stft-wgpu`
  for CPU-side chirp kernel construction in `StftChirpData::new`.

#### Changed
- `execute_forward` and `execute_inverse` in `StftWgpuBackend` (device.rs) no longer
  return `WgpuError::FrameLenNotPowerOfTwo`; non-power-of-two `frame_len` now
  automatically dispatches the Bluestein/Chirp-Z path.
- `kernel.rs`: non-PoT `frame_len` delegates to `execute_forward_fft_chirp` /
  `execute_inverse_chirp` instead of returning an error.
- `WgpuError::FrameLenNotPowerOfTwo` doc comment updated to reflect that this variant
  is no longer returned by primary dispatch; it may still be returned by
  `make_buffers`, `execute_forward_with_buffers`, `execute_inverse_with_buffers`.
- Verification tests `forward_rejects_non_power_of_two_frame_len` and
  `inverse_rejects_non_power_of_two_frame_len` renamed and inverted to
  `forward_accepts_non_power_of_two_frame_len_chirpz` and
  `inverse_accepts_non_power_of_two_frame_len_chirpz`.

#### Added (Tests)
- `forward_accepts_non_power_of_two_frame_len_structurally`: structural check that
  non-PoT no longer returns `FrameLenNotPowerOfTwo`.
- `forward_accepts_non_power_of_two_frame_len_chirpz`: same for `execute_forward`.
- `inverse_accepts_non_power_of_two_frame_len_chirpz`: same for `execute_inverse`.
- `forward_chirpz_non_pot_frame_len_400_when_device_exists` (GPU-gated): GPU forward
  Chirp-Z at frame_len=400, compared to CPU `apollo-stft` reference, TOL=1e-2.
- `inverse_chirpz_non_pot_frame_len_400_when_device_exists` (GPU-gated): GPU inverse
  Chirp-Z at frame_len=400, WOLA interior-sample recovery, TOL=5e-2.

---

## [0.8.5] — Closure XVII

### Closure XVII — STFT GPU Buffer-Reuse Criterion Benchmarks + README Usage Documentation [patch]
#### Added
- `bench_forward_reuse` benchmark group in `crates/apollo-stft-wgpu/benches/stft_bench.rs`:
  head-to-head comparison of `execute_forward` (allocating) vs `execute_forward_with_buffers`
  (buffer-reuse) at `frame_len` ∈ {256, 512, 1024}. Pre-allocates `StftGpuBuffers` outside
  the bench loop; measures only signal upload + GPU dispatch + readback per iteration.
- `bench_inverse_reuse` benchmark group: same head-to-head comparison for
  `execute_inverse` vs `execute_inverse_with_buffers`.
- Both groups added to `criterion_group!(benches, …)` in `stft_bench.rs`.
- Updated module docstring in `stft_bench.rs` to describe both allocating and buffer-reuse
  paths and their mathematical basis.
- "Buffer Reuse" section in `crates/apollo-stft-wgpu/README.md`: usage snippet showing
  `make_buffers` → `execute_forward_with_buffers` → `fwd_output()` pattern, constraint
  notes for `FrameLenNotPowerOfTwo` and `LengthMismatch` errors.
- "Benchmarks" section in `README.md`: table of all four benchmark groups with
  description and `cargo bench -p apollo-stft-wgpu` invocation.

---

## [0.8.4] — Closure XVI

### Closure XVI — StftGpuBuffers Pre-allocated Buffer Reuse [minor]
#### Added
- `StftGpuBuffers` struct in `crates/apollo-stft-wgpu/src/infrastructure/buffers.rs`:
  pre-allocates all GPU data buffers, staging buffers, bind groups, and per-stage butterfly
  uniform buffers for a fixed `(frame_count, frame_len, signal_len, hop_len)` quad.
  Eliminates 5–8 `device.create_buffer` calls, 4+ `device.create_bind_group` calls,
  and `log₂(N)` uniform-buffer allocations per dispatch.
- `StftWgpuBackend::make_buffers(plan, signal_len)`: constructs a `StftGpuBuffers` for
  a given plan shape with validation identical to the allocating paths.
- `StftWgpuBackend::execute_forward_with_buffers(plan, signal, buffers)`: zero-allocation
  forward STFT dispatch; result in `buffers.fwd_output()`.
- `StftWgpuBackend::execute_inverse_with_buffers(plan, spectrum, signal_len, buffers)`:
  zero-allocation inverse STFT dispatch; result in `buffers.inv_output()`.
- `StftGpuKernel::execute_forward_fft_with_buffers` and `execute_inverse_with_buffers`:
  kernel-level buffered dispatch methods.
- Verification test `reusable_buffers_match_allocating_forward_and_inverse_when_device_exists`
  (#[ignore = "requires wgpu device"]): asserts bit-exact agreement between allocating and
  buffered paths on a bin-aligned sinusoid (k=16, frame_len=512);
  verifies idempotent second-call buffer reuse.
- `pub use infrastructure::buffers::StftGpuBuffers` re-exported from crate root.

---

## [0.8.3] — Closure XV

### Closure XV — Radon FBP GPU Criterion Benchmarks
#### Added
- `crates/apollo-radon-wgpu/benches/radon_wgpu_bench.rs`: criterion benchmark suite with
  `radon_wgpu_forward` and `radon_wgpu_fbp` groups.
- `criterion = "0.5"` in `apollo-radon-wgpu` dev-dependencies.
- `[[bench]] name = "radon_wgpu_bench" harness = false` in `apollo-radon-wgpu/Cargo.toml`.

---

## [0.8.2] — Closure XIV

### Closure XIV — Dead-Code Removal: O(N²) Forward Pipeline
#### Removed
- `StftGpuKernel::execute()`: O(N²) direct DFT forward method (superseded by Closure XII FFT path).
- `forward_pipeline` field and shader creation code from `StftGpuKernel::new()`.
- `shaders/stft.wgsl`: O(N²) forward DFT WGSL shader (superseded by `stft_forward_fft.wgsl`).
- `stft_inverse_frames` entry point from `stft_inverse.wgsl` (superseded by Closure XI FFT inverse).
#### Changed
- `stft_inverse.wgsl` file header updated to reflect single-pass OLA role.
- `kernel.rs` module docstring, `WORKGROUP_SIZE` comment, struct docs updated.

---

## [0.8.1] — Closure XIII

### Closure XIII — STFT GPU Criterion Benchmarks
#### Added
- `crates/apollo-stft-wgpu/benches/stft_bench.rs`: criterion benchmark suite with
  `bench_forward_fft` and `bench_inverse_fft` groups.
- `criterion = { version = "0.5", features = ["html_reports"] }` in `apollo-stft-wgpu`
  dev-dependencies.
- `[[bench]] name = "stft_bench" harness = false` in `apollo-stft-wgpu/Cargo.toml`.

---

## [0.8.0] — Closure XII

### Closure XII — STFT Forward-Path GPU FFT Acceleration
#### Added
- `stft_forward_fft.wgsl`: new GPU shader with `stft_fwd_pack_window`, `stft_fwd_bitrev`,
  `stft_fwd_butterfly`, `stft_fwd_interleave` entry points (DFT twiddle `exp(−2πi·k/N)`).
- `FwdFftStageParams` struct (16 bytes, fields: frame_count, frame_len, hop_len, stage).
- `StftGpuKernel::execute_forward_fft`: O(N log N) GPU forward STFT, PoT frame_len required.
- `FrameLenNotPowerOfTwo` guard in `StftWgpuBackend::execute_forward`.
- Tests: `forward_rejects_non_power_of_two_frame_len`, `forward_fft_roundtrip_large_frame_when_device_exists`.
#### Changed
- `StftWgpuBackend::execute_forward` now routes to the FFT-accelerated path and requires
  power-of-two `frame_len` (previously accepted any `frame_len` via O(N²) direct DFT).
#### Breaking
- `execute_forward` with non-power-of-two `frame_len` now returns
  `Err(WgpuError::FrameLenNotPowerOfTwo)` instead of computing a result.

---

## [0.7.0] — Closure XI

### Added
- `apollo-stft-wgpu`: GPU STFT inverse O(N log N) acceleration. New `stft_inverse_fft.wgsl` with four entry points implementing a batched Cooley-Tukey Radix-2 DIT IFFT: `stft_deinterleave` (interleaved complex f32 → split re/im scratch), `stft_bitrev` (bit-reversal permutation, batched), `stft_butterfly` (one Radix-2 DIT stage per dispatch; IDFT twiddle exp(+2πi·k/N)), `stft_scale_and_window` (1/N scale + Hann synthesis window → frame_data). Two-bind-group architecture: group 0 = 4 data bindings (shared), group 1 = per-stage `FftStageParams` uniform (pre-allocated, one per butterfly pass). All passes in one `CommandEncoder`; implicit per-pass memory barriers ensure write visibility. OLA pass unchanged. Replaces the O(N²) `stft_inverse_frames` pipeline. Formal basis: Cooley & Tukey (1965); Allen & Rabiner (1977) Theorem 1. [minor]
- `apollo-stft-wgpu`: `WgpuError::FrameLenNotPowerOfTwo { frame_len: usize }` error variant. Returned by `execute_inverse` when `frame_len` is not a power of two (Radix-2 invariant); checked before GPU buffer allocation in `device.rs` and at IFFT entry in `kernel.rs`. [minor]
- `apollo-stft-wgpu`: `inverse_rejects_non_power_of_two_frame_len` test (CPU-only; asserts `FrameLenNotPowerOfTwo { frame_len: 6 }` for frame_len=6). [patch]
- `apollo-stft-wgpu`: `inverse_roundtrip_large_frame_1024_samples_when_device_exists` GPU-gated test (frame_len=1024, log₂N=10 butterfly stages, hop=512, signal_len=8192, analytic sine reference, TOL=5e-3). [patch]

---

## [0.6.0] — Closure X

### Added
- `apollo-radon-wgpu`: GPU ramp-filtered backprojection (FBP). New `radon_fbp_filter.wgsl` entry point `radon_fbp_filter` applies the Ram-Lak ramp filter to each sinogram projection row via circular convolution with the impulse response `h = IFFT(R)`, `R[k] = 2π·|signed_k|/(N·Δ)` (Bracewell & Riddle 1967; Shepp & Logan 1974). Filter kernel computed host-side from `apollo_radon::ramp_filter_projection([1,0,...], Δ)` and cast to f32. Two-pass single `CommandEncoder` (filter → backproject). Host-side `π/angle_count` normalization. `RadonWgpuBackend::execute_filtered_backproject`. `supports_filtered_backprojection: bool` field added to `WgpuCapabilities`. `WgpuCapabilities::forward_inverse_and_fbp` constructor added. [minor]
- `apollo-radon-wgpu`: `backproject_satisfies_adjoint_identity_when_device_exists` test verifies the Radon adjoint identity ⟨A·f, g⟩_sinogram = ⟨f, A†·g⟩_image (Natterer 2001, §II.2) to relative tolerance 5e-3. [patch]
- `apollo-stft-wgpu`: `inverse_roundtrip_for_multiple_cola_parameter_sets` tests three COLA-compliant parameter sets (frame_len=8/hop=4, 16/8, 32/16) with analytical reference signals. TOL=5e-3. [patch]

### Fixed
- `README.md`: stale WGPU crate descriptions for `apollo-radon-wgpu` (was "forward only"), `apollo-stft-wgpu` (was "forward only"), `apollo-hilbert-wgpu` (was "inverse unsupported"), and `apollo-sdft-wgpu` (was "inverse unsupported"). All now accurately describe implemented GPU inverse capabilities. [patch]
- `ARCHITECTURE.md`: Mixed-Precision Capability Table notes for `apollo-radon-wgpu`, `apollo-stft-wgpu`, `apollo-hilbert-wgpu`, and `apollo-sdft-wgpu` updated to reflect inverse capability status. [patch]

---

## [0.5.0] — Closure IX

### Added
- `apollo-stft-wgpu`: GPU inverse STFT via two-pass Weighted Overlap-Add (WOLA). New WGSL file `stft_inverse.wgsl` with entry points `stft_inverse_frames` (per-(frame, local_j) windowed IDFT: `frame_data[m·N+j] = (1/N)·Re{Σ_k X[m,k]·exp(+2πi·k·j/N)}·hann(j)`, spectrum as interleaved f32 pairs) and `stft_inverse_ola` (per-sample `y[n] = Σ_m frame_data[m·N+(n−start_m)] / Σ_m hann(n−start_m)²`, `start_m = m·hop−N/2`). Both passes share the existing 3-binding layout in one `CommandEncoder`. `StftGpuKernel::execute_inverse` (2-pass single encoder). `StftWgpuBackend::execute_inverse(plan, spectrum, signal_len)` + `execute_inverse_typed_into`. `WgpuCapabilities::forward_and_inverse` constructor added. Basis: WOLA identity (Allen–Rabiner 1977, Theorem 1). [minor]
- `apollo-radon-wgpu`: GPU Radon adjoint backprojection. New WGSL file `radon_backproject.wgsl` with entry `radon_backproject`: per-pixel `bp[r,c] = Σ_θ interp(sinogram[θ,·], x·cosθ + y·sinθ)` with linear interpolation and out-of-range zero-clamping, reusing the forward bind group layout. `RadonGpuKernel::execute_backproject`. `RadonWgpuBackend::execute_inverse(plan, sinogram, angles)` + `execute_inverse_flat_typed`. `WgpuCapabilities::forward_and_inverse` constructor added. `SinogramShapeMismatch` error variant added. Basis: Radon adjoint operator (Natterer 2001, §II.2). [minor]

### Fixed
- `gap_audit.md`: open-gap note incorrectly stated "CPU inverse paths are implemented" for `apollo-czt-wgpu` and `apollo-mellin-wgpu`. Corrected: those crates have no CPU inverse defined; `execute_inverse` returns `UnsupportedExecution` by architectural design, not by deferral. [patch]

---

## [0.4.0] — Closure VIII

### Added
- `apollo-hilbert-wgpu`: GPU inverse Hilbert transform. New WGSL entry point `hilbert_inverse_mask` recovers the original real-signal DFT spectrum from the DFT of the quadrature signal: positive bins X[k]=j·Q[k], negative bins X[k]=-j·Q[k], DC and Nyquist zeroed (unrecoverable; Bracewell 1965). New `HilbertGpuKernel::execute_inverse` runs 3 sequential passes in one command encoder (DFT of quadrature → inverse mask → IDFT of recovered spectrum). Exposed via `HilbertWgpuBackend::execute_inverse` and `execute_inverse_typed_into`. `WgpuCapabilities::forward_and_inverse` constructor added. [minor]
- `apollo-sdft-wgpu`: GPU inverse SDFT. New WGSL entry point `sdft_inverse_bins` computes x[n]=(1/K)·Σ_{b=0}^{K-1} X[b]·exp(+2πi·b·n/K). Complex bins passed as interleaved f32 pairs. Separate `forward_pipeline` and `inverse_pipeline` in `SdftGpuKernel`. Exposed via `SdftWgpuBackend::execute_inverse` and `execute_inverse_typed_into`. `WgpuCapabilities::forward_and_inverse` constructor added. [minor]

### Fixed
- `apollo-hilbert-wgpu`: pre-existing bug in `hilbert_inverse_dft` WGSL: real accumulator was written back as a stale self-assign (`inout_b[n].re = original`); corrected to `inout_b[n].re = acc.x * scale`. [patch]
- `apollo-czt`: proptest `bluestein_equals_direct_for_arbitrary_parameters` used a fixed 1e-9 absolute tolerance, which is violated when `|w|>1` amplifies output magnitudes by up to |w|^((N-1)²/2). Tolerance changed to `1e-9·max(|direct[k]|,1.0)` (relative bound). Formal basis: Bluestein relative error ≤ C·log₂(p)·ε_machine ≈12·2.2e-16≈2.6e-15 (Higham §3.10). [patch]

---

## [0.3.0] — Closure VII


### Added
- Six new published-reference fixtures in `apollo-validation`: SFT 1-sparse alternating tone, SHT monopole Y₀⁰ coefficient, STFT rectangular-window impulse frame, Hilbert cosine-to-sine 4-point, Mellin constant-function first moment, Radon θ=0 column-impulse projection. Fixture count rises from 22 to 28. [minor]
- Proptest coverage for four CPU transform crates previously lacking property tests: `apollo-czt` (Bluestein-vs-direct parity, spiral-collapse to DFT, linearity), `apollo-frft` (roundtrip, additivity, linearity), `apollo-nufft` (DC-mode invariant, fast-path tracks exact, Type-1 linearity), `apollo-sft` (K-sparse exact recovery, Parseval top-K, retained bins equal DFT). [minor]

### Changed
- `apollo-frft-wgpu`: `UnitaryFrftGpuKernel::execute` refactored from 3 separate command encoder submissions to a single command encoder with 3 sequential compute passes followed by a copy command. This reduces CPU–GPU round-trips from 4 submits + 5 polls to 1 submit + 2 polls while preserving the cross-pass write-visibility guarantee via the implicit memory barrier at each `ComputePass` boundary (WebGPU spec §3.4 sequential pass ordering). [patch]
- `design_history_file/`: removed stale shadow copies of `backlog.md`, `checklist.md`, and `gap_audit.md` (root artifacts are the SSOT). `adr_unitary_frft.md` retained as the authoritative ADR. [patch]

### Fixed
- `README.md`: updated `apollo-validation` fixture count from 10 (stale) to 28 (final Closure VII count) and replaced the stale fixture list with the complete 28-fixture inventory. [patch]

---

## [0.2.0] — Closure VI (NTT WGPU O(N log N), workspace unblock, expanded fixtures)

### Added
- `apollo-ntt-wgpu`: O(N log N) Cooley-Tukey DIT butterfly shader (`ntt_butterfly` + `ntt_scale` entry points), replacing the O(N²) DFT loop. Log₂(N) butterfly passes plus optional scale pass encoded in one command buffer with dynamic uniform offsets. [major]
- Two published-reference fixtures: NTT N=16 impulse (Pollard 1971) and NTT N=16 polynomial product via convolution theorem. Fixture count 20 → 22. [minor]
- CPU-only proptest tests in `apollo-ntt-wgpu` verification: `cpu_roundtrip_preserves_residue_class` and `convolution_theorem_holds_for_arbitrary_pairs`. [minor]

### Changed
- `apollo-ntt-wgpu`: removed `apollo_fft::PrecisionProfile` cross-domain import and `default_precision_profile` field from capabilities; NTT is exact integer arithmetic. [minor]
- GPU-dependent tests in `apollo-ntt-wgpu` annotated `#[ignore = "requires wgpu device"]` replacing silent early-return skips. [patch]

### Fixed
- Workspace compilation regression: reverted `apollo-fft/Cargo.toml` package name from `"apollo"` to `"apollo-fft"` and corrected dependent crate path keys. [patch]
- Removed `#![allow(unused_imports)]` from `apollo-ntt/src/lib.rs` and unused `Array1` import. [patch]

---

## [0.1.9] — Closure V (GPU Unitary FrFT, ADR, published fixtures)

### Added
- `apollo-frft-wgpu`: `UnitaryFrftGpuKernel` implementing DFrFT_a(x)=V·diag(exp(−iakπ/2))·V^T·x on GPU via three-submission pattern. `FrftWgpuBackend` exposes `plan_unitary`, `execute_unitary_forward`, `execute_unitary_inverse`. [minor]
- Three published-reference fixtures (count 17 → 20): FrFT unitary order-2 reversal (Candan 2000), DWT Haar one-level detail (Haar 1910/Mallat 1989), SDFT bin-zero unit impulse. [minor]
- `design_history_file/adr_unitary_frft.md`: ADR documenting Grünbaum eigendecomposition algorithm selection, unitarity proof, GPU ordering guarantee, and tolerance derivation. [patch]
- `ARCHITECTURE.md`: "Key: Unitary FrFT" subsection with CPU/GPU plan comparison table. [patch]

---

## [0.1.8] — Closure IV (FrFT unitarity, DCT-I/IV/DST-I/IV WGPU)

### Added
- `apollo-frft`: `GrunbaumBasis` and `UnitaryFrftPlan` (Candan 2000 eigendecomposition). O(N³) construction, O(N²) per call, provably unitary for all real orders. [minor]
- `apollo-dctdst-wgpu`: WGSL shader modes for DCT-I (mode 4), DCT-IV (mode 5), DST-I (mode 6), DST-IV (mode 7) with correct self-inverse scales. [minor]

---

## [0.1.7] — Closure III (validation mock removal, published fixtures, DCT-I/IV/DST-I/IV CPU)

### Added
- 7 published-reference fixtures (count 10 → 17): FFT inverse, DCT-II inverse pair, DHT self-reciprocal, FWHT 2-point, QFT 2-point, CZT spiral-collapse, GFT path graph. [minor]
- `apollo-validation` GPU suite: real 4×4×4 GpuFft3d roundtrip replacing hardcoded `passed: true` stub. [major]

### Fixed
- `apollo-validation` precision profile forward errors computed from actual GPU vs CPU f64 reference comparison. [patch]

---

## [0.1.6] — Closure II (fixture expansion, capability table)

### Added
- Published-reference fixtures expanded to 10. [minor]
- `ARCHITECTURE.md` Mixed-Precision Capability Table (authoritative per-crate precision record). [patch]

---

## [0.1.5] — Performance & Native GPU Precision

### Added
- `apollo-fft-wgpu`: `GpuFft3dF16Native` behind `native-f16` feature; native f16 arithmetic with `enable f16` WGSL. Bluestein chirp-Z f16 shader for non-power-of-two sizes. [minor]
- Criterion buffer-reuse benchmarks for `apollo-nufft-wgpu` and `apollo-fft-wgpu`. [minor]
- NUFFT and FFT reusable-buffer `with_buffers` façade methods. [minor]

---

## [0.1.4] — Extension Phase (mixed precision rollout, typed storage)

### Added
- Mixed-precision typed storage APIs across all CPU and WGPU transform crates. [minor]
- Exact quantized `u32` NTT-WGPU residue storage and reusable buffer dispatch. [minor]

---

## [0.1.3] — GPU Numerical Kernels (NUFFT, SHT, SFT, STFT, Wavelet, DCT/DST)

### Added
- GPU fast NUFFT 1D/3D Kaiser-Bessel gridding paths. [minor]
- WGPU backends for SHT, STFT, Haar DWT, DCT-II/III/DST-II/DST-III. [minor]

---

## [0.1.2] — Core Transform Crates (GFT, QFT, SDFT, SFT, Radon, Mellin, Hilbert, Wavelet, STFT, CZT, FWHT)

### Added
- New CPU transform crates with WGPU backends. [minor]

---

## [0.1.1] — Foundation (FFT, DHT, DCT/DST, NTT, NUFFT)

### Added
- Core CPU transform crates with O(N log N) kernels. [minor]
- `apollo-validation` published-reference suite with 10 initial fixtures. [minor]

---

## [0.1.0] — Initial release

### Added
- Workspace skeleton: `apollo-fft`, `apollo-fft-wgpu`, `apollo-nufft`, `apollo-nufft-wgpu`, `apollo-validation`, `apollo-python`. [minor]