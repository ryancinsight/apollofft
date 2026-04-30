# Changelog

All notable changes to the Apollo workspace are documented in this file.
Versioning follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
Change-class tags: [patch] backward-compatible fix, [minor] additive non-breaking, [major] breaking, [arch] cross-cutting redesign.

---

## [Unreleased] — Closure XII

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

## [Unreleased] — Closure XI

### Added
- `apollo-stft-wgpu`: GPU STFT inverse O(N log N) acceleration. New `stft_inverse_fft.wgsl` with four entry points implementing a batched Cooley-Tukey Radix-2 DIT IFFT: `stft_deinterleave` (interleaved complex f32 → split re/im scratch), `stft_bitrev` (bit-reversal permutation, batched), `stft_butterfly` (one Radix-2 DIT stage per dispatch; IDFT twiddle exp(+2πi·k/N)), `stft_scale_and_window` (1/N scale + Hann synthesis window → frame_data). Two-bind-group architecture: group 0 = 4 data bindings (shared), group 1 = per-stage `FftStageParams` uniform (pre-allocated, one per butterfly pass). All passes in one `CommandEncoder`; implicit per-pass memory barriers ensure write visibility. OLA pass unchanged. Replaces the O(N²) `stft_inverse_frames` pipeline. Formal basis: Cooley & Tukey (1965); Allen & Rabiner (1977) Theorem 1. [minor]
- `apollo-stft-wgpu`: `WgpuError::FrameLenNotPowerOfTwo { frame_len: usize }` error variant. Returned by `execute_inverse` when `frame_len` is not a power of two (Radix-2 invariant); checked before GPU buffer allocation in `device.rs` and at IFFT entry in `kernel.rs`. [minor]
- `apollo-stft-wgpu`: `inverse_rejects_non_power_of_two_frame_len` test (CPU-only; asserts `FrameLenNotPowerOfTwo { frame_len: 6 }` for frame_len=6). [patch]
- `apollo-stft-wgpu`: `inverse_roundtrip_large_frame_1024_samples_when_device_exists` GPU-gated test (frame_len=1024, log₂N=10 butterfly stages, hop=512, signal_len=8192, analytic sine reference, TOL=5e-3). [patch]

---

## [Unreleased] — Closure X

### Added
- `apollo-radon-wgpu`: GPU ramp-filtered backprojection (FBP). New `radon_fbp_filter.wgsl` entry point `radon_fbp_filter` applies the Ram-Lak ramp filter to each sinogram projection row via circular convolution with the impulse response `h = IFFT(R)`, `R[k] = 2π·|signed_k|/(N·Δ)` (Bracewell & Riddle 1967; Shepp & Logan 1974). Filter kernel computed host-side from `apollo_radon::ramp_filter_projection([1,0,...], Δ)` and cast to f32. Two-pass single `CommandEncoder` (filter → backproject). Host-side `π/angle_count` normalization. `RadonWgpuBackend::execute_filtered_backproject`. `supports_filtered_backprojection: bool` field added to `WgpuCapabilities`. `WgpuCapabilities::forward_inverse_and_fbp` constructor added. [minor]
- `apollo-radon-wgpu`: `backproject_satisfies_adjoint_identity_when_device_exists` test verifies the Radon adjoint identity ⟨A·f, g⟩_sinogram = ⟨f, A†·g⟩_image (Natterer 2001, §II.2) to relative tolerance 5e-3. [patch]
- `apollo-stft-wgpu`: `inverse_roundtrip_for_multiple_cola_parameter_sets` tests three COLA-compliant parameter sets (frame_len=8/hop=4, 16/8, 32/16) with analytical reference signals. TOL=5e-3. [patch]

### Fixed
- `README.md`: stale WGPU crate descriptions for `apollo-radon-wgpu` (was "forward only"), `apollo-stft-wgpu` (was "forward only"), `apollo-hilbert-wgpu` (was "inverse unsupported"), and `apollo-sdft-wgpu` (was "inverse unsupported"). All now accurately describe implemented GPU inverse capabilities. [patch]
- `ARCHITECTURE.md`: Mixed-Precision Capability Table notes for `apollo-radon-wgpu`, `apollo-stft-wgpu`, `apollo-hilbert-wgpu`, and `apollo-sdft-wgpu` updated to reflect inverse capability status. [patch]

---

## [Unreleased] — Closure IX

### Added
- `apollo-stft-wgpu`: GPU inverse STFT via two-pass Weighted Overlap-Add (WOLA). New WGSL file `stft_inverse.wgsl` with entry points `stft_inverse_frames` (per-(frame, local_j) windowed IDFT: `frame_data[m·N+j] = (1/N)·Re{Σ_k X[m,k]·exp(+2πi·k·j/N)}·hann(j)`, spectrum as interleaved f32 pairs) and `stft_inverse_ola` (per-sample `y[n] = Σ_m frame_data[m·N+(n−start_m)] / Σ_m hann(n−start_m)²`, `start_m = m·hop−N/2`). Both passes share the existing 3-binding layout in one `CommandEncoder`. `StftGpuKernel::execute_inverse` (2-pass single encoder). `StftWgpuBackend::execute_inverse(plan, spectrum, signal_len)` + `execute_inverse_typed_into`. `WgpuCapabilities::forward_and_inverse` constructor added. Basis: WOLA identity (Allen–Rabiner 1977, Theorem 1). [minor]
- `apollo-radon-wgpu`: GPU Radon adjoint backprojection. New WGSL file `radon_backproject.wgsl` with entry `radon_backproject`: per-pixel `bp[r,c] = Σ_θ interp(sinogram[θ,·], x·cosθ + y·sinθ)` with linear interpolation and out-of-range zero-clamping, reusing the forward bind group layout. `RadonGpuKernel::execute_backproject`. `RadonWgpuBackend::execute_inverse(plan, sinogram, angles)` + `execute_inverse_flat_typed`. `WgpuCapabilities::forward_and_inverse` constructor added. `SinogramShapeMismatch` error variant added. Basis: Radon adjoint operator (Natterer 2001, §II.2). [minor]

### Fixed
- `gap_audit.md`: open-gap note incorrectly stated "CPU inverse paths are implemented" for `apollo-czt-wgpu` and `apollo-mellin-wgpu`. Corrected: those crates have no CPU inverse defined; `execute_inverse` returns `UnsupportedExecution` by architectural design, not by deferral. [patch]

---

## [Unreleased] — Closure VIII

### Added
- `apollo-hilbert-wgpu`: GPU inverse Hilbert transform. New WGSL entry point `hilbert_inverse_mask` recovers the original real-signal DFT spectrum from the DFT of the quadrature signal: positive bins X[k]=j·Q[k], negative bins X[k]=-j·Q[k], DC and Nyquist zeroed (unrecoverable; Bracewell 1965). New `HilbertGpuKernel::execute_inverse` runs 3 sequential passes in one command encoder (DFT of quadrature → inverse mask → IDFT of recovered spectrum). Exposed via `HilbertWgpuBackend::execute_inverse` and `execute_inverse_typed_into`. `WgpuCapabilities::forward_and_inverse` constructor added. [minor]
- `apollo-sdft-wgpu`: GPU inverse SDFT. New WGSL entry point `sdft_inverse_bins` computes x[n]=(1/K)·Σ_{b=0}^{K-1} X[b]·exp(+2πi·b·n/K). Complex bins passed as interleaved f32 pairs. Separate `forward_pipeline` and `inverse_pipeline` in `SdftGpuKernel`. Exposed via `SdftWgpuBackend::execute_inverse` and `execute_inverse_typed_into`. `WgpuCapabilities::forward_and_inverse` constructor added. [minor]

### Fixed
- `apollo-hilbert-wgpu`: pre-existing bug in `hilbert_inverse_dft` WGSL: real accumulator was written back as a stale self-assign (`inout_b[n].re = original`); corrected to `inout_b[n].re = acc.x * scale`. [patch]
- `apollo-czt`: proptest `bluestein_equals_direct_for_arbitrary_parameters` used a fixed 1e-9 absolute tolerance, which is violated when `|w|>1` amplifies output magnitudes by up to |w|^((N-1)²/2). Tolerance changed to `1e-9·max(|direct[k]|,1.0)` (relative bound). Formal basis: Bluestein relative error ≤ C·log₂(p)·ε_machine ≈12·2.2e-16≈2.6e-15 (Higham §3.10). [patch]

---

## [Unreleased] — Closure VII


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