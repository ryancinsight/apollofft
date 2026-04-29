# Changelog

All notable changes to the Apollo workspace are documented in this file.
Versioning follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
Change-class tags: [patch] backward-compatible fix, [minor] additive non-breaking, [major] breaking, [arch] cross-cutting redesign.

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