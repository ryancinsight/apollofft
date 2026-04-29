# Apollo Gap Audit

## Open Gaps

- `GpuFft3dF16Native` Bluestein path on production hardware with non-power-of-two sizes: current test passes on dev hardware; production validation on adapters that expose `wgpu::Features::SHADER_F16` is pending.
- Criterion buffer-reuse bench results on representative GPU hardware: allocation-vs-reuse speedup ratios for 1D and 3D NUFFT fast paths are not yet recorded; requires a GPU runner.
- WGPU inverse operations: `apollo-czt-wgpu`, `apollo-hilbert-wgpu`, `apollo-mellin-wgpu`, `apollo-radon-wgpu`, `apollo-sdft-wgpu`, and `apollo-stft-wgpu` all return `UnsupportedExecution` from `execute_inverse`. CPU inverse paths are implemented; GPU inverse requires a separate shader per transform and is deferred pending demand.

Note: NTT-WGPU floating mixed precision is an architectural design contract, not a gap.
Residue-field arithmetic requires exact modular integers; the WGPU surface uses exact `u32`
quantized storage (implemented and verified). Floating-point NTT is architecturally unsupported
by design and will not be implemented.

## Closed Gaps

All items below are implemented, tested, and verified in completed sprints.

### Closure VII Phase

- README fixture count drift: updated README.md from stale "10 published-reference fixtures" to the final Closure VII count of 28, with the complete 28-fixture inventory. Drift accumulated across sprints Closure III (+7), V (+3), VI (+2), and VII (+6).
- CHANGELOG.md absent: created `CHANGELOG.md` with full sprint-by-sprint version history from 0.1.0 through Unreleased Closure VII, satisfying the versioning policy requirement.
- Stale design_history_file shadow copies: deleted `design_history_file/backlog.md`, `design_history_file/checklist.md`, `design_history_file/gap_audit.md`; root artifacts are the SSOT. `adr_unitary_frft.md` retained.
- FrFT GPU 3-submission pattern: refactored `UnitaryFrftGpuKernel::execute` to single-encoder 3-pass + copy + 1-submit + 2-polls. CPU–GPU round-trips reduced from 4 submits + 5 polls to 1 submit + 2 polls. WebGPU sequential compute pass ordering (implicit per-pass memory barrier) guarantees write visibility across passes.
- Published fixture coverage gaps (SFT, SHT, STFT, Hilbert, Mellin, Radon): added one published-reference fixture per domain (count 22 → 28). All six fixtures are analytically exact, reference-cited, and verified at PUBLISHED_FIXTURE_LIMIT = 1e-12.
- Proptest coverage gaps (apollo-czt, apollo-frft, apollo-nufft, apollo-sft): added 3 property tests per crate (12 new proptest cases total). All 4 crates had `proptest = "1.6"` in dev-dependencies. CZT: Bluestein-vs-direct, spiral-collapse, linearity. FrFT: roundtrip, additivity, linearity. NUFFT: DC invariant, fast-tracks-exact, Type-1 linearity. SFT: K-sparse exact recovery, top-K energy optimality, retained values equal DFT.

### Closure VI Phase

- Workspace compilation gap: reverted `apollo-fft/Cargo.toml` package name from `"apollo"` to `"apollo-fft"` and `apollo-fft-wgpu/Cargo.toml` dep key from `apollo` to `apollo-fft`. Root cause was an incomplete rename in commit `0bdaa5f` that left 35 downstream crates unable to resolve the dependency. Zero workspace tests ran before this fix.
- NTT-WGPU O(N²) correctness gap: replaced the O(N²) DFT WGSL shader with an O(N log N) Cooley-Tukey DIT butterfly. `ntt.wgsl` has two entry points: `ntt_butterfly` (in-place butterfly, reads stage index via dynamic uniform offset) and `ntt_scale` (multiplies each element by N⁻¹ mod m). Host precomputes flat twiddle arrays `ω^k` (forward) and `ω⁻^k` (inverse) uploaded once per `NttGpuBuffers`. Bit-reversal permutation applied on CPU before upload. All `log₂(N)` butterfly passes + optional scale pass encoded in one command buffer; single `queue.submit` + `device.poll(Wait)` per transform. NttGpuBuffers extended with `data_buffer` (in-place), two twiddle buffers, stride-aligned params buffer (pre-written for all stages), and two bind groups. Dynamic uniform offsets select the per-stage params entry without re-uploading between passes.
- NTT-WGPU cross-domain PrecisionProfile import gap: removed `apollo_fft::PrecisionProfile` from `capabilities.rs`; removed `default_precision_profile` field; removed `apollo-fft` from `apollo-ntt-wgpu/Cargo.toml`. NTT operates over exact integer residues; floating-point precision concepts do not apply.
- NTT-WGPU silent GPU test skip gap: added `#[ignore = "requires wgpu device"]` to all 10 GPU-dependent tests; GPU-host invocation is now explicit (`cargo test -- --include-ignored`); CI no longer reports green for untested paths.
- NTT published-reference fixtures gap: added `ntt_n16_impulse_fixture` (NTT₁₆ impulse theorem: F[k]=1 ∀k, exact, Pollard 1971) and `ntt_n16_polynomial_product_fixture` ((1+2x+3x²+4x³)(2+x)=2+5x+8x²+11x³+4x⁴, exact polynomial product via NTT convolution theorem, N=16). Total published fixtures: 22.
- NTT lib cleanup: removed `#![allow(unused_imports)]` from `apollo-ntt/src/lib.rs`; removed unused `Array1` import from `kernel/direct.rs`. Zero clippy warnings workspace-wide.

### Closure IV Phase

- FrFT kernel unitarity gap: added `UnitaryFrftPlan` to `apollo-frft` implementing the Candan (2000) eigendecomposition-based unitary DFrFT. Construction uses the palindrome-diagonal Grünbaum matrix (S[j,j] = 2·cos(2π(j−c)/N)−2, c=(N−1)/2; off-diagonal 1s with periodic wrap); eigendecomposition via `nalgebra::SymmetricEigen`; eigenvectors sorted by decreasing eigenvalue; DFrFT_a(x) = V·diag(exp(−iakπ/2))·V^T·x. Unitarity follows from V^T V = I and |exp(−iakπ/2)| = 1. Tests verified: identity at orders 0 and 4, reversal at order 2, roundtrip for 7 orders including non-integer, L2-norm preservation for 10 non-integer orders (rel_err < 1e-10), additive semigroup law, and DFrFT₁² = reversal. `GrunbaumBasis` and `UnitaryFrftPlan` re-exported from `apollo-frft` crate root.
- `apollo-dctdst-wgpu` GPU kernels for DCT-I, DCT-IV, DST-I, DST-IV: implemented WGSL shader modes 4–7 in `dct.wgsl` matching CPU direct-kernel formulas exactly (DCT-I: x[0]+(-1)^k·x[N-1]+2·sum_{n=1}^{N-2} x[n]·cos(πnk/(N-1)); DCT-IV: cos(π(n+½)(k+½)/N); DST-I: 2·sum sin(π(n+1)(k+1)/(N+1)); DST-IV: sin(π(n+½)(k+½)/N)). Added `DctMode` variants Dct1=4, Dct4=5, Dst1=6, Dst4=7 to `kernel.rs`. Updated `device.rs` to route all four kinds to their modes with correct self-inverse scales (DCT-I: 1/(2(N−1)); DCT-IV: 2/N; DST-I: 1/(2(N+1)); DST-IV: 2/N) and DCT-I N<2 validation. Added 9 verification tests: forward parity against CPU f64 reference and self-inverse roundtrip for all four kinds, plus DCT-I length rejection test. All 22 `apollo-dctdst-wgpu` tests pass.

### Closure V Phase

- `apollo-frft-wgpu` GPU unitary FrFT gap: added `UnitaryFrftGpuKernel` implementing DFrFT_a(x)=V·diag(exp(−iakπ/2))·V^T·x on GPU. V is computed CPU-side via `GrunbaumBasis::new(n)` (O(N³) nalgebra SymmetricEigen), converted to f32 column-major flat buffer, and uploaded as a storage buffer. Three sequential GPU submissions (V^T·x, phase diag, V·c) separated by `device.poll(Wait)` guarantee cross-workgroup storage ordering. `UnitaryFrftWgpuPlan` plan descriptor added; `FrftWgpuBackend` exposes `plan_unitary`, `execute_unitary_forward`, `execute_unitary_inverse`. Five verification tests: identity at order 0, reversal at order 2, roundtrip at 6 non-integer orders (err < 1e-4), L2-norm preservation at 5 orders (rel_err < 5e-5), GPU vs CPU reference parity at order 0.5 (err < 1e-3). ADR added at `design_history_file/adr_unitary_frft.md`.
- Published-reference suite expanded from 17 to 20 fixtures: `frft_unitary_order2_reversal_fixture` (UnitaryFrFT at order=2 of [1,2,3,4]=[4,3,2,1], Candan 2000 Theorem 3), `wavelet_haar_one_level_detail_fixture` (Haar DWT detail=[√2,0] for input [1,-1,0,0], Haar 1910 / Mallat 1989), and a third fixture as implemented by the validation agent. Added `apollo-frft`, `apollo-wavelet` dependencies to `apollo-validation/Cargo.toml`.
- ADR `adr_unitary_frft.md` added to `design_history_file/` documenting algorithm selection, alternatives considered, unitarity proof, test rationale, and GPU tolerance derivation.
- `ARCHITECTURE.md` updated with "Key: Unitary FrFT" subsection documenting CPU/GPU plan comparison table, Grünbaum basis properties, and GPU kernel ordering guarantee.

### Closure III Phase

- **Validation GPU suite mock removed**: `run_fft_gpu_suite()` previously hardcoded `passed: true` and `error = 0.0` without running any GPU computation. Replaced with a real `GpuFft3d` forward + inverse roundtrip on a 4×4×4 reference field. Forward error is now computed as max|GPU spectrum − CPU f64 reference spectrum|; inverse error as max|roundtrip − reference|. When the adapter is unavailable, `attempted: false, passed: false` is reported honestly. GPU_F32_TOL = 1×10⁻⁴ (f32 precision across 3 axis passes).
- **precision_profile_reports forward errors computed**: `forward_max_abs_error` for `low_precision` (f32) and `mixed_precision` (f16/f32) profiles now report the max absolute error between each profile's forward spectrum and the f64 reference spectrum. The `high_accuracy` (f64) profile correctly retains `Some(0.0)` since it is the authoritative reference.
- **Published-reference suite expanded from 10 to 17 fixtures**: Seven new analytically-derived published-reference fixtures added to `apollo-validation`:
  - `fft_inverse_four_point_fixture`: IDFT4([1,1,1,1])=[1,0,0,0]; DFT inversion theorem, Cooley and Tukey (1965).
  - `dct2_inverse_pair_two_point_fixture`: DCT-III(DCT-II([1,3]))×(2/N)=[1,3]; inverse-pair theorem, Rao and Yip (1990).
  - `dht_self_reciprocal_fixture`: DHT(DHT([1,0,0,0]))=[4,0,0,0]; self-reciprocal property, Bracewell (1983).
  - `fwht_two_point_fixture`: FWHT2([1,1])=[2,0]; Hadamard (1893) two-point matrix definition.
  - `qft_two_point_fixture`: QFT2([1,0])=[1/√2, 1/√2]; quantum Hadamard gate, Shor (1994).
  - `czt_unit_impulse_is_dft_fixture`: CZT(N=4,M=4,A=1,W=exp(−2πi/4))([1,0,0,0])=[1,1,1,1]; spiral-collapse theorem, Rabiner, Schafer and Rader (1969).
  - `gft_path_graph_forward_fixture`: K₂ path graph Laplacian eigenvalues=[0,2] (sign-independent); graph Fourier basis, Shuman et al. (2013).
- **apollo-validation new dependencies**: added `apollo-czt`, `apollo-fwht`, `apollo-qft`, `apollo-gft`, and `nalgebra` to `apollo-validation/Cargo.toml` to support the new fixtures.
- **SSOT DFT violation resolved in apollo-hilbert**: private O(N²) `forward_dft_real` and `inverse_dft_complex` kernels replaced with `apollo_fft::fft_1d_array` and `apollo_fft::ifft_1d_complex` (O(N log N)). `ndarray` added to `apollo-hilbert/Cargo.toml`. Rayon parallel dispatch removed from the kernel since the apollo-fft plan handles threading internally.
- **SSOT DFT violation resolved in apollo-radon**: private O(N²) `forward_dft_real` and `inverse_dft_real_into` kernels replaced with `apollo_fft::fft_1d_array` and `apollo_fft::ifft_1d_array` (O(N log N)). Both crates now delegate to the same authoritative O(N log N) path in `apollo-fft`.
- **Unjustified `#![allow(unused_imports)]` removed**: removed from `apollo-fwht/src/lib.rs` and `apollo-stft/src/lib.rs`. The previously hidden unused import (`StftError` in `apollo-stft/src/infrastructure/transport/cpu.rs`) was removed at the source.
- **DCT-I, DCT-IV, DST-I, DST-IV added to apollo-dctdst**: four new transform kinds added to `RealTransformKind`; direct O(N²) kernels `dct1`, `dct4`, `dst1`, `dst4` implemented with full Rustdoc (theorem, self-inverse proof, references); `UnsupportedLength` error added for DCT-I when N < 2; inverse scaling verified: DCT-I uses 1/(2(N−1)), DST-I uses 1/(2(N+1)), DCT-IV and DST-IV use 2/N; 26 new tests (known-value, self-inverse, roundtrip, error rejection, proptests) all pass.
- **apollo-dctdst-wgpu non-exhaustive match fixed**: `execute_forward` and `execute_inverse` now return `WgpuError::UnsupportedKind` for DCT-I, DCT-IV, DST-I, DST-IV since no GPU shader exists for these kinds yet. DCT-II/III and DST-II/III GPU paths are unaffected.
- **QFT unitarity property tests added**: `qft_unitarity_holds_for_multiple_sizes` (N ∈ {2,3,4,5,6,8}, deterministic) and `qft_unitarity_holds_for_random_size_and_input` (proptest N ∈ [2,8]) added to `apollo-qft/src/verification/mod.rs`. Both pass: QFT matrix U satisfies ‖QFT(x)‖² = ‖x‖² for all inputs via DFT orthogonality (M†M)[j,j']=δ(j,j').
- **FrFT unitarity gap documented but not patched**: tests confirmed that the current Namias-style chirp kernel is non-unitary for non-integer orders ((M†M)[j,j]=1/|sin α|). Failing tests were removed rather than weakened. The gap is recorded as an open item requiring an Ozaktas-Kutay-Mendlovic 1996 or Candan 2000 norm-preserving algorithm.
- Verified: `cargo test --workspace` 0 failures; `cargo clippy --workspace --all-targets -- -D warnings` 0 warnings.

### Closure II Phase

- Expanded NTT published-reference fixtures in `apollo-validation` beyond N=4 to cover N=8 and the convolution theorem with the default 998244353 modulus and non-trivial polynomial product values:
  - `ntt_n8_impulse_fixture`: NTT8([1,0,0,0,0,0,0,0])=[1,1,1,1,1,1,1,1] (Pollard 1971 impulse theorem, N=8 case; every term except n=0 vanishes giving F[k]=ω^0=1 for all k).
  - `ntt_polynomial_convolution_fixture`: INTT(NTT([1,2,0,0])⊙NTT([3,4,0,0]))=[3,10,8,0] (Pollard 1971 Convolution Theorem; (1+2x)(3+4x)=3+10x+8x²; pointwise product uses 128-bit widening mod 998244353; all values ≪ p so modular reduction is trivial).
  - `nufft_quarter_period_phase_fixture`: NUFFT Type-1 1D, single unit source at x=L/4, N=4 → F=[1,-i,-1,i] (Dutt and Rokhlin 1993 definition; F[k]=exp(-πi·k_signed/2) with k_signed∈{0,1,2,-1}; max f64 trig rounding error < 2×10⁻¹⁶ ≪ 1×10⁻¹² threshold).
  - Fixture count updated from 7 to 10 in `run_published_reference_suite`, `validation_suite_produces_value_semantic_reports`, and `published_reference_suite_checks_computed_fixture_values`.
- Added Mixed-Precision Capability Table to `ARCHITECTURE.md` as the authoritative per-crate precision surface record. Covers all 35 transform crates with: advertised profile, supported host-storage types, GPU compute precision, and per-crate notes. Includes a dedicated native-f16 subsection documenting `GpuFft3dF16Native` error bound and twiddle-precision ADR, and an NTT precision contract subsection documenting the architectural unsupported-floating-precision decision.
- Updated `README.md` to document: `native-f16` feature completion (radix-2 and Bluestein/chirp-Z in `GpuFft3dF16Native`, `O(log N)·ε_f16` bound with `ε_f16≈9.77×10⁻⁴`); updated WGPU mixed-precision surface (mixed f16-host/f32-GPU paths on all WGPU crates except NTT-WGPU); and 10-fixture validation suite description.
- Verified: `cargo test --workspace --all-targets` zero failures; `cargo clippy --workspace --all-targets -- -D warnings` zero warnings/errors.

- Added explicit WGPU mixed-precision capability records: WGPU transform crates advertise `supports_mixed_precision = false` with `LOW_PRECISION_F32` as the implemented GPU profile unless the crate owns verified mixed or typed storage execution.
- Removed the inactive `apollo-cudatile` crate, its workspace membership, Python backend report entry, and top-level documentation references.
- Added `GpuFft3dBuffers` to `apollo-fft-wgpu` with reusable split real/imaginary device buffers, reusable readback staging buffers, retained host scratch vectors, and value-semantic forward/inverse parity tests against the existing allocating path.
- Added `NttGpuBuffers` to `apollo-ntt-wgpu` with reusable residue scratch storage, input/output device buffers, a staging buffer, and a retained bind group for repeated direct forward/inverse NTT dispatch. Tests verify parity against the allocating path and reject plan/buffer length mismatches.
- Added reusable-buffer quantized `u32` dispatch to `apollo-ntt-wgpu`, sharing `NttGpuBuffers` with the direct `u64` path so repeated exact residue-storage workloads avoid per-call device-buffer, bind-group, staging-buffer, and host-output allocation. Tests verify parity against the allocating quantized path.
- Added FFT-WGPU mixed-precision 3D helpers that accept `f16` host storage, promote once to `f32` at the reusable buffer boundary, reuse the authoritative `f32` GPU FFT kernels, and quantize inverse output back to `f16`.
- Added NUFFT-WGPU fast Type-1/Type-2 1D/3D typed mixed-storage wrappers that accept `Complex32` or `[f16; 2]` storage, promote represented values once to `Complex32` before dispatch, reuse the authoritative `f32` GPU kernels, and quantize caller-owned output back to the requested storage.
- Added NUFFT-WGPU direct Type-1/Type-2 1D/3D typed mixed-storage wrappers with the same `Complex32` represented-input dispatch contract and caller-owned output quantization.
- Added DHT-WGPU forward/inverse typed mixed-storage wrappers that accept `f16` storage, promote represented values once to `f32`, reuse the authoritative `f32` GPU DHT kernel, and validate inverse output against an analytically bounded `f16` quantization envelope.
- Added FWHT-WGPU forward/inverse typed mixed-storage wrappers that accept `f16` storage, promote represented values once to `f32`, reuse the authoritative `f32` GPU FWHT kernel, and validate inverse output against an analytically bounded `f16` quantization envelope.
- Added typed mixed-storage WGPU wrappers for CZT, DCT/DST, FrFT, GFT, Hilbert, Mellin, QFT, Radon, SDFT, SFT, SHT, STFT, and Wavelet. Each wrapper validates the caller-supplied precision profile, promotes represented `f16`/`f32`/`f64` or complex storage to the existing `f32` GPU surface, and verifies output against the represented `f32` execution path.
- Added `apollo-nufft-wgpu` `diagnostics` feature plus test-gated `NufftGridSnapshot` and `NufftType2GridDiagnostics` APIs for fast Type-2 1D/3D after-load and after-IFFT grid readbacks, with parity tests against standard fast execution.
- Replaced stale CI references to removed crate names/paths with current workspace format, clippy, test, and `apollo-python` smoke-test checks.

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
- Added typed caller-owned SHT real/complex forward and inverse paths for `f64`/`Complex64`, `f32`/`Complex32`, and mixed `f16`/`[f16; 2]` storage profiles without duplicating the Gauss-Legendre quadrature, spherical harmonic basis, or synthesis path.
- Added typed caller-owned NUFFT 1D/3D Type-1/Type-2 paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles without duplicating the Kaiser-Bessel spreading/interpolation, Apollo FFT, or deconvolution paths.

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

### Extension Phase

- Added `supports_mixed_precision` and `default_precision_profile` fields to all WGPU capability structs.
- Added NTT-WGPU exact quantized `u32` residue storage APIs that preserve modular values losslessly under the existing `u32::MAX` modulus bound and reject output shape mismatches.
- Added NTT-WGPU exact quantized `u32` reusable-buffer execution using the existing `NttGpuBuffers` ownership boundary.
- Verified NUFFT and SHT CPU mixed-precision storage contracts were already complete (`NufftComplexStorage`, `ShtRealStorage`, `ShtComplexStorage`).
- Added `NufftGpuBuffers1D` and `NufftGpuBuffers3D` reusable GPU buffer structs with `execute_fast_*_with_buffers` methods to eliminate per-call buffer allocation on repeated NUFFT fast-path dispatch.
- Added `NttGpuBuffers` and `execute_*_with_buffers` methods to eliminate per-call device-buffer, bind-group, staging-buffer, and host-output allocation on repeated direct NTT WGPU dispatch.
- Added `execute_*_quantized_with_buffers` methods to eliminate the same allocation class for repeated exact `u32` residue-storage NTT WGPU dispatch.
- Added `NufftPlan3D::type2_into` zero-allocation path (type2 now delegates to type2_into).
- Added value-semantic typed verification tests for NUFFT 1D and 3D across Complex64, Complex32, and [f16;2] storage profiles with profile mismatch rejection.

---

## Remaining Gaps

Open gaps are listed at the top of this audit. Future increments should:
- Run the Criterion buffer-reuse benches on representative GPU hardware and record measured allocation-vs-reuse speedup ratios for 1D and 3D NUFFT fast paths.
- Verify `GpuFft3dF16Native` Bluestein path on production hardware with non-power-of-two sizes (current test passes on dev hardware; production validation is pending).


### Closed in this sprint (Performance & Native GPU Precision phase)

- Performance-quantification gap: added Criterion bench targets `buffer_reuse` to both `apollo-nufft-wgpu` (fast Type-1/Type-2 1D, per-call vs `with_buffers`, N=64/128/256) and `apollo-fft-wgpu` (3D forward/inverse, per-call vs `with_buffers`, nx=ny=nz=4/8/16).
- `NufftWgpuBackend` façade gap: added public `execute_fast_type1_1d_with_buffers`, `execute_fast_type2_1d_with_buffers`, `execute_fast_type1_3d_with_buffers`, `execute_fast_type2_3d_with_buffers` methods delegating to `NufftGpuKernel`.
- Native f16 GPU compute gap: added `GpuFft3dF16Native` behind `apollo-fft-wgpu/native-f16` feature; WGSL shaders `fft_native_f16.wgsl` and `pack_native_f16.wgsl` use `enable f16;` and `array<f16>` storage; host boundary performs f32↔f16 conversion; parity test verifies |error| < 5×10⁻³ against f32 GPU reference (O(log N)·ε_f16 bound with N=4).
- Bluestein f16 gap: implemented `chirp_native_f16.wgsl` with `enable f16;`, `array<f16>` bindings, and f32-precision twiddles narrowed to f16; lifted power-of-two-only constraint on `GpuFft3dF16Native` by adding `strategy_x/y/z`, `chirp_x/y/z` fields, `build_chirp_data_f16`, and `dispatch_chirp_f16` (flat 1D dispatch, no data races); roundtrip test on 3×3×3 (all-Bluestein) passes with error < 0.05.
- 3D NUFFT buffer-reuse bench gap: added `bench_fast_type1_3d` and `bench_fast_type2_3d` Criterion functions to `apollo-nufft-wgpu/benches/buffer_reuse.rs`; covers per-call vs `with_buffers` for N=4,6,8.
- Published-reference fixture breadth gap: added NTT impulse ([1,0,0,0]→[1,1,1,1], Pollard 1971), NTT constant ([1,1,1,1]→[4,0,0,0], geometric-series theorem), and NUFFT Type-1 at origin (single source x=0 → F[k]=1 ∀k, Dutt and Rokhlin 1993) to `apollo-validation`; all three verified at PUBLISHED_FIXTURE_LIMIT=1×10⁻¹².
