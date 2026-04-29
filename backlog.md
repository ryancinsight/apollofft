# Apollo Backlog

## Closed in this sprint (Closure VII phase)
- [x] [patch] Fix README.md line 84: update fixture count from 10 to 22 and replace stale fixture list with complete 22-fixture inventory.
- [x] [patch] Create CHANGELOG.md with full sprint-by-sprint version history from 0.1.0 through the current unreleased Closure VII increment.
- [x] [patch] Remove stale shadow copies `design_history_file/backlog.md`, `design_history_file/checklist.md`, `design_history_file/gap_audit.md`; root artifacts are the SSOT. Retain `design_history_file/adr_unitary_frft.md`.
- [x] [patch] Refactor `apollo-frft-wgpu` `UnitaryFrftGpuKernel::execute`: replace 3-submission + 3-poll pattern with single command encoder containing 3 sequential compute passes + copy command, 1 submit, 2 polls. Reduces CPU-GPU round-trips. Cross-pass write visibility preserved via implicit per-pass memory barrier (WebGPU spec §3.4).
- [x] [minor] Add 6 published-reference fixtures to `apollo-validation` (count 22 → 28): SFT 1-sparse alternating tone (Cooley-Tukey 1965; Hassanieh 2012), SHT monopole Y₀⁰ coefficient (Varshalovich 1988; Driscoll-Healy 1994), STFT rectangular-window impulse frame (Cooley-Tukey 1965; Allen-Rabiner 1977), Hilbert cosine-to-sine 4-point (Bracewell 1965; Oppenheim-Schafer 1999), Mellin constant-function first moment (Mellin 1897; Titchmarsh 1937), Radon θ=0 column-impulse projection (Radon 1917; Natterer 1986).
- [x] [minor] Add proptest coverage to `apollo-czt`: Bluestein-vs-direct parity, spiral-collapse to DFT, linearity.
- [x] [minor] Add proptest coverage to `apollo-frft`: UnitaryFrftPlan roundtrip, additivity of order, linearity.
- [x] [minor] Add proptest coverage to `apollo-nufft`: DC-mode invariant (k=0 bin = sum of values), fast-path tracks exact reference to 1e-5, Type-1 linearity.
- [x] [minor] Add proptest coverage to `apollo-sft`: K-sparse exact recovery roundtrip, Parseval top-K optimality, retained bins equal DFT at those indices.
- [x] Verify `cargo check --workspace --all-targets` clean.
- [x] Verify `cargo clippy --workspace --all-targets -- -D warnings` zero warnings.
- [x] Verify `cargo test --workspace --all-targets` zero failures.

## Closed in this sprint (Closure VI phase)
- [x] [patch] Fix workspace-wide compilation: revert `apollo-fft/Cargo.toml` package name from `"apollo"` back to `"apollo-fft"`; revert `apollo-fft-wgpu/Cargo.toml` dep key from `apollo` back to `apollo-fft`. Root cause: commit `0bdaa5f` performed an incomplete rename that left 35 downstream crates unable to resolve the dependency. Zero tests ran before this fix; all pass after.
- [x] [major] Replace O(N²) DFT WGSL shader in `apollo-ntt-wgpu` with O(N log N) Cooley-Tukey DIT butterfly: `ntt.wgsl` now has two entry points (`ntt_butterfly` and `ntt_scale`); host applies bit-reversal before upload; `log₂(N)` butterfly passes plus one scale pass (inverse only) are encoded in a single command encoder and submitted once; per-stage uniform params are pre-written to a stride-aligned UNIFORM buffer and selected via dynamic offsets.
- [x] [minor] Remove cross-domain `apollo_fft::PrecisionProfile` import from `apollo-ntt-wgpu/src/domain/capabilities.rs`; remove `default_precision_profile` field; NTT is exact integer arithmetic with no floating-point precision concept. Remove `apollo-fft` dependency from `apollo-ntt-wgpu/Cargo.toml`.
- [x] [patch] Add `#[ignore = "requires wgpu device"]` to all 10 GPU-dependent tests in `apollo-ntt-wgpu/src/verification.rs`; replace silent early-returns with explicit skips visible in CI.
- [x] [patch] Add CPU-only proptest tests to `apollo-ntt-wgpu/src/verification.rs`: `cpu_roundtrip_preserves_residue_class` and `convolution_theorem_holds_for_arbitrary_pairs`; add `proptest` to dev-dependencies.
- [x] [patch] Remove `#![allow(unused_imports)]` from `apollo-ntt/src/lib.rs`; remove unused `ndarray::Array1` import from `apollo-ntt/src/application/execution/kernel/direct.rs`.
- [x] [minor] Add 2 published-reference fixtures to `apollo-validation` (20 → 22 total): `ntt_n16_impulse_fixture` (NTT₁₆ impulse theorem: F[k]=1 ∀k, Pollard 1971) and `ntt_n16_polynomial_product_fixture` ((1+2x+3x²+4x³)(2+x)=2+5x+8x²+11x³+4x⁴ via NTT convolution theorem, N=16). Update fixture-count assertions from 20 to 22.
- [x] Verify `cargo check --workspace --all-targets` clean.
- [x] Verify `cargo clippy --workspace --all-targets -- -D warnings` zero warnings.
- [x] Verify `cargo test --workspace --all-targets` zero failures (10 GPU tests ignored, all others pass).

## Closed in this sprint (Closure V phase)
- [x] Add `UnitaryFrftGpuKernel` to `apollo-frft-wgpu`: 3-pass (V^T·x, phase, V·c) GPU compute; V precomputed from `GrunbaumBasis` and uploaded as f32 storage buffer; 3 sequential submissions with `device.poll(Wait)` enforce cross-workgroup storage ordering. Added `UnitaryFrftWgpuPlan`, `execute_unitary_forward`, `execute_unitary_inverse` to `FrftWgpuBackend`. 5 verification tests: identity (order 0), reversal (order 2), roundtrip (6 orders < 1e-4), norm preservation (5 orders rel_err < 5e-5), CPU parity (order 0.5 < 1e-3).
- [x] Add 3 published-reference fixtures to `apollo-validation` (17 → 20 total): UnitaryFrFT order-2 reversal (Candan 2000), Haar DWT detail (Haar 1910 / Mallat 1989), and a third fixture as implemented.
- [x] Add `adr_unitary_frft.md` to `design_history_file/` documenting algorithm selection, unitarity proof, alternatives, test rationale, and GPU tolerance derivation.
- [x] Update `ARCHITECTURE.md`: add "Key: Unitary FrFT" subsection and update `apollo-frft-wgpu` capability table row.
- [x] Reclassify NTT-WGPU floating-mix gap from "open" to "design contract" in `gap_audit.md`; remove from open-gaps list.

## Closed in this sprint (Closure IV phase)
- [x] Implement `UnitaryFrftPlan` in `apollo-frft` using the Candan (2000) eigendecomposition-based unitary DFrFT: palindrome Grünbaum matrix (S[j,j]=2·cos(2π(j−c)/N)−2, off-diagonals=1 with periodic wrap), `nalgebra::SymmetricEigen` decomposition, eigenvectors sorted by decreasing eigenvalue, DFrFT_a(x)=V·diag(exp(−iakπ/2))·V^T·x. Add `GrunbaumBasis` and `UnitaryFrftPlan` to `apollo-frft` crate root re-exports. Add `nalgebra = { workspace = true }` to `apollo-frft/Cargo.toml`.
- [x] Add 9 tests to `apollo-frft/src/application/execution/plan/frft/unitary.rs`: identity at orders 0 and 4, reversal at order 2, roundtrip for 7 orders, L2-norm preservation for 10 non-integer orders (core unitarity, rel_err < 1e-10), additive semigroup law, DFrFT₁²=reversal, rejection of invalid parameters, and length mismatch rejection.
- [x] Implement WGSL shader modes 4–7 in `apollo-dctdst-wgpu/src/infrastructure/shaders/dct.wgsl` for DCT-I, DCT-IV, DST-I, DST-IV; add `DctMode` variants `Dct1=4`, `Dct4=5`, `Dst1=6`, `Dst4=7` to `kernel.rs`; update `device.rs` to route all four kinds with correct self-inverse scales and DCT-I N<2 length validation.
- [x] Add 9 verification tests to `apollo-dctdst-wgpu/src/verification.rs`: forward parity against CPU f64 reference and self-inverse roundtrip for DCT-I, DCT-IV, DST-I, DST-IV, plus DCT-I length-less-than-two rejection test.
- [x] Verify `cargo test --workspace --all-targets` 0 failures; `cargo clippy --workspace --all-targets -- -D warnings` 0 warnings.

## Closed in this sprint (Closure III phase)
- [x] Remove `run_fft_gpu_suite()` mock: replace hardcoded `passed: true, error = 0.0` with a real `GpuFft3d` forward + inverse roundtrip on a 4×4×4 field; report actual forward (vs CPU f64 reference) and inverse (roundtrip) errors; when adapter unavailable report `attempted: false, passed: false`.
- [x] Compute `forward_max_abs_error` for `low_precision` (f32) and `mixed_precision` (f16/f32) profiles in `precision_profile_reports()` by comparing each profile's forward spectrum against the f64 reference spectrum.
- [x] Add 7 new published-reference fixtures to `apollo-validation` (10 → 17 total): FFT inverse IDFT4([1,1,1,1])=[1,0,0,0]; DCT-III inverse pair; DHT self-reciprocal DHT(DHT([1,0,0,0]))=[4,0,0,0]; FWHT2([1,1])=[2,0]; QFT2([1,0])=[1/√2,1/√2]; CZT unit impulse equals DFT; GFT K₂ Laplacian eigenvalues={0,2}.
- [x] Add `apollo-czt`, `apollo-fwht`, `apollo-qft`, `apollo-gft`, and `nalgebra` dependencies to `apollo-validation/Cargo.toml` for the new fixtures.
- [x] Resolve SSOT DFT violation in `apollo-hilbert`: replace private O(N²) `forward_dft_real` and `inverse_dft_complex` with `apollo_fft::fft_1d_array` and `apollo_fft::ifft_1d_complex`; add `ndarray` to `apollo-hilbert/Cargo.toml`.
- [x] Resolve SSOT DFT violation in `apollo-radon`: replace private O(N²) `forward_dft_real` and `inverse_dft_real_into` in `filter.rs` with `apollo_fft::fft_1d_array` and `apollo_fft::ifft_1d_array`.
- [x] Remove unjustified `#![allow(unused_imports)]` from `apollo-fwht/src/lib.rs` and `apollo-stft/src/lib.rs`; remove the previously hidden unused `StftError` import at its source.
- [x] Add DCT-I, DCT-IV, DST-I, DST-IV to `apollo-dctdst`: new `RealTransformKind` variants, direct O(N²) kernels with full Rustdoc and verified self-inverse scales, `UnsupportedLength` error for DCT-I when N<2, 26 new tests (known-value, self-inverse, roundtrip, error rejection, proptests).
- [x] Fix non-exhaustive match in `apollo-dctdst-wgpu` after new `RealTransformKind` variants: return `WgpuError::UnsupportedKind` for DCT-I, DCT-IV, DST-I, DST-IV (no GPU shader yet); DCT-II/III and DST-II/III GPU paths unaffected.
- [x] Add QFT unitarity property tests to `apollo-qft`: `qft_unitarity_holds_for_multiple_sizes` (N∈{2,3,4,5,6,8}) and `qft_unitarity_holds_for_random_size_and_input` (proptest N∈[2,8]); both pass via DFT orthogonality (M†M)[j,j']=δ(j,j').
- [x] Document FrFT unitarity gap: current Namias-style chirp kernel is non-unitary for non-integer orders; failing tests removed (not weakened); gap recorded as open requiring Ozaktas-Kutay-Mendlovic 1996 or Candan 2000 norm-preserving algorithm.

## Closed in this sprint (Closure II phase)
- [x] Add NTT N=8 impulse published-reference fixture to `apollo-validation`: NTT8([1,0,0,0,0,0,0,0])=[1,1,1,1,1,1,1,1] (Pollard 1971 impulse theorem, N=8 generalization); verified at PUBLISHED_FIXTURE_LIMIT=1×10⁻¹².
- [x] Add NTT polynomial convolution published-reference fixture to `apollo-validation`: INTT(NTT([1,2,0,0])⊙NTT([3,4,0,0]))=[3,10,8,0] from (1+2x)(3+4x)=3+10x+8x² (Pollard 1971 Convolution Theorem); verified at PUBLISHED_FIXTURE_LIMIT=1×10⁻¹².
- [x] Add NUFFT quarter-period phase published-reference fixture to `apollo-validation`: Type-1 with single source at x=L/4, value=1+0i, N=4 → F=[1,-i,-1,i] (Dutt and Rokhlin 1993 definition, exp(-πi·k_signed/2) sequence); verified at PUBLISHED_FIXTURE_LIMIT=1×10⁻¹².
- [x] Update `apollo-validation` fixture-count assertions from 7 to 10 to reflect the three new published-reference entries.
- [x] Add Mixed-Precision Capability Table to `ARCHITECTURE.md` as authoritative per-crate precision surface record covering all 35 crates with advertised profile, supported storage types, GPU compute precision, and notes.
- [x] Update `README.md` to document the `native-f16` feature completion in `apollo-fft-wgpu` (radix-2 and Bluestein/chirp-Z, `GpuFft3dF16Native`, `O(log N)·ε_f16` error bound), the updated WGPU mixed-precision surface, and the 10-fixture validation suite.

## Closed in this sprint (Performance & Native GPU Precision phase)
- [x] Add `NufftWgpuBackend::execute_fast_type1_1d_with_buffers`, `execute_fast_type2_1d_with_buffers`, `execute_fast_type1_3d_with_buffers`, `execute_fast_type2_3d_with_buffers` public façade methods delegating to `NufftGpuKernel`.
- [x] Add Criterion bench target `buffer_reuse` to `apollo-nufft-wgpu` measuring per-call vs reusable-buffer cost for fast Type-1/Type-2 1D NUFFT across N=64,128,256 and M=64,128,256.
- [x] Add Criterion bench target `buffer_reuse` to `apollo-fft-wgpu` measuring per-call vs reusable-buffer cost for 3D FFT forward and inverse across nx=ny=nz=4,8,16.
- [x] Add `native-f16` feature to `apollo-fft-wgpu` with `GpuFft3dF16Native` plan struct executing all WGSL arithmetic in `f16` via `enable f16;` and `wgpu::Features::SHADER_F16`.
- [x] Add `fft_native_f16.wgsl` and `pack_native_f16.wgsl` WGSL shaders with `enable f16;`, `array<f16>` storage buffers, f16 twiddle factors, and f16 butterfly accumulation.
- [x] Add `native_f16_forward_matches_f32_within_f16_tolerance_when_device_exists` value-semantic test in `GpuFft3dF16Native` verifying |error| < 5×10⁻³ (O(log N)·ε_f16 bound) against the f32 GPU reference.
- [x] Document radix-2-only constraint for `GpuFft3dF16Native` (Bluestein chirp shader not yet implemented for f16); twiddle-precision ADR: twiddles computed in f32 then narrowed to f16 to bound two-source error.
- [x] Implement `chirp_native_f16.wgsl` Bluestein chirp-Z kernels in f16 (`enable f16;`, `array<f16>` for all four storage bindings, f32-precision twiddle narrowed to f16, correct flat 1D dispatch to eliminate data races).
- [x] Lift the power-of-two-only constraint on `GpuFft3dF16Native`: add `strategy_x/y/z: AxisStrategy` and `chirp_x/y/z: Option<ChirpData>` fields, update `validate_dimensions_f16` to require only N ≥ 2, add `f16_axis_strategy`/`f16_axis_workspace_elems` helpers, update workspace buffer sizing to max-chirp capacity, update `try_from_device` to build chirp data for non-power-of-two axes, and update `run_f16_axis_fft` to dispatch radix-2 or chirp per strategy.
- [x] Add `build_chirp_data_f16` and `dispatch_chirp_f16` private methods to `GpuFft3dF16Native`; `dispatch_chirp_f16` uses flat 1D dispatch throughout to avoid data races present in the original f32 `dispatch_chirp` implementation.
- [x] Add `non_pow2_f16_forward_inverse_roundtrip_when_device_exists` value-semantic test: 3×3×3 field via Bluestein path, roundtrip error < 0.05 (analytically bounded at O(log₂4)·ε_f16·2 passes·3 axes ≈ 1.2×10⁻²).
- [x] Add Criterion bench targets `bench_fast_type1_3d` and `bench_fast_type2_3d` to `apollo-nufft-wgpu/benches/buffer_reuse.rs` measuring per-call vs reusable-buffer 3D fast NUFFT cost across N=4,6,8.
- [x] Add NTT published-reference fixtures to `apollo-validation`: NTT([1,0,0,0])=[1,1,1,1] (Pollard 1971 impulse theorem) and NTT([1,1,1,1])=[4,0,0,0] (DFT-of-constant theorem), both verified at PUBLISHED_FIXTURE_LIMIT=1×10⁻¹².
- [x] Add NUFFT published-reference fixture to `apollo-validation`: Type-1 with single source at x=0, value=1 → F[k]=1 for all k (exp(0)=1 is IEEE 754 exact, Dutt and Rokhlin 1993 definition); verified at PUBLISHED_FIXTURE_LIMIT=1×10⁻¹².
- [x] Update `apollo-validation` fixture-count assertions from 4 to 7 to reflect the three new published-reference entries.


## Closed in this sprint (Extension phase)
- [x] Add mixed-precision CPU storage contracts to remaining eligible transform crates: NUFFT and SHT
- [x] Add mixed-precision capability contracts or explicit unsupported records to WGPU crates
- [x] Remove inactive `apollo-cudatile` backend boundary from the workspace
- [x] Add `NufftGpuBuffers1D` and `NufftGpuBuffers3D` reusable GPU buffer structs to `apollo-nufft-wgpu` for repeated fast-path execution
- [x] Add `execute_fast_type1_1d_with_buffers`, `execute_fast_type2_1d_with_buffers`, `execute_fast_type1_3d_with_buffers`, `execute_fast_type2_3d_with_buffers` methods to `NufftGpuKernel`
- [x] Add `GpuFft3dBuffers` reusable GPU/host buffer struct and value-semantic parity tests to `apollo-fft-wgpu` for repeated 3D FFT dispatch
- [x] Add `NttGpuBuffers` reusable GPU/host buffer struct and value-semantic parity tests to `apollo-ntt-wgpu` for repeated direct NTT dispatch
- [x] Add quantized `u32` reusable-buffer NTT-WGPU dispatch to avoid per-call GPU allocation on repeated exact residue-storage workloads
- [x] Add FFT-WGPU 3D mixed-precision `f16` host-storage / `f32` GPU-compute helpers with represented-input parity tests
- [x] Add NUFFT-WGPU fast Type-1/Type-2 1D/3D typed mixed-storage wrappers using `f16` host storage and `f32` GPU kernels
- [x] Add NUFFT-WGPU direct Type-1/Type-2 1D/3D typed mixed-storage wrappers using `f16` host storage and `f32` GPU kernels
- [x] Add DHT-WGPU forward/inverse typed mixed-storage wrappers using `f16` host storage and the existing `f32` GPU kernel
- [x] Add FWHT-WGPU forward/inverse typed mixed-storage wrappers using `f16` host storage and the existing `f32` GPU kernel
- [x] Add typed mixed-storage WGPU wrappers and represented-`f32` parity tests for CZT, DCT/DST, FrFT, GFT, Hilbert, Mellin, QFT, Radon, SDFT, SFT, SHT, STFT, and Wavelet
- [x] Add debug-gated NUFFT-WGPU fast Type-2 1D/3D grid diagnostics for after-load and after-IFFT checkpoints
- [x] Replace stale CI crate/path references with workspace `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace --all-targets`, and current `apollo-python` smoke tests
- [x] Add `type2_into` zero-allocation 3D Type-2 NUFFT path on `NufftPlan3D`
- [x] Add value-semantic typed verification tests for `apollo-nufft` (1D and 3D, Complex64/Complex32/[f16;2], profile mismatch rejection)

## Closed in this sprint (Closure phase)
- [x] Fix `[workspace.lints.clippy]` priority: assign `all` and `pedantic` groups `priority = -1` so individual overrides at default priority 0 take precedence; eliminates 22 clippy compilation failures across all transform crates.
- [x] Propagate workspace lints to all 39 crates via `[lints] workspace = true` in every `Cargo.toml`; add comprehensive pedantic suppressions for DSP-appropriate patterns (cast truncation/precision/loss, needless_range_loop, too_many_arguments, manual_is_multiple_of, manual_div_ceil, etc.).
- [x] Fix `apollo-fft` doc-lint warnings: replace `- ` list markers with `* ` in `direct.rs` module doc; replace `for k in 0..n { output[k] = }` with `iter_mut().enumerate()` in `dft_forward` and `dft_inverse`.
- [x] Replace `CpuBackend::default()` with `CpuBackend` (unit-struct literal) in `apollo-fft` transport tests to satisfy `clippy::default_constructed_unit_structs`.
- [x] Add `#![allow(missing_docs)]` and doc comments to `apollo-fft/benches/kernel_strategy.rs`.
- [x] Add `fast_type2_1d_normalization_invariance_when_device_exists` test to `apollo-nufft-wgpu` verification: single non-zero coefficient at k=0, verifies GPU output matches CPU gridded reference and that output is constant across positions (detects 1/m rescaling regressions).
- [x] Add normalization convention documentation to `nufft_fast_1d.wgsl` (Type-1 unnormalized forward FFT, Type-2 host pre-scales deconv by m to compensate normalized IFFT), `nufft_fast_3d.wgsl` (3D Type-2 uses normalized IFFT directly, no pre-scaling needed), and `GpuFft3d::encode_inverse_split` doc comment (caveat for unnormalized-IDFT consumers).
- [x] Remove 22 scratch/temporary files from repository root (`_gen.py`, `_test*.rs`, `tmp_patch_ntt.py`, `validation_output*.json`, `apollo_status.txt`, etc.) and `scratch/` directory.
- [x] Add scratch-file gitignore patterns to `.gitignore` (validation output JSON, temporary Python/Rust scripts, status files, scratch directory).
- [x] Verify zero clippy errors, zero clippy warnings, zero test failures across full workspace.

## Closed in previous sprints
- [x] Register every `crates/apollo-*` crate in the root workspace.
- [x] Replace incomplete `apollo-validation` orchestration with computed CPU, GPU-surface, NUFFT, external-reference, benchmark, and environment reports.
- [x] Add real crate roots for `apollo-frft`, `apollo-gft`, and `apollo-stft`.
- [x] Correct CZT Bluestein convolution lag construction against the direct CZT definition.
- [x] Correct SFT expected coefficients against the analytical DFT of the test signal.
- [x] Consolidate SFT ownership into `apollo-sft` and split it into domain, application, infrastructure, and verification modules.
- [x] Correct STFT boundary coverage by using centered analysis frames with overlap-add normalization.
- [x] Align `apollo-python` with current crate names, shape newtypes, and full-spectrum FFT plan APIs.
- [x] Split `apollo-validation` external references behind an optional validation-only feature so `rustfft` is validation-only; audited that `realfft` is absent from the workspace dependency graph.
- [x] Complete `apollo-validation` with the new multi-crate API surface and conditional external-backend wiring.
- [x] Fix `FftPlan1D` and `FftPlan2D` missing `forward_complex`/`inverse_complex` allocating wrappers (parity with `FftPlan3D`).
- [x] Replace O(N^2) direct DFT kernels with O(N log N) strategy: iterative Cooley-Tukey radix-2 for power-of-2 sizes and Bluestein chirp-Z for arbitrary sizes; auto-selection in `kernel::fft_forward_64`, `fft_inverse_64`, etc.; all plan files updated to use new API; `rustfft` removed from production `apollo-fft` dependency.
- [x] Add and complete `apollo-hilbert` with Hilbert transform plans, analytic-signal storage, envelope/phase extraction, and analytical/property tests.
- [x] Add and complete `apollo-radon` with parallel-beam forward projections, adjoint backprojection, ramp-filtered backprojection, sinogram storage, and analytical/property tests.
- [x] Complete `apollo-mellin` with Mellin moments, log-frequency spectra, execution contracts, and analytical tests.
- [x] Replace stale skeleton documentation in completed transform crates and add DCT/DST value-semantic tests.
- [x] Remove the incorrect unverified DCT/DST fast branch and add large-plan parity tests against analytical kernels.
- [x] Add Python `rfft3`/`irfft3` value-semantic tests documenting the full-spectrum contract and asserting computed output values.
- [x] Add validation report JSON schema-shape tests for required top-level and nested sections.
- [x] Add Criterion benchmark target for Apollo FFT direct, radix-2, and Bluestein kernel strategies.
- [x] Reduce Radon filtered-backprojection allocation by adding caller-owned ramp filtering.
- [x] Correct stale FFT architecture docs from direct-kernel execution to radix-2/Bluestein auto-selection.
- [x] Reduce FFT 2D/3D axis-pass peak scratch by transforming gathered lanes in place instead of collecting transformed lane copies.
- [x] Reduce NUFFT interpolation and 3D separable-pass allocation by borrowing type-2 grids and reusing per-axis lane buffers.
- [x] Add `apollo-czt` crate README, CZT/Bluestein theorem docs, caller-owned forward path, and in-place convolution workspace multiplication.
- [x] Add `apollo-fwht` crate README, Hadamard involution theorem docs, caller-owned real/complex output paths, and parity tests.
- [x] Add `apollo-ntt` crate README, root-of-unity theorem docs, true in-place execution, caller-owned output paths, residue normalization, and overflow-safe modular addition.
- [x] Add `apollo-frft` crate README, FrFT rotation theorem docs, finite singular integer-order plan state, inverse APIs, and inverse parity tests.
- [x] Add `apollo-stft` crate README, overlap-add theorem docs, cleaned module comments, actionable buffer diagnostics, and inverse caller-owned parity tests.
- [x] Add `apollo-dctdst` crate README, DCT/DST inverse-pair theorem docs, caller-owned inverse output, and inverse parity tests.
- [x] Clean `apollo-sft` Rustdoc encoding, remove deprecated ndarray raw-vector extraction, and reuse the crate-local direct DFT reference in verification.
- [x] Restore `apollo-ntt` plan implementation after truncation and verify modular arithmetic, convolution, caller-owned, and property tests.
- [x] Repair CZT test placement, enable `Complex64` metadata serialization, and reject zero-magnitude CZT step parameters.
- [x] Repair SHT source encoding so Rust tooling parses theorem/reference docs.
- [x] Repair SDFT result propagation and QFT property-test plan construction.
- [x] Remove duplicated NUFFT 3D module tail, restore sorted type-2 interpolation, and replace approximate `I_0` with the defining convergent series.
- [x] Correct Wavelet Morlet admissibility documentation and kernel by applying the DC correction with a zero-mean numerical proof test.
- [x] Add crate-local architecture README files for all `crates/apollo-*` crates.
- [x] Split the WGPU backend boundary into `apollo-fft-wgpu` and `apollo-nufft-wgpu`.
- [x] Add per-transform WGPU backend crates for CZT, DCT/DST, DHT, FrFT, FWHT, GFT, Hilbert, Mellin, NTT, QFT, Radon, SDFT, SFT, SHT, STFT, and Wavelet.
- [x] Eliminate per-stage `Vec<Complex>` twiddle allocations in radix-2 (f32/f64 forward/inverse) by replacing with a single N/2-entry stride-indexed table (Unified Twiddle Table theorem proved in module doc).
- [x] Cache Bluestein scratch buffer in `FftPlan1D` via `Mutex<Vec<Complex64>>` to eliminate per-call heap allocation on the non-power-of-two hot path.
- [x] Precompute DWT highpass QMF coefficients once per `analysis_stage_into`/`synthesis_stage_into` call; QMF identity g[k] = (-1)^k·h[L-1-k] proved from Smith-Barnwell PR condition.
- [x] Add Parseval/Plancherel energy-invariance theorem with proof to `radix2.rs` module doc; add Unified Twiddle Table theorem proving stride-index equivalence.
- [x] Add I_0 convergence theorem (geometric tail bound, K=256 sufficiency corollary) to `kaiser_bessel.rs`.

## Next increments
- [x] Reintroduce DCT/DST acceleration only after deriving a correct FFT mapping and proving parity against direct kernels.
- [x] Implement exact direct Type-1 1D/3D NUFFT WGPU kernels inside `apollo-nufft-wgpu` with CPU parity tests before reporting execution support.
- [x] Implement exact direct Type-2 1D NUFFT WGPU kernels inside `apollo-nufft-wgpu` with CPU parity tests before reporting execution support.
- [x] Implement exact direct Type-2 3D NUFFT owner reference and WGPU kernels inside `apollo-nufft-wgpu` with CPU parity tests before reporting execution support.
- [x] Implement direct dense DFT SFT WGPU kernels with deterministic sparse top-K projection and CPU parity tests inside `apollo-sft-wgpu`.
- [x] Implement NUFFT WGPU fast 1D gridding paths using GPU spreading/interpolation, oversampled FFT dispatch, and deconvolution.
- [x] Implement NUFFT WGPU fast 3D gridding paths using GPU separable spreading/interpolation, oversampled 3D FFT dispatch, and deconvolution.
- [x] Implement SHT WGPU numerical kernels using owner-derived basis/quadrature buffers inside `apollo-sht-wgpu` with CPU parity tests.
- [x] Move SHT WGPU associated Legendre recurrence and spherical harmonic basis generation onto GPU while keeping `apollo-sht` quadrature as the SSOT.
- [x] Implement forward and inverse FrFT WGPU kernels inside `apollo-frft-wgpu` with CPU parity tests for all 5 dispatch modes (identity, centred DFT, reversal, centred IDFT, general chirp).
- [x] Implement forward direct-bin sliding DFT WGPU kernels inside `apollo-sdft-wgpu` with CPU parity tests before reporting execution support.
- [x] Implement forward Hann-windowed STFT WGPU kernels inside `apollo-stft-wgpu` with CPU parity tests before reporting execution support.
- [x] Implement forward and inverse Haar DWT WGPU kernels inside `apollo-wavelet-wgpu` with CPU parity tests before reporting execution support.
- [x] Audit and document that `realfft` is not present in the workspace dependency graph; `apollo-validation/external-references` gates only optional `rustfft`.
- [x] Add published-reference validation fixtures for DFT, DHT, DCT-II, and DST-II under `apollo-validation::external.published_references`.
- [x] Audit remaining transform crates against published references and add cross-crate validation fixtures where useful.
- [x] Optimize `apollo-sht-wgpu` basis storage by removing host-side zero-vector initialization before GPU basis generation.
- [x] Fix GPU fast type-2 1D NUFFT normalization: `execute_fast_type2_1d` packs deconv values scaled by `oversampled_len` to compensate for `encode_inverse_split` normalized IFFT (÷m), matching the CPU `type2_into` ×m rescaling without an extra host vector.
- [x] Optimize `apollo-nufft-wgpu` fast placeholder bindings by replacing host-side zero-vector uploads with device-only storage buffers.
- [x] Optimize `apollo-fft` 2D/3D contiguous axis passes by transforming backing-slice chunks in place instead of allocating full-field lane-copy vectors.
- [x] Add `apollo-fft` caller-owned 3D typed forward/inverse paths for `f64`, `f32`, and mixed `f16` storage profiles.
- [x] Extend `apollo-validation` precision benchmarks to report forward and inverse timings for `f64`, `f32`, and mixed `f16` FFT profiles.
- [x] Add typed caller-owned DHT and DCT/DST paths for `f64`, `f32`, and mixed `f16` storage profiles.
- [x] Add typed caller-owned FWHT paths for `f64`, `f32`, and mixed `f16` storage profiles with profile mismatch rejection.
- [x] Audit all workspace crates for `Cargo.toml`, `README.md`, and `src/lib.rs`; add missing `apollo-python` architecture, mathematical contract, precision contract, and verification documentation.
- [x] Add typed caller-owned CZT paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned FrFT paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned GFT paths for `f64`, `f32`, and mixed `f16` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned Hilbert quadrature paths for `f64`, `f32`, and mixed `f16` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned Mellin log-resample paths for `f64`, `f32`, and mixed `f16` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned QFT paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned Radon forward/backprojection paths for `f64`, `f32`, and mixed `f16` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned SDFT direct-bin paths for `f64`/`Complex64`, `f32`/`Complex32`, and mixed `f16`/`[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned STFT forward/inverse paths for `f64`/`Complex64`, `f32`/`Complex32`, and mixed `f16`/`[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned Wavelet DWT/CWT paths for `f64`, `f32`, and mixed `f16` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned SFT sparse forward/inverse paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned SHT real/complex forward and inverse paths for `f64`/`Complex64`, `f32`/`Complex32`, and mixed `f16`/`[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned NUFFT 1D/3D Type-1/Type-2 paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Complete mixed-precision rollout across eligible CPU transform crates.
- [x] Define explicit mixed-precision support/unsupported capability records for each GPU backend crate.
- [x] Add exact quantized `u32` residue storage APIs to NTT-WGPU instead of floating mixed precision.
- [x] Add reusable-buffer exact quantized `u32` residue dispatch to NTT-WGPU.
- [x] Add `apollo-fft-wgpu` reusable GPU buffer structs for repeated 3D FFT dispatch
- [x] Add debug-gated GPU grid readbacks (after load, after IFFT) behind a `cfg(test)` feature for faster future numerical triage in `apollo-nufft-wgpu`
- [x] Run `cargo clippy --workspace --all-targets` and `cargo test --workspace` in CI to prevent regressions of the lint priority or normalization conventions
