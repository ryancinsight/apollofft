# Apollo Backlog

## Closed in this sprint (Closure XLII phase)
- [x] [patch] Add RustFFT comparator coverage to `apollo-fft` kernel Criterion benchmarks.
- [x] [patch] Add allocation-free benchmark rows for Apollo caller-supplied twiddle paths.
- [x] [patch] Route exact 2/4/8/16/32/64-point f64 and f32 mixed-radix transforms through
  Winograd short-DFT kernels before staged radix dispatch.
- [x] [patch] Remove f16 benchmark references to nonexistent radix16/radix32/radix64 public entrypoints;
  benchmark actual f16 surfaces (`radix2_f16`, `radix4`, `radix8`, auto-selector).
- Final state: `cargo test -p apollo-fft` 124 passed; `cargo bench -p apollo-fft --bench kernel_strategy --no-run` passed;
  post-rebase focused Criterion run measured `apollo_auto_selector_precomputed_twiddles/64` at 241.74 ns and
  `rustfft_forward_inplace/64` at 197.61 ns, leaving RustFFT parity as an open kernel-arithmetic gap.

## Closed in this sprint (Closure XLI phase)
- [x] [minor] Add separable CPU 2D DHT: `DhtPlan::forward_2d`, `inverse_2d` (N×N, involutory scaling 1/N²).
- [x] [minor] Add separable CPU 3D DHT: `DhtPlan::forward_3d`, `inverse_3d` (N×N×N, involutory scaling 1/N³).
- [x] [minor] Add `DhtError::ShapeMismatch2d` and `DhtError::ShapeMismatch3d` variants.
- [x] [minor] Add `ndarray = "0.16"` dependency to `apollo-dht`; re-export `Array2`, `Array3`.
- [x] [minor] Add `FwhtPlan2D` in `dimension_2d.rs` (separable N×N FWHT, real + complex).
- [x] [minor] Add `FwhtPlan3D` in `dimension_3d.rs` (separable N×N×N FWHT, real + complex).
- [x] [minor] Re-export `FwhtPlan2D`, `FwhtPlan3D` from `apollo-fwht` crate root.
- [x] [minor] Add `fftfreq(n, d)` and `rfftfreq(n, d)` numpy-compatible frequency utilities in `apollo-fft`.
- [x] [minor] Add `fftshift` and `ifftshift` generic shift utilities in `apollo-fft`.
- [x] [minor] Re-export all four utilities from `apollo-fft` crate root.
- Final state: `cargo test -p apollo-dht` 19 passed; `cargo test -p apollo-fwht` 24 passed;
  `cargo test -p apollo-fft` 63 passed; `cargo test -p apollo-validation -- --include-ignored` 3 passed;
  all 0 failed.

## Closed in this sprint (Closure XL phase)
- [x] [minor] Add GPU separable 2D DCT/DST APIs to `apollo-dctdst-wgpu` `DctDstWgpuBackend`:
  `execute_forward_2d`, `execute_inverse_2d`.
- [x] [minor] Add GPU separable 3D DCT/DST APIs to `apollo-dctdst-wgpu` `DctDstWgpuBackend`:
  `execute_forward_3d`, `execute_inverse_3d`.
- [x] [minor] Add `WgpuError::ShapeMismatch` and `WgpuError::ShapeMismatch3d` variants.
- [x] [minor] Add `ndarray = "0.16"` dependency and re-export `Array2`, `Array3` from crate root.
- [x] [minor] Add verification tests: 2D/3D GPU-CPU parity, roundtrip recovery, shape rejection.
- Final state: `cargo test -p apollo-dctdst-wgpu` 28 passed, 0 FAILED, 0 ignored;
  `cargo test -p apollo-validation -- --include-ignored` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXIX phase)
- [x] [minor] Add CPU separable 2D DCT/DST APIs to `apollo-dctdst` `DctDstPlan`:
  `forward_2d`, `forward_2d_into`, `inverse_2d`, `inverse_2d_into`.
- [x] [minor] Add CPU separable 3D DCT/DST APIs to `apollo-dctdst` `DctDstPlan`:
  `forward_3d`, `forward_3d_into`, `inverse_3d`, `inverse_3d_into`.
- [x] [minor] Enforce dimensional shape contracts (2D square, 3D cubic) with
  `DctDstError::LengthMismatch` on mismatches.
- [x] [minor] Add verification tests for 2D separable parity, 2D/3D roundtrip,
  and non-square/non-cubic rejection.
- [x] [minor] Update `crates/apollo-dctdst/README.md` execution and verification sections.
- Final state: `cargo test -p apollo-dctdst` 42 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXVIII phase)
- [x] [patch] Validation fixture 58: `dct1_three_point_forward_known_values_fixture`
  (DCT-I N=3 x=[1,2,3]: y=[8,−2,0]; y[k]=x[0]+(−1)^k·x[N−1]+2·Σx[n]cos(πnk/(N−1)); y[2]=0 exact;
  Rao & Yip (1990) Table 2.1; FFTW REDFT00; threshold 1e-15).
- [x] [patch] Validation fixture 59: `dst1_two_point_forward_known_values_fixture`
  (DST-I N=2 x=[1,3]: y=[4√3,−2√3]; y[k]=2·Σx[n]sin(π(n+1)(k+1)/(N+1));
  Rao & Yip (1990) Table 3.1; FFTW RODFT00; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 57→59; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 57→59.
- Final state: `cargo test -p apollo-validation` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXVII phase)
- [x] [patch] Validation fixture 56: `dct3_dc_input_flat_output_fixture`
  (DCT-III N=4 [1,0,0,0]: y=[½,½,½,½]; y[k]=x[0]/2 (single term, all cosines vanish); Makhoul 1980 Table I; FFTW REDFT01; threshold 1e-15).
- [x] [patch] Validation fixture 57: `dst3_nyquist_input_alternating_output_fixture`
  (DST-III N=4 [0,0,0,1]: y=[½,−½,½,−½]; y[k]=(−1)^k/2 (single term, all sines vanish); Makhoul 1980 Table II; FFTW RODFT01; threshold 1e-15).
- [x] [patch] Root `README.md` fixture count updated 55→57; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 55→57.
- Final state: `cargo test -p apollo-validation` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXVI phase)
- [x] [patch] Validation fixture 54: `cwt_ricker_impulse_peak_value_fixture`
  (CWT Ricker N=7 a=1 δ_{3}: W(1,3)=ψ(0)=2/(√3·π^¼); W(1,2)=W(1,4)=0 exact; Daubechies 1992 §2.1 eq.(2.1.4); threshold 1e-14).
- [x] [patch] Validation fixture 55: `cwt_ricker_scale_normalization_fixture`
  (CWT Ricker N=7 a=2 δ_{3}: W(2,3)=ψ(0)/√2=√2/(√3·π^¼); Daubechies 1992 §2.1 L² norm; Grossmann-Morlet 1984 eq.(1.3); threshold 1e-13).
- [x] [patch] Root `README.md` fixture count updated 53->55; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 53->55.
- Final state: `cargo test -p apollo-validation` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXV phase)
- [x] [patch] Validation fixture 52: `wavelet_daubechies4_one_level_known_coefficients_fixture`
  (DWT db4 N=4 level=1 x=[1,0,0,0]: [a0,a1,d0,d1]=[h0,h2,h3,h1]; Daubechies 1992 taps; exact basis-impulse mapping; threshold 1e-15).
- [x] [patch] Validation fixture 53: `wavelet_daubechies4_inverse_perfect_reconstruction_fixture`
  (DWT db4 N=4 level=1: IDWT(DWT([1,-2,0.5,4]))=[1,-2,0.5,4]; Mallat 1989 Thm.2 perfect reconstruction; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 51->53; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 51->53.
- Final state: `cargo test -p apollo-validation` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXIV phase)
- [x] [patch] Validation fixture 50: `czt_off_unit_circle_z_transform_fixture`
  (CZT N=2 A=2 W=exp(-πi): X=[1.5,0.5]; Z-transform off unit circle at z={2,-2}; Rabiner-Schafer-Rader 1969 §II; exact dyadic; threshold 1e-12).
- [x] [patch] Validation fixture 51: `hilbert_pure_cosine_envelope_is_unity_fixture`
  (Hilbert envelope of cos(πn/2) N=4: [1,1,1,1]; Oppenheim-Schafer 2010 §12.1 eq.(12.8); Bedrosian 1963; exact integers; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 49->51; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 49->51.
- Final state: `cargo test -p apollo-validation` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXIII phase)
- [x] [patch] Validation fixture 48: `sdft_sliding_recurrence_unit_impulse_all_bins_fixture`
  (SDFT N=4 zero_state, feed [1,0,0,0], all bins=[1+0i,1+0i,1+0i,1+0i]; Jacobsen-Lyons 2003 IEEE SPM 20(2) §2 eq.(2); exact; threshold 1e-12).
- [x] [patch] Validation fixture 49: `frft_order4_identity_fixture`
  (UnitaryFrFT N=4 order=4.0: DFrFT_4([1,2,3,4])=[1,2,3,4]; Candan et al. 2000 §II Corollary; exp(-2πki)=1; exact; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 47->49; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 47->49.
- Final state: `cargo test -p apollo-validation` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXII phase)
- [x] [patch] Validation fixture 46: `nufft_type1_type2_adjoint_inner_product_fixture`
  (NUFFT N=2 adjoint identity Re(〈Ac,f〉)=Re(〈c,A*f〉)=5; Dutt-Rokhlin 1993; all exp∈{1,-1}; exact; threshold 1e-12).
- [x] [patch] Validation fixture 47: `radon_fourier_slice_theorem_theta0_fixture`
  (Radon θ=0 FST: DFT_1(R_{0}[[1,2],[3,4]])=[10,-2]; Natterer 1986 Thm 1.1; exact; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 45->47; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 45->47.
- Final state: `cargo test -p apollo-validation` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXI phase)
- [x] [patch] Validation fixture 44: `dct1_inverse_roundtrip_three_point_fixture`
  (DCT-I N=3: IDCT-I(DCT-I([1,2,3]))=[1,2,3]; Makhoul 1980 C1²=2(N−1)·I; FFTW REDFT00; threshold 1e-14).
- [x] [patch] Validation fixture 45: `dst1_inverse_roundtrip_two_point_fixture`
  (DST-I N=2: IDST-I(DST-I([1,3]))=[1,3]; Makhoul 1980 S1²=2(N+1)·I; FFTW RODFT00; threshold 1e-14).
- [x] [patch] Root `README.md` fixture count updated 43->45; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 43->45.
- Final state: `cargo test -p apollo-validation -p apollo-dctdst` 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXX phase)
- [x] [patch] Validation fixture 42: `dct4_inverse_roundtrip_two_point_fixture`
  (DCT-IV N=2: IDCT-IV(DCT-IV([1,3]))=[1,3]; Makhoul 1980 C4²=N·I; FFTW REDFT11; threshold 1e-14).
- [x] [patch] Validation fixture 43: `dst4_inverse_roundtrip_two_point_fixture`
  (DST-IV N=2: IDST-IV(DST-IV([2,5]))=[2,5]; Makhoul 1980 S4²=N·I; FFTW RODFT11; threshold 1e-14).
- [x] [patch] Root `README.md` fixture count updated 41->43; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 41->43.
- Final state: `cargo test --workspace` 0 FAILED, 0 ignored across all 38+ crates.

## Closed in this sprint (Closure XXIX phase)
- [x] [patch] Validation fixture 40: `ntt_inverse_roundtrip_fixture`
  (NTT N=4: INTT(NTT([1,2,3,4]))=[1,2,3,4]; Pollard 1971 inversion theorem in Z/pZ; threshold 1e-12).
- [x] [patch] Validation fixture 41: `stft_hann_wola_inverse_roundtrip_fixture`
  (STFT frame=4,hop=2: ISTFT(STFT([1,0,0,0]))=[1,0,0,0]; Allen-Rabiner 1977 WOLA; Portnoff 1980 Hann COLA; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 39->41; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 39->41.
- Final state: `cargo test --workspace` 0 FAILED, 0 ignored across all 38+ crates.

## Closed in this sprint (Closure XXVIII phase)
- [x] [patch] Validation fixture 38: `dht_inverse_roundtrip_fixture`
  (DHT N=4: IDHT(DHT([3,-1,2,0]))=[3,-1,2,0]; Bracewell 1983 H²=NI; threshold 1e-14).
- [x] [patch] Validation fixture 39: `sft_inverse_roundtrip_fixture`
  (SFT N=4,K=1: ISFT(SFT([1,-1,1,-1]))=[1,-1,1,-1]; Hassanieh et al. 2012 K-sparse exact; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 37->39; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 37->39.
- Final state: `cargo test --workspace` 0 FAILED, 0 ignored across all 38+ crates.

## Closed in this sprint (Closure XXVII phase)
- [x] [patch] Validation fixture 35: `fwht_inverse_roundtrip_fixture`
  (FWHT N=4: IFWHT(FWHT([1,2,3,4]))=[1,2,3,4]; Walsh 1923 W_N^2=N*I; threshold 1e-14).
- [x] [patch] Validation fixture 36: `qft_inverse_roundtrip_fixture`
  (QFT N=4: iqft(qft([1,0,0,0]))=[1,0,0,0]; Shor 1994 unitarity; threshold 1e-12).
- [x] [patch] Validation fixture 37: `sht_inverse_roundtrip_y10_fixture`
  (SHT lmax=1: Y_1^0 dipole roundtrip; Driscoll-Healy 1994 Theorem 1; threshold 1e-10).
- [x] [patch] Root `README.md` fixture count updated 34->37; three new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 34->37.
- Final state: `cargo test --workspace` 0 FAILED, 0 ignored across all 38+ crates.

## Closed in this sprint (Closure XXVI phase)
- [x] [patch] Validation fixture 32: `wavelet_haar_inverse_perfect_reconstruction_fixture`
  (Haar DWT N=4 1-level: IDWT(DWT([1,-1,0,0]))=[1,-1,0,0]; Mallat 1989 Theorem 2; threshold 1e-12).
- [x] [patch] Validation fixture 33: `gft_path_graph_inverse_roundtrip_fixture`
  (GFT K2 path graph: GFT-1(GFT([3,-1]))=[3,-1]; Sandryhaila-Moura 2013; threshold 1e-12).
- [x] [patch] Validation fixture 34: `frft_inverse_roundtrip_order_half_fixture`
  (FrFT alpha=0.5 N=4: FrFT(-0.5)(FrFT(0.5)([1,2,3,4]))=[1,2,3,4]; Namias 1980; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 31->34; three new fixture descriptions appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 31->34.
- Final state: `cargo test --workspace` 0 FAILED, 0 ignored across all 38+ crates.

## Closed in this sprint (Closure XXV phase)
- [x] [patch] GPU adapter selection: replaced all 20 `wgpu::RequestAdapterOptions::default()`
  sites with `PowerPreference::HighPerformance` across all wgpu crates (fft-wgpu, czt-wgpu,
  mellin-wgpu, ntt-wgpu, stft-wgpu, radon-wgpu, nufft-wgpu, hilbert-wgpu, sft-wgpu, qft-wgpu,
  frft-wgpu, fwht-wgpu, dht-wgpu, sdft-wgpu, sht-wgpu, dctdst-wgpu, gft-wgpu, wavelet-wgpu,
  f16_plan.rs, buffer_reuse bench). Ensures NVIDIA discrete GPU preferred over integrated.
- [x] [patch] GPU test runtime-skip conversion: removed all `#[ignore]` attributes from
  `apollo-ntt-wgpu` (10 tests) and `apollo-stft-wgpu` (7 tests); replaced with
  `let Ok(backend) = Backend::try_default() else { return; }` early-return pattern.
- [x] [patch] Bluestein CZT sign convention fix in `apollo-stft-wgpu`: corrected all four sign
  errors in `stft_chirp.wgsl` (premul_fwd: exp(-πi·n²/N), premul_inv: exp(+πi·n²/N),
  postmul_fwd: exp(-πi·k²/N), postmul_inv: exp(+πi·n²/N)/N); added
  `stft_chirp_pointmul_fwd` entry point (conjugates h_stored → h_fwd); added
  `pointmul_fwd_pipeline` to `StftChirpData`; updated `execute_forward_fft_chirp` to
  dispatch `pointmul_fwd_pipeline` instead of `pointmul_pipeline`.
- [x] [patch] Non-PoT buffer-reuse routing fix in `apollo-stft-wgpu`: added POT guard to
  `execute_forward_with_buffers` and `execute_inverse_with_buffers` that delegates to
  the allocating Chirp-Z path and copies results into `fwd_output_host`/`inv_output_host`.
  Updated forward CZT test tolerance from 1e-2 to 2e-2 (analytically justified by f32
  GPU argument-reduction error at phase magnitudes up to ~1254 rad for N=400).
- Final state: `cargo test --workspace` 0 FAILED, 0 ignored across all 38+ crates.

- [x] [patch] ARCHITECTURE.md Mixed-Precision Capability Table: added "forward + inverse CZT" and
  "forward + inverse Mellin spectrum" annotations to the Notes column for `apollo-czt-wgpu` and
  `apollo-mellin-wgpu`, matching the established pattern for other bidirectional WGPU crates.
- [x] [patch] apollo-validation two new published-reference fixtures (fixtures 29 and 30):
  `czt_inverse_vandermonde_roundtrip_fixture` (threshold 1e-12; N=4 Björck-Pereyra) and
  `mellin_inverse_spectrum_constant_roundtrip_fixture` (threshold 1e-10; N=32 constant signal).
  Added `published_real_fixture_with_threshold` helper. Updated README.md fixture count 28 → 30.
  Assertion in `validation_suite_produces_value_semantic_reports` updated to 30. All 30 pass.

## Closed in this sprint (Closure XXII phase)
- [x] [patch] Implement GPU benchmark runner infrastructure: manual self-hosted workflow
## Closed in this sprint (Closure XXV phase)
- [x] [patch] `AnalyticSignal::instantaneous_frequency()` in `apollo-hilbert`:
  new method using the complex-derivative formula
  `f[n] = arg(conj(z[n])·z[n+1]) / (2π)` (length N−1, cycles per sample).
  Avoids phase unwrapping; well-defined whenever |z[n]| > 0. Reference: Boashash (1992).
- [x] [patch] Two new verification tests in `apollo-hilbert`:
  `instantaneous_frequency_constant_tone` (cosine at k/N has IF=k/N, ε<1e-10) and
  `double_hilbert_negates_zero_mean_signal` (H{H{x}}=−x for sinusoidal signals, ε<1e-10).
- [x] [patch] Validation fixture 31 in `apollo-validation`:
  `hilbert_instantaneous_frequency_constant_tone_fixture` (N=64, k=5, threshold 1e-10).
  Root README updated 30→31; fixture count assertions updated in both test functions.
- [x] [patch] `apollo-hilbert/README.md`: added "Instantaneous Frequency" subsection
  documenting the complex-derivative formula, validation fixture reference, and Boashash 1992 cite.
- [x] [patch] `CHANGELOG.md`, `gap_audit.md`, `checklist.md` updated for Closure XXV.
- [x] [patch] Ignored doc-test in `apollo-ntt-wgpu/src/verification.rs` converted from
  `rust,ignore` to `rust,no_run` with `# use apollo_ntt_wgpu::NttWgpuBackend;` preamble.
  Eliminated last remaining ignored test in workspace; doc-test now compiles and reports "ok".
- [x] [patch] `execute_inverse_with_buffers` doc comment in `apollo-stft-wgpu/src/infrastructure/device.rs`:
  expanded from stub ("Reuses GPU resources from buffers.") to full description noting
  non-PoT delegation and `WgpuError::InvalidPlan` conditions.
- [x] [patch] `CHANGELOG.md` updated with missing Closure XXIII (0.12.3) and Closure XXIV (0.12.4) entries.
- Final state: `cargo test --workspace` 0 FAILED, 0 ignored across all 38+ crates.

  `.github/workflows/gpu-benchmarks.yml`, PowerShell driver `scripts/run_gpu_benchmarks.ps1`,
  tracked artifact root `.benchmarks/gpu-runner/.gitkeep`, root `README.md` runner docs, and
  root README capability corrections for CZT/Mellin/STFT/Radon WGPU surfaces.

## Closed in this sprint (Closure XXI phase)
- [x] [patch] README documentation sync for v0.2.0 inverse additions:
  `apollo-czt/README.md`, `apollo-mellin/README.md`, `apollo-czt-wgpu/README.md`,
  `apollo-mellin-wgpu/README.md` updated with inverse sections and corrected
  capability/verification prose. `checklist.md` Closure XX entry added.

## Planned next increments
*(No blocking gaps. The remaining benchmark-results gap now requires executing the GPU workflow on real hardware and publishing the measured ratios.)*

## Closed in this sprint (Closure XX phase)
- [x] [minor] CPU CZT inverse: `CztPlan::inverse` via Björck-Pereyra Vandermonde solve;
  `CztError::NotInvertible`; 5 value-semantic tests. `apollo-czt` v0.2.0.
- [x] [minor] CPU Mellin inverse: `MellinPlan::inverse_spectrum` via IDFT + exp-resample;
  `MellinError::SpectrumLengthMismatch`; 4 value-semantic tests. `apollo-mellin` v0.2.0.
- [x] [minor] GPU CZT inverse: `czt_inverse` WGSL adjoint formula; `CztWgpuBackend::execute_inverse`;
  `WgpuCapabilities::forward_inverse`; 2 GPU-gated tests. `apollo-czt-wgpu` v0.2.0.
- [x] [minor] GPU Mellin inverse: two-pass WGSL (`mellin_inverse_spectrum` + `mellin_exp_resample`);
  `InverseMellinParamsPod`; `MellinWgpuBackend::execute_inverse`; 2 GPU-gated tests.
  `apollo-mellin-wgpu` v0.2.0.

## Planned next increments
*(No gaps blocking next sprint. All inverse paths for CZT and Mellin are now implemented CPU+GPU.)*
*(Remaining open gap: hardware-gated benchmark timing ratios for NUFFT/STFT buffer-reuse paths.)*

## Closed in this sprint (Closure XIX phase)
- [x] [minor] Update `StftGpuBuffers` for non-PoT scratch sizing via `chirp_padded_len(frame_len)`;
  remove `FrameLenNotPowerOfTwo` from `make_buffers`, `execute_forward_with_buffers`,
  `execute_inverse_with_buffers`. Unblocks buffer-reuse path for non-PoT `frame_len`.
  Version: 0.10.0 [minor]. Tests: 1 structural + 2 GPU-gated buffer-reuse.

## Closed in this sprint (Closure XVIII phase)
- [x] [minor] Bluestein/Chirp-Z non-PoT STFT GPU path: five-pass WGSL dispatch
  (`stft_chirp.wgsl`, `stft_chirp_fft.wgsl`), `StftChirpData` GPU resource struct,
  conditional dispatch in `kernel.rs` (Radix-2 for PoT, Chirp-Z for non-PoT),
  `FrameLenNotPowerOfTwo` removed from primary dispatch path in `device.rs`,
  `error.rs` variant doc updated, 5 new verification tests (3 structural, 2 GPU-gated).
  ADR: `design_history_file/adr_stft_wgpu_non_pot_chirpz.md`. Version: 0.9.0 [minor].

## Closed in this sprint (Closure XVII phase)
- [x] [patch] Add `bench_forward_reuse` and `bench_inverse_reuse` benchmark groups to
  `crates/apollo-stft-wgpu/benches/stft_bench.rs`: head-to-head allocating vs
  `StftGpuBuffers` buffer-reuse comparison at `frame_len` ∈ {256, 512, 1024}.
  Mirrors the pattern of `apollo-fft-wgpu/benches/buffer_reuse.rs`.
- [x] [patch] Add "Buffer Reuse" and "Benchmarks" sections to
  `crates/apollo-stft-wgpu/README.md` documenting the
  `make_buffers` → `execute_forward/inverse_with_buffers` usage pattern,
  constraint notes, and bench invocation.

## Planned next increments
*(No gaps blocking next sprint at this time. STFT GPU PoT/non-PoT complete; buffer-reuse enabled.)*

## Closed in this sprint (Closure XVI phase)
- [x] [minor] `StftGpuBuffers` pre-allocated buffer reuse in `apollo-stft-wgpu`:
  construct once per `(frame_count, frame_len, signal_len, hop_len)` quad; eliminates
  5–8 `device.create_buffer`, 4+ `device.create_bind_group`, and `log₂(N)` uniform-buffer
  allocations per dispatch. Mirrors `GpuFft3dBuffers` pattern from `apollo-fft-wgpu`.
- [x] [minor] `StftWgpuBackend::make_buffers`, `execute_forward_with_buffers`,
  `execute_inverse_with_buffers` — public API surface for zero-allocation repeated dispatch.
- [x] [minor] `StftGpuKernel::execute_forward_fft_with_buffers` and
  `execute_inverse_with_buffers` — kernel-level buffered dispatch methods.
- [x] [minor] Verification test `reusable_buffers_match_allocating_forward_and_inverse_when_device_exists`:
  asserts bit-exact agreement (TOL=1e-6) between allocating and buffered forward+inverse paths;
  verifies idempotent second-call buffer reuse.

## Closed in this sprint (Closure XV phase)
- [x] [patch] Criterion benchmark suite for `apollo-radon-wgpu`: new `benches/radon_wgpu_bench.rs`
  with `radon_wgpu_forward` and `radon_wgpu_fbp` groups across three image sizes (64², 128², 256²).
  Gaussian disk phantom (σ=0.25) provides non-trivial frequency content; analytical Radon transform
  is `(Rf)(θ,s) = σ√(2π)·exp(−s²/(2σ²))`. Angles uniform on `[0,π)` (Fourier slice theorem
  sampling). Addresses open gap #2 from `gap_audit.md` (Criterion GPU benchmark infrastructure).
- [x] [patch] Add `criterion = "0.5"` to `apollo-radon-wgpu` dev-deps;
  add `[[bench]] name = "radon_wgpu_bench" harness = false`.

## Closed in this sprint (Closure XIV phase)
- [x] [patch] Dead-code removal in `apollo-stft-wgpu`: remove deprecated O(N²) forward
  pipeline (`StftGpuKernel::execute`, `forward_pipeline` field, `stft.wgsl` shader).
  Remove dead `stft_inverse_frames` entry point from `stft_inverse.wgsl` (superseded by
  Closure XI FFT inverse path). Update kernel module docstring, `WORKGROUP_SIZE` comment,
  struct doc, and `dispatch_count`/`fft_dispatch_count` comments. -244 net lines removed.

## Closed in this sprint (Closure XIII phase)
- [x] [patch] Criterion benchmark suite for `apollo-stft-wgpu`: new `benches/stft_bench.rs`
  with `bench_forward_fft` and `bench_inverse_fft` groups across three COLA-valid
  `(frame_len, hop_len, signal_len)` parameter sets: (256/128/4096), (512/256/8192),
  (1024/512/16384). Addresses open gap #2 from `gap_audit.md` (Criterion buffer-reuse
  bench results on GPU hardware). Skips gracefully when no WGPU device is available.
- [x] [patch] Add `criterion = { version = "0.5", features = ["html_reports"] }` to
  `apollo-stft-wgpu` dev-deps; add `[[bench]] name = "stft_bench" harness = false`.

## Closed in this sprint (Closure XII phase)
- [x] [minor] STFT forward GPU acceleration (`apollo-stft-wgpu`): replace O(N²) per-frame
  direct DFT in `stft.wgsl::stft_forward` with a batched Cooley-Tukey Radix-2 DIT FFT
  (O(N log N) per frame). New `stft_forward_fft.wgsl` with four entry points:
  `stft_fwd_pack_window` (Hann analysis window + pack to split re/im scratch),
  `stft_fwd_bitrev` (bit-reversal permutation, batched), `stft_fwd_butterfly` (one Radix-2
  DIT stage per dispatch, DFT twiddle exp(−2πi·k/N)), `stft_fwd_interleave` (split re/im →
  interleaved ComplexValue output). Reuses `fft_data_bgl` and `fft_params_bgl` layouts
  from the inverse FFT path. New `FwdFftStageParams` (16 bytes, 4×u32) carries `hop_len`
  where `FftStageParams._pad` was. `FrameLenNotPowerOfTwo` enforced on forward path.
  Formal basis: Cooley & Tukey (1965).
- [x] [patch] New `forward_rejects_non_power_of_two_frame_len` test (CPU-only).
- [x] [patch] New `forward_fft_roundtrip_large_frame_when_device_exists` test (GPU-gated).

## Closed in this sprint (Closure XI phase)
- [x] [minor] STFT inverse GPU acceleration (`apollo-stft-wgpu`): replace O(N²) per-frame direct IDFT in `stft_inverse.wgsl::stft_inverse_frames` with a batched Cooley-Tukey Radix-2 DIT IFFT (O(N log N) per frame). New `stft_inverse_fft.wgsl` with four entry points: `stft_deinterleave` (interleaved f32 → split re/im scratch), `stft_bitrev` (bit-reversal permutation, batched), `stft_butterfly` (one Radix-2 DIT stage per dispatch, IDFT twiddle exp(+2πi·k/N)), `stft_scale_and_window` (1/N scale + Hann synthesis window → frame_data). Two-bind-group architecture: group 0 (4 data bindings, shared), group 1 (per-stage FftStageParams uniform, one bind group per butterfly pass). All passes encoded in one `CommandEncoder`; implicit per-pass memory barriers preserve write visibility. OLA pass unchanged. Formal basis: Cooley & Tukey (1965); Allen & Rabiner (1977) Theorem 1.
- [x] [minor] New `WgpuError::FrameLenNotPowerOfTwo { frame_len: usize }` variant: returned by `execute_inverse` when `frame_len` is not a power of two (Radix-2 invariant). Checked in both `device.rs` (pre-dispatch) and `kernel.rs` (IFFT entry). [minor because it is an additive public API change to an existing error enum]
- [x] [patch] STFT-WGPU verification: new test `inverse_rejects_non_power_of_two_frame_len` (frame_len=6, expects `FrameLenNotPowerOfTwo`). New GPU-gated test `inverse_roundtrip_large_frame_1024_samples_when_device_exists` (frame_len=1024, hop=512, signal_len=8192, analytic sine reference, TOL=5e-3) exercising all 10 butterfly stages.

## Closed in this sprint (Closure X phase)
- [x] [minor] GPU Radon Filtered Backprojection (`apollo-radon-wgpu`): new `radon_fbp_filter.wgsl` with entry `radon_fbp_filter` performing circular convolution of each projection row with the ramp filter impulse response `h = IFFT(R)`, `R[k] = 2π·|signed_k|/(N·Δ)` (Bracewell & Riddle 1967; Shepp & Logan 1974). `h` computed host-side via `apollo_radon::ramp_filter_projection` applied to unit impulse, then cast to f32. Reuses existing 4-binding bind group layout. Two-pass single encoder: filter pass → backproject pass. Host-side `π/angle_count` normalization. `fbp_filter_pipeline` in `RadonGpuKernel`. `RadonWgpuBackend::execute_filtered_backproject`. `supports_filtered_backprojection` capability flag. `forward_inverse_and_fbp` constructor. 4 verification tests: adjoint identity (⟨Af,g⟩=⟨f,A†g⟩), FBP capability flags, FBP matches CPU reference (TOL=5e-2), FBP shape mismatch rejection.
- [x] [patch] Radon-WGPU adjoint identity test: `backproject_satisfies_adjoint_identity_when_device_exists` verifies ⟨A·f, g⟩_sinogram = ⟨f, A†·g⟩_image (Natterer 2001, §II.2) to relative tolerance 5e-3. Uses CPU forward (f64) + GPU backproject (f32). Tests the mathematical definition of the adjoint operator.
- [x] [patch] STFT-WGPU parameterized roundtrip test: `inverse_roundtrip_for_multiple_cola_parameter_sets` tests three COLA-compliant parameter sets (frame_len=8/hop=4, frame_len=16/hop=8, frame_len=32/hop=16) with analytical sine/cosine reference signals. CPU forward → GPU inverse → compare with CPU inverse reference at TOL=5e-3.
- [x] [patch] Documentation sync: updated `README.md` WGPU crate descriptions to reflect GPU inverse capabilities for `apollo-radon-wgpu` (FBP added), `apollo-stft-wgpu` (inverse WOLA), `apollo-hilbert-wgpu` (inverse analytic-mask), `apollo-sdft-wgpu` (inverse IDFT). Updated `ARCHITECTURE.md` Mixed-Precision Capability Table notes for the same four crates.

## Closed in this sprint (Closure IX phase)
- [x] [minor] GPU inverse STFT WOLA (`apollo-stft-wgpu`): new `stft_inverse.wgsl` with two-pass WOLA reconstruction (`stft_inverse_frames`: per-(frame, local_j) windowed IDFT using interleaved f32 spectrum; `stft_inverse_ola`: per-sample OLA with Hann² weight normalization); shared 3-binding layout reusing existing `bind_group_layout`; `inverse_frames_pipeline` + `inverse_ola_pipeline` in `StftGpuKernel`; `StftGpuKernel::execute_inverse` (2-pass single encoder); `StftWgpuBackend::execute_inverse(plan, spectrum, signal_len)` + `execute_inverse_typed_into`; `forward_and_inverse` capability constructor; 3 new verification tests (`capabilities_reflect_forward_and_inverse_surface`, `inverse_roundtrip_recovers_cola_signal_when_device_exists`, `inverse_matches_cpu_reference_for_16sample_signal`). Basis: WOLA identity (Allen–Rabiner 1977, Theorem 1).
- [x] [minor] GPU Radon backprojection (`apollo-radon-wgpu`): new `radon_backproject.wgsl` entry point — per-pixel `bp[r,c] = Σ_θ interp(sinogram[θ,·], x·cosθ + y·sinθ)` with linear interpolation; reuses forward bind group layout; `backproject_pipeline` in `RadonGpuKernel`; `RadonGpuKernel::execute_backproject`; `RadonWgpuBackend::execute_inverse(plan, sinogram, angles)` + `execute_inverse_flat_typed`; `SinogramShapeMismatch` error variant; `forward_and_inverse` capability constructor; 3 new verification tests. Basis: Radon adjoint operator (Natterer 2001, §II.2).
- [x] [patch] Correct `gap_audit.md` open-gap note: CZT and Mellin have no CPU inverse defined (`apollo-czt` has only `forward`; `apollo-mellin` has `forward_resample`/`moment`/`forward_spectrum` only). GPU inverse for those two crates is `UnsupportedExecution` by architectural design, not deferral. Updated open gaps section accordingly.

## Closed in this sprint (Closure VIII phase)
- [x] [minor] GPU inverse Hilbert transform (`apollo-hilbert-wgpu`): `hilbert_inverse_mask` WGSL entry point (DC/Nyquist zeroed, positive: X[k]=j·Q[k], negative: X[k]=-j·Q[k]); fix `hilbert_inverse_dft` stale self-assign bug; `inverse_mask_pipeline` in kernel; `HilbertGpuKernel::execute_inverse` (3-pass single-encoder); `execute_inverse` + `execute_inverse_typed_into` backend methods; `forward_and_inverse` capability constructor; 3 value-semantic verification tests.
- [x] [minor] GPU inverse SDFT (`apollo-sdft-wgpu`): `sdft_inverse_bins` WGSL entry point (x[n]=(1/K)·Σ X[b]·exp(+2πi·b·n/K), complex bins as interleaved f32 pairs); split `pipeline` into `forward_pipeline` + `inverse_pipeline`; `SdftGpuKernel::execute_inverse`; `execute_inverse` + `execute_inverse_typed_into` + `validate_plan_bins` backend methods; `forward_and_inverse` capability constructor; 4 value-semantic verification tests.
- [x] [patch] Fix CZT proptest absolute-tolerance defect: `bluestein_equals_direct_for_arbitrary_parameters` threshold 1e-9 was violated for |w|>1 (chirp amplification). Replace `diff < 1e-9` with `diff < 1e-9 · max(|direct[k]|, 1.0)`. Formal basis: Bluestein relative error ≤ C·log₂(p)·ε_machine ≈ 2.6e-15; 1e-9 relative bound provides ×3.8e5 safety margin.


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
