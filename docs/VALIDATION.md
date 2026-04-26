# Apollo Validation

Apollo validation is structured as named suites that can run independently or as a single report via
`cargo run -p apollo-validation`. For performance comparisons, use
`cargo run -p apollo-validation --release`; debug builds are useful for correctness but not for timing claims.

## Suite API

`apollo-validation` exposes a multi-crate API so callers can select only the validation surfaces they
need. The top-level crate exports the suite entry points, while the internal application module
composes the reports from Apollo FFT, Apollo NUFFT, Apollo FFT WGPU, Apollo NUFFT WGPU, and optional external reference providers.

The primary entry points are:

- `run_fft_cpu_suite()`
- `run_fft_gpu_suite()`
- `run_nufft_suite()`
- `run_external_comparison_suite()`
- `run_benchmark_suite()`
- `run_validation_suite()`
- `run_smoke_suite()`
- `run_full_suite()`

## Report Shape

The validation CLI emits JSON with these top-level sections:

- `fft_cpu`
- `fft_gpu`
- `nufft`
- `external`
- `benchmarks`
- `environment`

The crate test suite serializes `ValidationReport` and checks that these
sections, CPU invariant fields, and external-comparison fields remain present.
This is a schema-shape guard; numeric thresholds remain in the value-semantic
suite checks.

The `external` section contains per-backend subreports for optional comparison engines such as
`rustfft` and `pyfftw`, plus NumPy-backed comparison data when available. The report records
availability, whether a comparison was attempted, and any skip or failure notes. It also includes
`external.published_references`, a deterministic fixture suite checked against closed-form tables
from published transform definitions.

The FFT sections also include precision-aware detail:

- `fft_cpu.precision_profiles`
- `fft_gpu.precision_profiles`
- `external.precision_comparisons`
- `benchmarks.precision_benchmarks`

These report `high_accuracy`, `low_precision`, and `mixed_precision` runs when implemented by the
backend. Apollo currently reports CPU support for all three FFT profiles and WGPU support only for
`low_precision`.

## Optional External Validation Dependencies

External reference engines are optional and must not be required by crates other than
`apollo-validation`.

- `rustfft` is used only by validation reference code.
- `realfft` is not a workspace dependency and is not imported by validation code.
- `pyfftw` is optional and is probed only when the host Python environment can import it.
- NumPy-backed comparison probes are also optional and are skipped when the interpreter or module is
  unavailable.

Production crates do not depend on these validation-only engines. The validation crate owns the
reference integration boundary and should be the only place where these dependencies are activated.
The current external Rust reference feature is `apollo-validation/external-references`, which enables
only the optional `rustfft` dependency.

## Precision Thresholds

Apollo stores mixed-precision validation thresholds centrally in the validation suite:

- CPU `high_accuracy`: current machine-precision thresholds.
- CPU `low_precision`: relative error target below `1e-5`.
- CPU `mixed_precision`: relative error target below `1e-2`.
- WGPU `low_precision`: current forward/inverse parity envelope against the Apollo CPU reference.
- WGPU `mixed_precision`: not attempted because it is not implemented yet.

The current CPU `mixed_precision` FFT path means `half::f16` storage with `f32` compute. Apollo
does not use that label for WGPU because the current shader path is `f32` only.

## Adversarial Coverage

- Degenerate dimensions.
- Buffer/device limit rejection in the FFT WGPU crate.
- Shape mismatches.
- Invalid scale bounds in Mellin and wavelet plans.
- Non-finite angles and invalid detector geometry in Radon plans.
- Non-contiguous Python buffers.
- Repeated-run stability against optional external references.
- Published-reference fixture parity for DFT, DHT, DCT-II, and DST-II definitions.
- Non-finite signal propagation checks in Apollo CPU FFT.
- Prime-sized and mixed-shape external parity checks.
- Clustered and wrapped NUFFT point distributions.
- Precision-profile comparisons against the Apollo `f64` reference for implemented lower-precision FFT paths.
- Backend-unavailable paths with explicit notes instead of silent fallbacks.
- Cache reuse under contention.

## Transform-Specific Analytical Checks

## Published-Reference Fixtures

`apollo-validation` now records deterministic published-reference fixtures under
`external.published_references`. The current fixture set validates:

- DFT4 `[1, 0, -1, 0] -> [0, 2, 0, 2]` from the finite root-of-unity DFT
  definition used by Cooley and Tukey.
- DHT4 `[1, 0, -1, 0] -> [0, 2, 0, 2]` from Bracewell's
  `cas(theta) = cos(theta) + sin(theta)` Hartley basis.
- DCT-II length-2 `[1, 3] -> [4, -sqrt(2)]` from the unnormalized FFTW REDFT10
  convention.
- DST-II length-2 `[1, 3] -> [sqrt(2), -2]` from the unnormalized FFTW RODFT10
  convention.

These fixtures are version-stable and do not require optional external
dependencies.

- DHT: impulse response, DC behavior, Parseval scaling, and involution.
- DCT/DST: two-point analytical projections, inverse-pair scaling, and
  caller-owned inverse parity.
- Hilbert: cosine-to-sine quadrature, constant/DC behavior, Nyquist behavior,
  analytic-signal envelope, and real-part preservation.
- Mellin: log-resampling endpoints, constant moments, power-law moments, and
  log-frequency DC behavior.
- Radon: row/column projection identities, forward/adjoint inner-product
  identity, ramp-filter DC removal, and projection mass conservation.
- SDFT: recurrence states compared with direct DFT recomputation after updates.
- SFT: top-K support selection, dense inverse reconstruction, and direct DFT
  reference comparison through the crate-owned verification kernel.
- SHT: constant-surface and single-mode spherical harmonic identities.
- Wavelet: analytical Haar coefficients, inverse reconstruction, and CWT impulse
  localization.

## Optional External Assets

`apollo/external/rustfft` and `apollo/external/pyfftw` remain intentionally untracked. Apollo validation
must still run without them; the report records their presence and includes skipped/comparison notes
rather than failing by omission. Direct NumPy comparison is attempted through the host Python
interpreter, and optional `pyfftw` comparison is enabled only when the module is importable.

## Current Audit Direction

Recent Apollo optimization work follows the same themes emphasized by current high-performance FFT
and NUFFT literature:

- Keep NUFFT spreading cache-aware by reordering points and improving locality.
- Reduce transient allocation in hot loops so bandwidth is spent on useful arithmetic.
- Treat GPU FFT throughput as a kernel-orchestration problem: host-device traffic and per-stage
  launches dominate small and medium transforms, so future Apollo WGPU work should keep axis data
  resident on device and fuse more stages.

Primary references:

- FINUFFT documentation highlights point bin-sorting, SIMD spreading, and lower RAM use as major CPU
  NUFFT performance drivers.
- Barnett et al., “A parallel non-uniform fast Fourier transform library based on an exponential of
  semicircle kernel” motivates cache-aware spreading and memory-efficient kernels.
- Shih et al., “cuFINUFFT” emphasizes cache-aware point reordering and blocked spreading in shared
  memory for GPU NUFFT.
- Wu et al., “TurboFFT” shows that kernel fusion is central to competitive GPU FFT performance.

## Local Criterion Benchmarks

Apollo FFT kernel strategy benchmarks live in `crates/apollo-fft/benches/kernel_strategy.rs`.
Run them with:

```text
cargo bench -p apollo-fft --bench kernel_strategy
```

The benchmark group compares:

- `direct_dft`: the O(N^2) analytical baseline.
- `radix2_inplace`: the power-of-two Cooley-Tukey path.
- `bluestein_inplace`: the arbitrary-length chirp-Z path.

These benchmarks are performance diagnostics only. Correctness remains enforced
by unit and property tests against analytical identities and direct references.

## Current Memory-Efficiency Checks

- FFT 2D/3D axis passes gather non-contiguous ndarray lanes once, transform the
  gathered lane buffers in place, and scatter them back without constructing a
  second transformed-lane collection.
- CZT Bluestein execution reuses its convolution workspace after the forward
  FFT, multiplies by the precomputed kernel in place, and inverse-transforms the
  same buffer instead of copying to a separate product vector.
- FWHT exposes caller-owned real and complex output paths that copy once into
  the output buffer and then reuse the in-place butterfly kernel.
- FrFT integer-order plans store finite cotangent/cosecant state for singular
  quarter rotations, and inverse execution is verified through caller-owned and
  allocating paths.
- NTT exposes caller-owned and true in-place paths, normalizes inputs into
  residue classes before butterflies, and uses overflow-safe modular addition.
- STFT documents centered-frame overlap-add reconstruction, removes filler
  module comments, and verifies caller-owned inverse parity.
- SFT uses non-deprecated ndarray vector extraction and reuses the crate-local
  direct DFT reference in verification.
- NUFFT 1D type-2 interpolation borrows the inverse oversampled grid as a
  contiguous slice instead of copying it before interpolation. The fast path
  restores Apollo FFT's inverse `1/M` normalization before interpolation so the
  gridded adjoint matches the exact type-2 exponential sum.
- NUFFT 3D separable FFT passes reuse one `Array1<Complex64>` lane buffer per
  axis pass rather than allocating per transformed lane.
- NUFFT 3D type-2 interpolation sorts positions directly and writes results
  back by original index, avoiding dummy value buffers.
- NUFFT Kaiser-Bessel `I_0` uses the defining positive-term series with
  machine-precision truncation rather than low-order polynomial coefficients.
- Wavelet Morlet validation includes a zero-mean admissibility check for the
  DC-corrected real Morlet mother wavelet.
- WGPU backend validation is split by transform domain: `apollo-fft-wgpu`
  owns dense FFT device and shader checks, while `apollo-nufft-wgpu` now
  validates exact direct Type-1 and Type-2 execution plus fast Kaiser-Bessel
  gridding for 1D and 3D against `apollo-nufft` exact and gridded references.
- `apollo-fwht-wgpu` now validates real 1D forward and inverse FWHT execution
  against the owning CPU crate and reports support only for that implemented
  `f32` kernel surface.
- `apollo-dctdst-wgpu` now validates real 1D DCT-II, DCT-III, DST-II, and
  DST-III forward and inverse execution against the owning CPU crate and
  reports full DCT/DST support for that implemented `f32` kernel surface.
- `apollo-czt-wgpu` now validates direct complex forward CZT execution against
  the owning CPU crate's direct reference kernel and reports forward-only
  support for the implemented `f32` complex surface.
- `apollo-gft-wgpu` now validates forward and inverse graph Fourier basis
  execution against the owning CPU crate and reports support for the
  implemented `f32` real basis surface.
- `apollo-hilbert-wgpu` now validates forward analytic-signal and quadrature
  Hilbert execution against the owning CPU crate and reports forward-only
  support for that implemented `f32` surface.
- `apollo-mellin-wgpu` now validates forward Mellin log-frequency spectrum
  execution against the owning CPU crate and reports forward-only support for
  that implemented `f32` surface.
- `apollo-ntt-wgpu` now validates forward and inverse modular NTT execution
  against the owning CPU crate and reports support for the implemented 32-bit
  modulus surface.
- `apollo-qft-wgpu` now validates forward and inverse dense unitary QFT
  execution against the owning CPU crate and reports support for the
  implemented `f32` complex surface.
- `apollo-radon-wgpu` now validates forward parallel-beam Radon projection
  execution against the owning CPU crate and reports forward-only support for
  the implemented `f32` image/sinogram surface.
- `apollo-sdft-wgpu` now validates forward direct-bin sliding DFT execution
  against the owning CPU crate and reports forward-only support for the
  implemented `f32` real-window / complex-bin surface.
- `apollo-sft-wgpu` now validates dense direct DFT sparse top-k projection and
  inverse sparse reconstruction against the owning CPU crate and reports
  support for the implemented `f32` complex execution surface.
- `apollo-sht-wgpu` now validates forward and inverse complex SHT execution
  against the owning CPU crate and reports support for direct GPU matrix sums.
  The WGPU path now receives owner-derived quadrature samples and generates the
  associated-Legendre/spherical-harmonic basis buffer on GPU before reduction.
  The generated basis storage is allocated on device without a host-side zero
  upload because the basis-generation pass writes every entry before use.
- `apollo-stft-wgpu` now validates forward Hann-windowed STFT execution
  against the owning CPU crate and reports forward-only support for the
  implemented `f32` signal / complex-spectrum surface.
- `apollo-wavelet-wgpu` now validates forward and inverse Haar DWT execution
  against the owning CPU crate and reports support for the implemented `f32`
  Haar surface.
- `apollo-dht-wgpu` now validates real 1D forward and inverse DHT execution
  against the owning CPU crate and reports support only for that implemented
  `f32` kernel surface.
- `apollo-nufft-wgpu` now validates exact direct Type-1 and Type-2 summations
  for 1D and 3D against `apollo-nufft` exact reference functions and reports
  support for those direct GPU surfaces.
- Transform WGPU crates report execution support only for implemented kernels
  with CPU parity tests against their owning transform crates.
