# Apollo Architecture

## Dependency Rules

- `domain` defines the single source of truth for shapes, normalization, capabilities, device descriptors, and error contracts.
- `application` owns orchestration, cache policy, reusable plans, and zero-allocation execution paths.
- `infrastructure` owns concrete adapters such as CPU and WGPU.
- Public APIs are exposed from `lib.rs` and narrow facade modules only.

Allowed dependency direction:

`api/lib.rs -> application -> domain`

`api/lib.rs -> infrastructure -> application -> domain`

Infrastructure implementations may depend on shared traits and shared data contracts, but `domain` and `application` may not depend on a backend-specific module.

## SOC, SSOT, SRP, DIP, DRY

- Separation of concerns: numerical contracts, execution plans, and backend transport are split into separate modules.
- Single source of truth: shape metadata, normalization conventions, cache keys, and backend capability descriptors are defined once in `domain`.
- Single responsibility: each plan type owns exactly one dimensionality and one normalization convention.
- Dependency inversion: consumers program against `FftBackend`, not against CPU or GPU internals.
- Do not repeat yourself: helper constructors and shared validation live centrally and are reused by all crates.

## Precision Model

- Precision is a domain-level contract, not an incidental backend detail.
- `PrecisionMode`, `StoragePrecision`, `ComputePrecision`, and `PrecisionProfile` are defined once
  in Apollo domain types and reused across Rust, Python, validation, and compatibility layers.
- Backends must advertise only the precision profiles they truly implement.
- Apollo never silently upgrades or downgrades a caller into mixed precision; lower-precision paths
  are explicit plan or API choices.
- Apollo currently defines `mixed_precision` for CPU FFT as `half::f16` storage with `f32` compute.

## Documentation Standard

All public types and methods must document:

- The algorithm family in use.
- The key theorem or invariant relied upon.
- A proof sketch or reasoning note.
- Complexity and allocation behavior.
- Normalization rules.
- Failure modes.

## Mixed-Precision Capability Table

The table below is the authoritative record of per-crate precision support. "Advertised profile" is the `default_precision_profile` value exposed in each crate's capability struct. "Supported storage" lists accepted host-storage types for typed/caller-owned APIs. "GPU compute" is the arithmetic precision used inside shaders or GPU kernels.

| Crate | Backend | Advertised profile | Supported storage | GPU compute | Notes |
|---|---|---|---|---|---|
| apollo-fft | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 at plan boundary |
| apollo-czt | CPU | HIGH_ACCURACY | Complex64, Complex32, [f16;2] | — | f16 promoted to Complex32 |
| apollo-dctdst | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-dht | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-frft | CPU | HIGH_ACCURACY | Complex64, Complex32, [f16;2] | — | f16 promoted to Complex32 |
| apollo-fwht | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-gft | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-hilbert | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-mellin | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-ntt | CPU | exact u64 residues | u64 mod p | — | floating mixed precision unsupported by design |
| apollo-nufft | CPU | HIGH_ACCURACY | Complex64, Complex32, [f16;2] | — | f16 promoted to Complex32 |
| apollo-qft | CPU | HIGH_ACCURACY | Complex64, Complex32, [f16;2] | — | f16 promoted to Complex32 |
| apollo-radon | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-sdft | CPU | HIGH_ACCURACY | f64/Complex64, f32/Complex32, f16/[f16;2] | — | f16 promoted to f32/Complex32 |
| apollo-sft | CPU | HIGH_ACCURACY | Complex64, Complex32, [f16;2] | — | f16 promoted to Complex32 |
| apollo-sht | CPU | HIGH_ACCURACY | f64/Complex64, f32/Complex32, f16/[f16;2] | — | f16 promoted to f32/Complex32 |
| apollo-stft | CPU | HIGH_ACCURACY | f64/Complex64, f32/Complex32, f16/[f16;2] | — | f16 promoted to f32/Complex32 |
| apollo-wavelet | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-fft-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 (mixed) | f32 | native-f16 feature: arithmetic in f16 via SHADER_F16 |
| apollo-czt-wgpu | WGPU | LOW_PRECISION_F32 | f32, [f16;2] host (mixed) | f32 | f16 promoted to f32 at host boundary |
| apollo-dctdst-wgpu | WGPU | LOW_PRECISION_F32 | f32 | f32 | mixed f16 host path present |
| apollo-dht-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 host (mixed) | f32 | f16 promoted at host boundary |
| apollo-frft-wgpu | WGPU | LOW_PRECISION_F32 | f32, [f16;2] host (mixed) | f32 | f16 promoted at host boundary; UnitaryFrftGpuKernel available |
| apollo-fwht-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 host (mixed) | f32 | f16 promoted at host boundary |
| apollo-gft-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 host (mixed) | f32 | f16 promoted at host boundary |
| apollo-hilbert-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 host (mixed) | f32 | forward + inverse analytic-mask; f16 promoted at host boundary |
| apollo-mellin-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 host (mixed) | f32 | f16 promoted at host boundary |
| apollo-ntt-wgpu | WGPU | exact u32 residues | u32 quantized | u32 modular | floating mixed precision explicitly unsupported |
| apollo-nufft-wgpu | WGPU | LOW_PRECISION_F32 | Complex32, [f16;2] host (mixed) | f32 | f16 promoted at host boundary |
| apollo-qft-wgpu | WGPU | LOW_PRECISION_F32 | Complex32, [f16;2] host (mixed) | f32 | f16 promoted at host boundary |
| apollo-radon-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 host (mixed) | f32 | forward + adjoint backprojection + FBP; f16 promoted at host boundary |
| apollo-sdft-wgpu | WGPU | LOW_PRECISION_F32 | f32, [f16;2] host (mixed) | f32 | forward + inverse direct-bins IDFT; f16 promoted at host boundary |
| apollo-sft-wgpu | WGPU | LOW_PRECISION_F32 | Complex32, [f16;2] host (mixed) | f32 | f16 promoted at host boundary |
| apollo-sht-wgpu | WGPU | LOW_PRECISION_F32 | Complex32, [f16;2] host (mixed) | f32 | f16 promoted at host boundary |
| apollo-stft-wgpu | WGPU | LOW_PRECISION_F32 | Complex32, [f16;2] host (mixed) | f32 | forward + inverse FFT-accelerated (Radix-2 DIT, O(N log N)); PoT frame_len required; f16 promoted at host boundary |
| apollo-wavelet-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 host (mixed) | f32 | f16 promoted at host boundary |

### Key: native-f16 GPU (apollo-fft-wgpu)

When the `native-f16` feature is enabled and the WGPU adapter exposes `wgpu::Features::SHADER_F16`, `GpuFft3dF16Native` executes all butterfly arithmetic in `f16` inside the shader. The host boundary converts `f32` input to `f16` before upload and `f16` output to `f32` after readback. Twiddle factors are computed in `f32` then narrowed to `f16` at plan build time to bound two-source error. Accumulation error is `O(log N)·ε_f16` where `ε_f16 ≈ 9.77×10⁻⁴`. Non-power-of-two sizes are supported via a Bluestein chirp-Z f16 shader (`chirp_native_f16.wgsl`).

### Key: NTT precision contract

`apollo-ntt` and `apollo-ntt-wgpu` operate exclusively on exact modular residues. Floating-point mixed precision is architecturally unsupported because modular arithmetic requires exact integer representation. The WGPU surface uses `u32` residues (values mod p where p ≤ u32::MAX); the CPU surface uses `u64` residues for the default 998244353 modulus with 128-bit-widened intermediate products.

### Key: Unitary FrFT (apollo-frft, apollo-frft-wgpu)

Two plans coexist for the fractional Fourier transform:

| Plan | Construction | Per-call | Unitarity |
|---|---|---|---|
| `FrftPlan` | O(1) | O(N²) | Non-unitary for non-integer orders (‖M†M‖[j,j] = 1/|sin α|) |
| `UnitaryFrftPlan` | O(N³) | O(N²) | Provably unitary: ‖DFrFT_a(x)‖₂ = ‖x‖₂ for all real a |

`UnitaryFrftPlan` uses the Candan (2000) eigendecomposition of the Grünbaum commuting matrix.
Its eigenvector basis V satisfies V^T V = I and eigenvectors are symmetric or antisymmetric
under index reversal. DFrFT_a(x) = V · diag(exp(−iakπ/2)) · V^T · x.

The GPU backend `apollo-frft-wgpu` exposes `execute_unitary_forward` and `execute_unitary_inverse`
via `UnitaryFrftGpuKernel`. V is precomputed on CPU (O(N³)) and uploaded as an f32 storage buffer.
Three sequential GPU submissions execute the 3-pass algorithm with `device.poll` barriers between
passes to guarantee cross-workgroup storage ordering.

See `design_history_file/adr_unitary_frft.md` for the full algorithm selection record.
