# Apollo FFT Workspace

Apollo is a standalone nested workspace for reusable FFT planning, execution, validation, and Python bindings.

Stage 2 moves Apollo beyond the initial compatibility cut:

- `apollofft` owns reusable CPU FFT and NUFFT implementations.
- `apollofft-wgpu` owns the real shader-backed 3D WGPU FFT path with radix-2 and Bluestein/Chirp-Z axis strategies.
- `apollo-validation` emits structured CPU, GPU, NUFFT, benchmark, and external-comparison reports, including direct parity against `rustfft` and NumPy when Python is available.
- `pyapollofft` exposes FFT, R2C/C2R, NUFFT, and backend capability introspection for Python callers.

Mixed precision is now a first-class Apollo concept:

- CPU defaults to `high_accuracy` (`f64` storage and `f64` compute).
- CPU also supports opt-in `low_precision` (`f32` storage and `f32` compute).
- CPU also supports opt-in `mixed_precision` (`half::f16` storage with `f32` compute in the current FFT path).
- WGPU currently exposes only the truthful `low_precision` profile because the shipped shaders execute in `f32`; Apollo does not advertise mixed precision there until a real mixed arithmetic path exists.

## Crates

- `apollofft`: CPU FFT plans, cache management, shared types, and backend abstractions.
- `apollofft-wgpu`: Real WGPU backend and GPU parity surface.
- `apollofft-cudatile`: trait-compatible cudatile adapter surface.
- `apollo-validation`: parity, adversarial, benchmark, and external-reference runners.
- `pyapollofft`: PyO3 bindings, NumPy interop, and backend introspection.

## Precision Profiles

Apollo exposes these precision descriptors through Rust and Python:

- `high_accuracy`
- `low_precision`
- `mixed_precision`

In Rust, they are represented by `PrecisionMode`, `StoragePrecision`, `ComputePrecision`, and
`PrecisionProfile`. Existing APIs keep their current default behavior; lower-precision paths are
opt-in via `with_precision(...)` constructors or the generic `*_typed(...)` helpers that dispatch on
`RealFftData`.

Apollo now also exposes explicit `*_f16` helpers for real-domain FFT storage. The current Python
binding still accepts `float32` buffers for `mixed_precision` and quantizes through the half-storage
path internally, because the active `numpy` crate does not expose `half::f16` as a native element
type in this environment. Those suffixed helpers remain as compatibility shims; the maintainable
Rust surface is the generic typed API.

## Design Rules

- `apollo/` is intentionally **not** a member of the root `d:\\kwavers\\Cargo.toml` workspace.
- Shared transform invariants live in one place and are re-exported instead of duplicated.
- `kwavers` consumes Apollo FFT/NUFFT through compatibility re-exports instead of owning reusable transform implementations.
- Solver-specific spectral helpers remain in `kwavers` until they prove broadly reusable.

## References

- [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- [`docs/THEORY.md`](./docs/THEORY.md)
- [`docs/VALIDATION.md`](./docs/VALIDATION.md)
- [`docs/MIGRATION_KWAVERS.md`](./docs/MIGRATION_KWAVERS.md)
