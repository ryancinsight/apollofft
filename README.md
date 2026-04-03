# Apollo FFT Workspace

Apollo is a standalone nested workspace for reusable FFT planning, execution, validation, and Python bindings.

Stage 2 moves Apollo beyond the initial compatibility cut:

- `apollofft` owns reusable CPU FFT and NUFFT implementations.
- `apollofft-wgpu` owns the real shader-backed 3D WGPU FFT path with radix-2 and Bluestein/Chirp-Z axis strategies.
- `apollo-validation` emits structured CPU, GPU, NUFFT, benchmark, and external-comparison reports, including direct parity against `rustfft` and NumPy when Python is available.
- `pyapollofft` exposes FFT, R2C/C2R, NUFFT, and backend capability introspection for Python callers.

## Crates

- `apollofft`: CPU FFT plans, cache management, shared types, and backend abstractions.
- `apollofft-wgpu`: Real WGPU backend and GPU parity surface.
- `apollofft-cudatile`: trait-compatible cudatile adapter surface.
- `apollo-validation`: parity, adversarial, benchmark, and external-reference runners.
- `pyapollofft`: PyO3 bindings, NumPy interop, and backend introspection.

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
