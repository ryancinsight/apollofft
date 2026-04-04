# Apollo Architecture

## Dependency Rules

- `domain` defines the single source of truth for shapes, normalization, capabilities, device descriptors, and error contracts.
- `application` owns orchestration, cache policy, reusable plans, and zero-allocation execution paths.
- `infrastructure` owns concrete adapters such as CPU, WGPU, and cudatile.
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
