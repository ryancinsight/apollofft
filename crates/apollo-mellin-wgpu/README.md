# Apollo Mellin WGPU

`apollo-mellin-wgpu` owns the WGPU backend boundary for Apollo Mellin execution.

## Architecture

``text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors
  infrastructure/  WGPU device acquisition boundary
  verification/    capability and descriptor tests
``

The crate depends inward on `apollo-mellin` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides:

- forward Mellin log-frequency spectrum execution on WGPU for real 1D `f32` signals
- truthful capability reporting that marks forward support present and inverse support absent
- metadata-preserving plan descriptors carrying sample count and Mellin scale grid

The GPU path mirrors the owning CPU crate's definition: log-resampling onto the
validated scale grid followed by the direct log-frequency Mellin spectrum sum.
Inverse or adjoint execution remains unsupported until the owning `apollo-mellin`
crate defines that contract.

## Verification

Tests cover capability reporting, plan metadata preservation, invalid plan and
signal-domain rejection, and CPU parity for forward spectrum execution when a
WGPU adapter/device is available.
