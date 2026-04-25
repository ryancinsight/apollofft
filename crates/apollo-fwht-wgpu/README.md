# Apollo FWHT WGPU

`apollo-fwht-wgpu` owns the WGPU backend boundary for Apollo FWHT execution.

## Architecture

``text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors
  infrastructure/  WGPU device acquisition boundary
  verification/    capability and descriptor tests
``

The crate depends inward on `apollo-fwht` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides:

- real 1D `f32` FWHT forward execution on WGPU
- real 1D `f32` FWHT inverse execution on WGPU
- truthful capability reporting tied to the implemented kernel surface
- metadata-preserving plan descriptors

The compute shader uses the Hadamard butterfly factorization directly. Each stage
dispatches one kernel over `n / 2` butterfly pairs. The inverse reuses the same
butterfly network and applies `1 / n` scaling after the final stage, matching the
CPU crate's normalized inverse contract.

## Verification

Tests cover capability reporting, plan metadata preservation, invalid-length rejection,
and CPU parity for forward and inverse execution when a WGPU adapter/device is available.
