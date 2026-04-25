# Apollo NTT WGPU

`apollo-ntt-wgpu` owns the WGPU backend boundary for Apollo NTT execution.

## Architecture

``text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors
  infrastructure/  WGPU device acquisition boundary
  verification/    capability and descriptor tests
``

The crate depends inward on `apollo-ntt` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides:

- forward modular NTT execution on WGPU
- inverse modular NTT execution on WGPU
- truthful capability reporting that marks both forward and inverse support present
- metadata-preserving plan descriptors carrying length, modulus, and primitive root

The GPU path evaluates the direct modular transform sum over the supported
32-bit modulus surface. This matches the owning CPU crate's residue-field
contract while remaining architecturally isolated inside the transform-specific
WGPU crate.

## Verification

Tests cover capability reporting, plan metadata preservation, invalid-plan rejection,
and CPU parity for forward and inverse execution when a WGPU adapter/device is
available.
