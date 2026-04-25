# Apollo FrFT WGPU

`apollo-frft-wgpu` owns the WGPU backend boundary for Apollo FrFT execution.

## Architecture

``text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors
  infrastructure/  WGPU device acquisition boundary
  verification/    capability and descriptor tests
``

The crate depends inward on `apollo-frft` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides device acquisition, truthful capability reporting,
and metadata-preserving plan descriptors. Numerical kernels must be added here before
this crate reports execution support.

## Verification

Tests cover unsupported execution capabilities, descriptor metadata preservation, and error text.
