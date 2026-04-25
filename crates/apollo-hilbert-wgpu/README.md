# Apollo Hilbert WGPU

`apollo-hilbert-wgpu` owns the WGPU backend boundary for Apollo Hilbert execution.

## Architecture

``text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors
  infrastructure/  WGPU device acquisition boundary
  verification/    capability and descriptor tests
``

The crate depends inward on `apollo-hilbert` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides:

- forward analytic-signal execution on WGPU for real 1D `f32` signals
- forward Hilbert quadrature extraction on WGPU for real 1D `f32` signals
- truthful capability reporting that marks forward support present and inverse support absent
- metadata-preserving plan descriptors

The GPU path mirrors the owning CPU crate's definition: direct DFT, analytic-spectrum
masking, and inverse DFT. Inverse or adjoint execution remains unsupported until the
owning `apollo-hilbert` crate defines that contract.

## Verification

Tests cover capability reporting, plan metadata preservation, invalid-length rejection,
and CPU parity for analytic-signal and quadrature execution when a WGPU adapter/device
is available.
