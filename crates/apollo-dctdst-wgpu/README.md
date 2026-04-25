# Apollo DCT/DST WGPU

`apollo-dctdst-wgpu` owns the WGPU backend boundary for Apollo DCT/DST execution.

## Architecture

``text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors
  infrastructure/  WGPU device acquisition boundary
  verification/    capability and descriptor tests
``

The crate depends inward on `apollo-dctdst` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides:

- real 1D `f32` DCT-II forward execution on WGPU
- real 1D `f32` DCT-III forward execution on WGPU
- real 1D `f32` DST-II forward execution on WGPU
- real 1D `f32` DST-III forward execution on WGPU
- normalized inverse recovery for DCT-II/DCT-III and DST-II/DST-III using the paired transform and `2 / n` scaling
- truthful capability reporting that marks both DCT and DST support present
- metadata-preserving plan descriptors carrying both length and transform kind

## Verification

Tests cover capability reporting, plan metadata preservation, invalid-length rejection,
and CPU parity for DCT-II/DCT-III/DST-II/DST-III forward and inverse execution
when a WGPU adapter/device is available.
