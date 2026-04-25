# Apollo CZT WGPU

`apollo-czt-wgpu` owns the WGPU backend boundary for Apollo CZT execution.

## Architecture

``text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors
  infrastructure/  WGPU device acquisition boundary
  verification/    capability and descriptor tests
``

The crate depends inward on `apollo-czt` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides:

- direct complex 1D forward CZT execution on WGPU for the `f32` kernel surface
- truthful capability reporting that marks forward support present and inverse support absent
- metadata-preserving plan descriptors carrying input length, output length, and spiral parameters

The GPU path evaluates the direct CZT definition
`X[k] = sum_n x[n] a^-n w^(nk)` on the device. Inverse or adjoint execution
remains unsupported because the owning `apollo-czt` crate does not yet define
an authoritative inverse CZT contract.

## Verification

Tests cover capability reporting, plan metadata preservation, invalid length and
parameter rejection, and CPU parity against `apollo-czt` direct forward execution
when a WGPU adapter/device is available.
