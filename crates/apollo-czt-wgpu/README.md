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
- direct complex 1D **inverse CZT** execution on WGPU via the adjoint formula
  `x[n] = (A^n / N) · sum_k X[k] · W^{-nk}` (exact for unitary DFT parameters)
- truthful capability reporting: both forward and inverse marked supported
- metadata-preserving plan descriptors carrying input length, output length, and spiral parameters

The GPU forward path evaluates the direct CZT definition
`X[k] = sum_n x[n] a^-n w^(nk)` on the device. The GPU inverse requires a
square plan (M == N); `WgpuError::LengthMismatch` is returned for non-square
plans.

## Verification

Tests cover capability reporting, plan metadata preservation, invalid length and
parameter rejection, CPU parity against `apollo-czt` direct forward execution,
inverse roundtrip at DFT parameters, and rejection of non-square plans.
