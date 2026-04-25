# Apollo GFT WGPU

`apollo-gft-wgpu` owns the WGPU backend boundary for Apollo GFT execution.

## Architecture

```text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors carrying graph basis data
  infrastructure/  WGPU device acquisition and graph-basis kernel
  verification/    capability, contract, and CPU-parity tests
```

The crate depends inward on `apollo-gft` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides truthful forward and inverse capability
reporting, basis-carrying plan descriptors, and a direct real graph Fourier
basis kernel over the implemented `f32` surface.

Spectral-basis construction remains owned by `apollo-gft`; this crate executes
only against a precomputed orthonormal basis supplied by the plan.

## Verification

Tests cover capability truthfulness, plan metadata preservation, invalid-plan
rejection, input-length validation, forward CPU parity, inverse CPU parity, and
GPU roundtrip recovery.
