# Apollo Radon WGPU

`apollo-radon-wgpu` owns the WGPU backend boundary for Apollo Radon execution.

## Architecture

```text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors carrying Radon geometry
  infrastructure/  WGPU device acquisition and forward projection kernel
  verification/    capability, contract, and CPU-parity tests
```

The crate depends inward on `apollo-radon` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides truthful forward-only capability reporting,
geometry-preserving plan descriptors, and a real forward parallel-beam projection
kernel over `f32` images and sinograms.

Inverse execution, adjoint backprojection, and filtered backprojection remain
unsupported in this crate until the owning `apollo-radon` contract defines and
verifies their GPU surface.

## Verification

Tests cover capability truthfulness, plan metadata preservation, invalid-plan
rejection, shape validation, and forward CPU parity against `apollo-radon`.
