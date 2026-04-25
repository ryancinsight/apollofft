# Apollo QFT WGPU

`apollo-qft-wgpu` owns the WGPU backend boundary for Apollo QFT execution.

## Architecture

```text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors
  infrastructure/  WGPU device acquisition and direct QFT kernel
  verification/    capability, contract, and CPU-parity tests
```

The crate depends inward on `apollo-qft` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides truthful forward and inverse capability
reporting, metadata-preserving plan descriptors, and a direct dense unitary QFT
kernel over the implemented `f32` complex surface.

## Verification

Tests cover capability truthfulness, plan metadata preservation, invalid-plan
rejection, input-length validation, forward CPU parity, inverse CPU parity, and
GPU roundtrip recovery.
