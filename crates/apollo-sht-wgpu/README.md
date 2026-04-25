# Apollo SHT WGPU

`apollo-sht-wgpu` owns the WGPU backend boundary for Apollo SHT execution.

## Architecture

```text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors
  infrastructure/  WGPU device acquisition and direct SHT kernels
  verification/    capability, descriptor, and CPU parity tests
```

The crate depends inward on `apollo-sht` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation executes forward and inverse complex SHT by direct
matrix summation on WGPU. Basis and quadrature values are derived from
`apollo-sht`, preserving that crate as the mathematical SSOT.

## Verification

Tests cover capability truthfulness, descriptor metadata preservation, invalid
plan rejection, sample-shape rejection, forward coefficient parity, and inverse
sample parity against `apollo-sht::ShtPlan`.
