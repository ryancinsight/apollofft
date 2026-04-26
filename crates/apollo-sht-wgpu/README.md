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
matrix summation on WGPU. `apollo-sht` remains the mathematical SSOT for
Gauss-Legendre quadrature nodes and weights. The WGPU backend receives those
grid samples, evaluates associated Legendre recurrence and spherical harmonic
basis values in a compute pass, then consumes the generated basis buffer in the
matrix-reduction pass.

The basis buffer is allocated directly as GPU storage and is written by the
basis-generation pass. It is not initialized through a host-side zero vector, so
execution avoids an O(mode_count * sample_count) CPU allocation and upload.

## Verification

Tests cover capability truthfulness, descriptor metadata preservation, invalid
plan rejection, sample-shape rejection, forward coefficient parity, and inverse
sample parity against `apollo-sht::ShtPlan`.
