# Apollo DHT WGPU

`apollo-dht-wgpu` owns the WGPU backend boundary for Apollo DHT execution.

## Architecture

``text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors
  infrastructure/  WGPU device acquisition boundary
  verification/    capability and descriptor tests
``

The crate depends inward on `apollo-dht` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides:

- real 1D `f32` DHT forward execution on WGPU
- real 1D `f32` DHT inverse execution on WGPU
- truthful capability reporting tied to the implemented kernel surface
- metadata-preserving plan descriptors

The compute shader evaluates the Hartley sum
`H[k] = sum_n x[n] * (cos(theta) + sin(theta))` directly for each output index.
The inverse reuses the same transform and applies `1 / n` scaling, matching the
self-inverse Hartley contract in the owning CPU crate.

## Verification

Tests cover capability reporting, plan metadata preservation, invalid-length rejection,
and CPU parity for forward and inverse execution when a WGPU adapter/device is available.
