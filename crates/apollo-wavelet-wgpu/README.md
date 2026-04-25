# Apollo Wavelet WGPU

`apollo-wavelet-wgpu` owns the WGPU backend boundary for Apollo Wavelet execution.

## Architecture

```text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors for Haar length and level metadata
  infrastructure/  WGPU device acquisition and Haar analysis/synthesis kernels
  verification/    capability, contract, analytical, and CPU-parity tests
```

The crate depends inward on `apollo-wavelet` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides truthful forward and inverse capability
reporting, metadata-preserving plan descriptors, and multi-level Haar DWT
analysis/synthesis kernels over the implemented `f32` surface.

Daubechies4 DWT and CWT remain CPU-only until their GPU contracts and parity
tests are implemented in this crate.

## Verification

Tests cover capability truthfulness, plan metadata preservation, invalid-plan
rejection, analytical Haar coefficients, roundtrip recovery, Parseval energy
preservation, and CPU parity against `apollo-wavelet::DwtPlan` for Haar.
