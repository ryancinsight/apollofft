# Apollo STFT WGPU

`apollo-stft-wgpu` owns the WGPU backend boundary for Apollo STFT execution.

## Architecture

```text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors for frame and hop metadata
  infrastructure/  WGPU device acquisition and forward STFT kernel
  verification/    capability, contract, and CPU-parity tests
```

The crate depends inward on `apollo-stft` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides truthful forward-only capability reporting,
metadata-preserving plan descriptors, and a forward Hann-windowed STFT kernel
over the implemented `f32` signal / complex-spectrum surface.

Inverse STFT is unsupported because GPU weighted overlap-add reconstruction is
not implemented on this surface.

## Verification

Tests cover capability truthfulness, plan metadata preservation, invalid-plan
rejection, inverse rejection, and forward CPU parity against
`apollo-stft::StftPlan::forward`.
