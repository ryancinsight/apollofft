# Apollo SDFT WGPU

`apollo-sdft-wgpu` owns the WGPU backend boundary for Apollo SDFT execution.

## Architecture

```text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors for window and bin metadata
  infrastructure/  WGPU device acquisition and direct-bins kernel
  verification/    capability, contract, and CPU-parity tests
```

The crate depends inward on `apollo-sdft` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides truthful forward-only capability reporting,
metadata-preserving plan descriptors, and a direct-bin sliding DFT kernel over
the implemented `f32` real-window / complex-bin surface.

Inverse is unsupported because this crate exposes direct bin evaluation for a
current window rather than a streaming-state reconstruction contract.

## Verification

Tests cover capability truthfulness, plan metadata preservation, invalid-plan
rejection, window-length validation, and forward CPU parity against
`apollo-sdft::SdftPlan::direct_bins`.
