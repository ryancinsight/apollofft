# Apollo FFT WGPU

`apollo-fft-wgpu` owns the WGPU dense FFT backend boundary for Apollo.

## Architecture

```text
src/
  domain/          GPU capability and error contracts
  application/     dense FFT backend adapter surface
  infrastructure/  GPU FFT dispatch, pipeline, strategy, workspace, shaders
```

The crate isolates device/queue ownership and dense GPU FFT pipeline setup from
`apollo-fft`. CPU FFT algorithms remain independent of backend integration, and
NUFFT WGPU execution belongs in `apollo-nufft-wgpu`.

## Execution Contract

The backend advertises GPU capability when WGPU resources are available. Device
limit checks reject invalid dimensions and workspace sizes before pipeline
execution.

## Verification

Tests cover axis strategy selection, workspace geometry, power-of-two rounding,
valid small shapes, zero-size rejection, and device-limit rejection.
