# Apollo NUFFT WGPU

`apollo-nufft-wgpu` owns the WGPU backend boundary for Apollo NUFFT execution.

## Architecture

```text
src/
  domain/          backend capability and error contracts
  application/     NUFFT WGPU plan descriptors
  infrastructure/  WGPU device acquisition and direct Type-1/Type-2 kernels
  verification/    capability, contract, and CPU parity tests
```

The crate depends inward on `apollo-nufft` for mathematical metadata. It does
not own dense FFT contracts and does not place NUFFT logic in
`apollo-fft-wgpu`.

## Execution Contract

The current crate implements exact direct Type-1 and Type-2 NUFFT summations on
WGPU for 1D and 3D. Fast Kaiser-Bessel gridding remains future work until GPU
spreading/interpolation, oversampled FFT dispatch, and deconvolution are
implemented inside this crate.

## Verification

Tests cover capability truthfulness, descriptor metadata preservation, input
length and mode-shape rejection, and direct Type-1/Type-2 1D/3D parity against
`apollo-nufft` exact references.
