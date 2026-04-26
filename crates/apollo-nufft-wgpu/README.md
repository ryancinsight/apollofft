# Apollo NUFFT WGPU

`apollo-nufft-wgpu` owns the WGPU backend boundary for Apollo NUFFT execution.

## Architecture

```text
src/
  domain/          backend capability and error contracts
  application/     NUFFT WGPU plan descriptors
  infrastructure/  WGPU device acquisition, direct kernels, and fast gridding kernels
  verification/    capability, contract, and CPU parity tests
```

The crate depends inward on `apollo-nufft` for mathematical metadata. It does
not own dense FFT contracts and does not place NUFFT logic in
`apollo-fft-wgpu`.

## Execution Contract

The crate implements exact direct Type-1 and Type-2 NUFFT summations on WGPU
for 1D and 3D.

The crate also implements fast Kaiser-Bessel Type-1 and Type-2 gridding on
WGPU for 1D and 3D. The fast paths perform GPU spreading or interpolation,
dispatch oversampled FFTs through `apollo-fft-wgpu`, and apply GPU
deconvolution against the same Kaiser-Bessel metadata used by `apollo-nufft`.
The 3D fast grid uses radix-2 oversampled dimensions large enough to avoid
periodic overlap of the compact Kaiser-Bessel support.

Shared fast-path bind-group layouts keep inactive entry-point bindings as
device-only placeholder storage buffers. The backend does not allocate or
upload host-side zero vectors for bindings that a selected shader entry point
does not read.

## Verification

Tests cover capability truthfulness, descriptor metadata preservation, input
length and mode-shape rejection, direct Type-1/Type-2 1D/3D parity against
`apollo-nufft` exact references, and fast Type-1/Type-2 1D/3D parity against
`apollo-nufft` gridded references.
