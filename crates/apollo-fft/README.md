# Apollo FFT

`apollo-fft` owns Apollo's dense CPU Fourier transform implementation, shared
shape contracts, backend abstractions, and cache-backed plan surfaces.

## Architecture

```text
src/
  domain/          backend, error, and shape contracts
  application/     FFT plans and plan cache orchestration
  infrastructure/  CPU backend transport
```

The dense FFT crate is the single source of truth for 1D, 2D, and 3D uniform
FFT plans. NUFFT and SFT logic live in their own crates.

## Mathematical Contract

The forward complex FFT computes

```text
X[k] = sum_n x[n] exp(-2*pi*i*k*n/N)
```

The inverse computes the conjugate-sign transform and applies Apollo's selected
normalization. The kernel strategy auto-selects radix-2 Cooley-Tukey for
power-of-two lengths and Bluestein chirp-Z for arbitrary lengths. The direct
DFT kernel remains a crate-local reference for verification.

2D and 3D plans execute separable axis passes. Contiguous row/depth-axis passes
operate directly on backing-slice chunks with Rayon, avoiding full-field
lane-copy vectors and scatter copies. Non-contiguous axes still gather one lane
buffer per lane before scattering because ndarray strides are not contiguous.

## Verification

Tests cover analytical small transforms, radix-2 and Bluestein parity against
direct DFT, inverse roundtrips, Parseval-style energy checks, linearity,
caller-owned output paths, shape rejection, precision profile behavior, and
2D/3D separable axis execution.
