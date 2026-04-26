# Apollo GFT

`apollo-gft` owns graph Fourier transforms over real weighted undirected
graphs.

## Architecture

```text
src/
  domain/          graph adjacency validation and error contracts
  application/     reusable graph Fourier plan
  infrastructure/  Laplacian and eigensystem construction
  verification/    spectral, roundtrip, and property tests
```

`GftPlan` builds the combinatorial Laplacian and stores the orthonormal
eigenvector basis as the transform matrix.

Typed execution uses Apollo's shared precision profile contract:

- `HIGH_ACCURACY_F64`: `f64` storage and owner `f64` graph-basis multiply.
- `LOW_PRECISION_F32`: `f32` storage converted through the owner path and
  quantized once into the caller-owned output.
- `MIXED_PRECISION_F16_F32`: `f16` storage converted through the owner path and
  quantized once into the caller-owned output.

Profile/storage mismatches return `GftError::PrecisionMismatch`.

## Mathematical Contract

For symmetric adjacency `A`, degree matrix `D`, and Laplacian `L = D - A`, the
real-symmetric eigendecomposition gives `L = U Lambda U^T` with `U^T U = I`.
The graph Fourier transform is

```text
X = U^T x
x = U X
```

so inverse reconstruction follows from orthonormality.

## Verification

Tests cover invalid graph contracts, known two-vertex spectra, zero constant
mode for a path graph, eigenbasis orthonormality, weighted graph roundtrips, and
random graph roundtrips. Typed tests cover `f64`, `f32`, mixed `f16`, inverse
roundtrip, caller-owned parity, and precision/profile mismatch rejection.
