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
random graph roundtrips.
