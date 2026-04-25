# Apollo SHT

`apollo-sht` owns spherical harmonic transforms for functions sampled on a
spherical surface.

## Architecture

```text
src/
  domain/          spherical grid metadata, errors, and coefficient storage
  application/     reusable SHT plan
  infrastructure/  Gauss-Legendre and spherical harmonic kernels
  verification/    quadrature, basis, and roundtrip tests
```

`ShtPlan` owns sampling metadata and transform degree bounds. Infrastructure
owns associated Legendre evaluation, normalization, and quadrature weights.

## Mathematical Contract

Complex spherical harmonics are

```text
Y_l^m(theta, phi) = N_lm P_l^m(cos(theta)) exp(i m phi)
```

with Condon-Shortley phase and orthonormal normalization. Gauss-Legendre
quadrature integrates the polar dimension and uniform azimuthal sampling
integrates Fourier modes.

## Verification

Tests cover known associated Legendre values, Gauss-Legendre weight sums and
polynomial exactness, invalid sampling, constant-surface coefficients,
single-mode reconstruction, shape mismatches, and small-degree roundtrip.
