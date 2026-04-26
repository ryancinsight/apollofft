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

Typed execution supports real `f64`, `f32`, and mixed `f16` sample storage plus
complex `Complex64`, `Complex32`, and mixed `[f16; 2]` coefficient/sample
storage. The Gauss-Legendre quadrature, spherical harmonic basis, and synthesis
remain the authoritative `f64`/`Complex64` owner path; typed APIs convert at
the boundary and write into caller-owned arrays.

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
single-mode reconstruction, shape mismatches, small-degree roundtrip, typed
real and complex forward parity, typed inverse roundtrip, mixed `f16`
coefficient parity, and profile/storage mismatch rejection.
