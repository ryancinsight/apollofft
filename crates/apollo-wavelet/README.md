# Apollo Wavelet

`apollo-wavelet` owns discrete and continuous wavelet transforms for Apollo
multiresolution analysis.

## Architecture

```text
src/
  domain/          wavelet descriptors, errors, and coefficient storage
  application/     reusable DWT and CWT plans
  infrastructure/  orthogonal filter banks and CWT mother-wavelet kernels
  verification/    analytical, boundary, and property tests
```

`DwtPlan` and `CwtPlan` are the authoritative execution surfaces. Domain types
define admissible wavelets and coefficient ownership; infrastructure kernels
only evaluate mathematical primitives.

Typed execution uses the same owner kernels for `f64`, `f32`, and mixed `f16`
storage. The typed DWT and CWT APIs convert represented input into the
authoritative `f64` arithmetic path and quantize once into caller-owned output
buffers, so storage precision does not create parallel algorithm families.

## Mathematical Contract

The DWT uses orthogonal analysis/synthesis filters with periodic boundaries. For
Haar and Daubechies-4, the synthesis filters are the quadrature mirror inverse
of the analysis filters, so multilevel inverse reconstruction recovers the
original power-of-two signal up to floating-point roundoff.

The CWT computes

```text
W_x(a, b) = a^(-1/2) sum_n x[n] psi((n - b) / a)
```

for positive scale `a`. Ricker and DC-corrected real Morlet wavelets are
zero-mean analysis kernels, so constant signals have no continuous-limit
wavelet response.

## Verification

The crate verifies analytical Haar coefficients, Haar/Daubechies-4 inverse
reconstruction, invalid contracts, CWT impulse localization, zero-signal
response, Morlet finite coefficients, Morlet zero-mean admissibility, typed
DWT/CWT parity for `f64`, represented-input parity for `f32` and mixed `f16`,
inverse DWT roundtrip for `f32`, shape rejection, and profile/storage mismatch
rejection.
