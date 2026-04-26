# Apollo Hilbert

`apollo-hilbert` owns Hilbert transforms and analytic-signal extraction for
real Apollo signals.

## Architecture

```text
src/
  domain/          signal length contracts and analytic-signal storage
  application/     reusable Hilbert plan
  infrastructure/  direct frequency-domain analytic masking
  verification/    quadrature, envelope, and property tests
```

`HilbertPlan` validates length and exposes Hilbert quadrature, analytic signal,
envelope, and phase surfaces.

Typed execution uses Apollo's shared precision profile contract:

- `HIGH_ACCURACY_F64`: `f64` input and quadrature storage with owner `f64`
  analytic masking.
- `LOW_PRECISION_F32`: `f32` input and quadrature storage converted through the
  owner path and quantized once into the caller-owned output.
- `MIXED_PRECISION_F16_F32`: `f16` input and quadrature storage converted
  through the owner path and quantized once into the caller-owned output.

Profile/storage mismatches return `HilbertError::PrecisionMismatch`.

## Mathematical Contract

The discrete Hilbert transform shifts positive-frequency components by
`-pi/2`, negative-frequency components by `+pi/2`, and leaves DC and Nyquist
components without quadrature contribution. The analytic signal is

```text
z[n] = x[n] + i H{x}[n]
```

with envelope `|z[n]|` and instantaneous phase `arg(z[n])`.

## Verification

Tests cover cosine-to-sine quadrature, constant/DC behavior, even-length
Nyquist behavior, analytic real-part preservation, unit-cosine envelope, and
random real-signal preservation. Typed tests cover `f64`, `f32`, mixed `f16`,
caller-owned quadrature parity, analytic-signal real-part preservation, and
precision/profile mismatch rejection.
