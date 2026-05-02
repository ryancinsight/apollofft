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

The instantaneous frequency is derived from the analytic signal using the
complex-derivative formula

```text
f[n] = arg(conj(z[n]) · z[n+1]) / (2π)     cycles per sample
```

which avoids explicit phase unwrapping and is well-defined whenever
`|z[n]| > 0`. `AnalyticSignal::instantaneous_frequency()` returns a `Vec<f64>`
of length `N − 1`. For a discrete cosine at normalised frequency `k/N` the
result is constant and equal to `k/N` (verified by fixture 31 in
`apollo-validation`; reference: Boashash 1992, Proc. IEEE 80(4)).

## Verification

Tests cover cosine-to-sine quadrature, constant/DC behavior, even-length
Nyquist behavior, analytic real-part preservation, unit-cosine envelope,
instantaneous-frequency constant-tone, double-Hilbert negation, and
random real-signal preservation. Typed tests cover `f64`, `f32`, mixed `f16`,
caller-owned quadrature parity, analytic-signal real-part preservation, and
precision/profile mismatch rejection.
