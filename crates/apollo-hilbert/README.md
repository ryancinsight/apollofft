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
random real-signal preservation.
