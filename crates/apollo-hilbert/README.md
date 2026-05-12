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
envelope, and phase surfaces. The analytic-signal owner kernel applies the
frequency-domain mask in the forward FFT spectrum through Apollo FFT's
slice-level real forward plan, then runs the complex inverse in place on the
same buffer. `analytic_signal_into` lets callers supply that complex output
buffer directly.
Caller-owned quadrature execution writes the imaginary analytic component
directly into the supplied output slice through a reused per-thread complex
analytic scratch buffer instead of constructing an intermediate analytic vector
or quadrature vector.
Caller-owned envelope and phase execution use the same analytic scratch
discipline, and `AnalyticSignal` exposes caller-owned real, quadrature,
envelope, phase, and instantaneous-frequency projections.

Typed execution uses Apollo's shared precision profile contract:

- `HIGH_ACCURACY_F64`: `f64` input and quadrature storage with owner `f64`
  analytic masking.
- `LOW_PRECISION_F32`: `f32` input and quadrature storage converted through the
  owner path using per-thread bridge workspaces and quantized once into the
  caller-owned output.
- `MIXED_PRECISION_F16_F32`: `f16` input and quadrature storage converted
  through the owner path using per-thread bridge workspaces and quantized once
  into the caller-owned output.

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
precision/profile mismatch rejection. Workspace tests assert stable typed
bridge capacities across repeated calls and bitwise equal repeated output.
Direct kernel tests cover caller-owned analytic parity, caller-owned quadrature
parity, quadrature scratch capacity reuse, and output-length rejection.
Analytic-signal tests cover caller-owned observable projection parity and length
rejection.
