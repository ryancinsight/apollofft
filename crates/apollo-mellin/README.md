# Apollo Mellin

`apollo-mellin` owns real Mellin moments and log-frequency Mellin spectra over
positive scale-domain signals.

## Architecture

```text
src/
  domain/          positive scale-range contracts and errors
  application/     reusable Mellin plan
  infrastructure/  log resampling, moments, and log-frequency kernels
  verification/    analytical integral and contract tests
```

`MellinPlan` is the authoritative scale configuration and execution surface.

Typed execution uses Apollo's shared precision profile contract:

- `HIGH_ACCURACY_F64`: `f64` input and output storage with owner `f64`
  log-resampling, moment, and spectrum kernels.
- `LOW_PRECISION_F32`: `f32` input and output storage converted through the
  owner path and quantized once into caller-owned real outputs.
- `MIXED_PRECISION_F16_F32`: `f16` input and output storage converted through
  the owner path and quantized once into caller-owned real outputs.

Profile/storage mismatches return `MellinError::PrecisionMismatch`.

## Mathematical Contract

For positive scale coordinate `r`, the Mellin moment is

```text
M(s) = int_a^b f(r) r^(s - 1) dr
```

The substitution `r = exp(u)` maps multiplicative scale changes to additive
translations in `u`, enabling Fourier analysis on a logarithmic grid.

## Verification

Tests cover constant and power-law analytical integrals, log-resampling
endpoints, uniform resampling, invalid scale contracts, and log-frequency DC
behavior. Typed tests cover `f64`, `f32`, mixed `f16`, represented-input
moments, represented-input spectra, and precision/profile mismatch rejection.
