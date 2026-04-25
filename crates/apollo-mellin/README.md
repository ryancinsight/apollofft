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
behavior.
