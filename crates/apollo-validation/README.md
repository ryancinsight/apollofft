# Apollo Validation

`apollo-validation` owns cross-crate validation reports, optional external
reference comparison, and benchmark orchestration.

## Architecture

```text
src/
  domain/          validation report schema
  application/     suite composition and computed checks
  infrastructure/  optional NumPy and rustfft reference adapters
```

This crate is the only allowed production workspace boundary for Rust external
FFT reference usage. The current feature-gated Rust reference dependency is
`rustfft`; `realfft` is not present in the workspace dependency graph.

## Validation Contract

Validation reports include CPU FFT, GPU surface, NUFFT, optional external
comparison, published-reference fixtures, benchmarks, and environment sections.
Optional dependencies report availability and skip notes instead of silently
succeeding.

## Verification

Tests serialize the report and assert required schema sections, computed
value-semantic fields, and published-reference fixture values for FFT, DHT,
DCT-II, and DST-II.
