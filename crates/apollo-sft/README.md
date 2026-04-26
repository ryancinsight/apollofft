# Apollo SFT

`apollo-sft` owns sparse Fourier transform modeling and execution. Dense FFT
kernels remain in `apollo-fft`; sparse support selection and reconstruction
remain here.

## Architecture

```text
src/
  domain/          sparse spectrum storage and validation
  application/     sparse FFT plan and execution
  infrastructure/  dense reference kernel for verification
  verification/    sparse-support, inverse, and contract tests
```

`SparseFftPlan` is the plan-level SSOT for signal length, sparsity, bucket
count, trial count, and threshold.

Typed execution supports `Complex64`, `Complex32`, and mixed `[f16; 2]`
storage through one generic sparse API. The dense FFT, deterministic top-`k`
selector, and sparse inverse remain the `Complex64` owner path; typed storage
converts represented input into the owner path and quantizes only retained
coefficients or reconstructed samples at the API boundary.

## Mathematical Contract

Forward execution computes dense Fourier coefficients, orders them by
magnitude, and retains the configured top-`k` sparse support. Inverse execution
evaluates the inverse dense Fourier sum from retained sparse coefficients.

## Verification

Tests cover constructor metadata, invalid contracts, sparse spectrum insertion,
dominant coefficient retention, inverse reconstruction on retained support,
direct DFT reference parity, zero-signal behavior, pure-tone support, `k = n`,
DC-only constant signals, typed `Complex64` parity, represented-input
`Complex32` and mixed `[f16; 2]` parity, typed inverse roundtrip, sparse
frequency/value shape rejection, and profile/storage mismatch rejection.
