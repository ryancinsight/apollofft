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

## Mathematical Contract

Forward execution computes dense Fourier coefficients, orders them by
magnitude, and retains the configured top-`k` sparse support. Inverse execution
evaluates the inverse dense Fourier sum from retained sparse coefficients.

## Verification

Tests cover constructor metadata, invalid contracts, sparse spectrum insertion,
dominant coefficient retention, inverse reconstruction on retained support,
direct DFT reference parity, zero-signal behavior, pure-tone support, `k = n`,
and DC-only constant signals.
