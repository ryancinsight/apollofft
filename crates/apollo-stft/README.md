# Apollo STFT

`apollo-stft` owns short-time Fourier transform planning and execution for
Apollo.

## Architecture

```text
src/
  domain/          frame, hop, and execution error contracts
  application/     reusable STFT plan and overlap-add execution
  infrastructure/  CPU convenience wrappers
```

`StftPlan` is the single source of truth for frame length, hop length, Hann
window coefficients, frame count, and the backing Apollo FFT plan.

## Mathematical Contract

Forward STFT uses centered frames. Each frame is multiplied by the Hann window
and transformed by the Apollo FFT plan. Inverse STFT applies the inverse frame
FFT, multiplies by the same window, overlap-adds, and divides each sample by
the accumulated squared-window weight.

For every sample with non-zero weight,

```text
sum_m x[t] w[t - mH]^2 / sum_m w[t - mH]^2 = x[t]
```

This gives exact reconstruction in exact arithmetic for covered samples.

## Execution Surfaces

- `forward` and `inverse` allocate returned arrays.
- `forward_into` and `inverse_into` use caller-owned output buffers.
- `forward_inplace` and `inverse_inplace` are compatibility aliases for the
  allocating paths.

## Verification

The crate verifies Hann symmetry, forward/inverse reconstruction,
caller-owned forward and inverse parity, invalid configuration rejection,
short-input rejection, and property-based reconstruction over deterministic
signals.
