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
- `forward_typed_into` and `inverse_typed_into` support Apollo precision
  profiles without duplicating frame or FFT kernels.
- `forward_inplace` and `inverse_inplace` are compatibility aliases for the
  allocating paths.

Typed execution uses Apollo's shared precision profile contract:

- `HIGH_ACCURACY_F64`: `f64` signal storage and `Complex64` spectrum storage.
- `LOW_PRECISION_F32`: `f32` signal storage and `Complex32` spectrum storage,
  converted through the owner path and quantized once into caller-owned output.
- `MIXED_PRECISION_F16_F32`: `f16` signal storage and `[f16; 2]` spectrum
  storage, converted through the owner path and quantized once into
  caller-owned output.

Profile/storage mismatches return `StftError::PrecisionMismatch`.

## Verification

The crate verifies Hann symmetry, forward/inverse reconstruction,
caller-owned forward and inverse parity, invalid configuration rejection,
short-input rejection, and property-based reconstruction over deterministic
signals. Typed tests cover `f64`, `f32`, mixed `f16`, represented-input
spectrum parity, `f32` inverse roundtrip, and precision/profile mismatch
rejection.
