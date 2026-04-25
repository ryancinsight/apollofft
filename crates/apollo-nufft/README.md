# Apollo NUFFT

`apollo-nufft` owns non-uniform Fourier transform logic for Apollo. Dense FFT
execution is delegated to `apollo-fft`; NUFFT-specific spreading,
interpolation, and deconvolution remain in this crate.

## Architecture

```text
src/
  domain/          uniform-domain/grid metadata
  application/     1D and 3D Type-1/Type-2 NUFFT plans
  infrastructure/  Kaiser-Bessel kernels and Fourier-grid helpers
  verification/    analytical and fast/direct parity tests
```

`NufftPlan1D` and `NufftPlan3D` own oversampling shape, kernel width,
Kaiser-Bessel parameters, deconvolution factors, and reusable FFT plans.

## Mathematical Contract

Type-1 maps non-uniform samples to uniform Fourier modes:

```text
f_k = sum_j c_j exp(-2*pi*i*k*x_j/L)
```

Type-2 maps uniform Fourier modes back to non-uniform positions. Fast execution
uses Kaiser-Bessel spreading/interpolation on an oversampled grid followed by
Apollo FFT execution and deconvolution.

## Verification

Tests cover exact DC invariants, fast/direct agreement for fixed inputs,
Kaiser-Bessel non-negativity and peak behavior, `I_0` reference values, Fourier
transform limits, signed index mapping, and 3D finite-output behavior.
