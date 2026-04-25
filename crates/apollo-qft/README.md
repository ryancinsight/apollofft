# Apollo QFT

`apollo-qft` owns dense quantum Fourier transform plans for Apollo state
vectors.

## Architecture

```text
src/
  domain/          quantum state-dimension contracts and errors
  application/     reusable QFT plan
  infrastructure/  dense unitary kernel execution
  verification/    unitarity, roundtrip, and property tests
```

`QftPlan` caches the state dimension and twiddle factors used by forward and
inverse dense QFT execution.

## Mathematical Contract

For state vector `x` of length `N`, the unitary QFT is

```text
X[k] = N^(-1/2) sum_j x[j] exp(2*pi*i*j*k/N)
```

The inverse uses the conjugate phase. The transform preserves norm because its
matrix columns are orthonormal.

## Verification

Tests cover two-point analytical output, norm preservation, inverse roundtrip,
in-place/convenience parity, invalid contracts, unitary matrix columns, `N=1`,
non-power-of-two `N=3`, and randomized vectors.
