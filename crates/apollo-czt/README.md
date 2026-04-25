# Apollo CZT

`apollo-czt` owns the chirp z-transform implementation for Apollo.

## Architecture

```text
src/
  domain/          errors and validated contracts
  application/     reusable CZT plan and execution policy
  infrastructure/  CPU convenience transport
```

The application plan is the single source of truth for transform lengths,
spiral parameters, chirp factors, the convolution kernel, and the backing
Apollo FFT plan.

## Mathematical Contract

For input `x[0..n)`, output length `m`, and non-zero finite spiral parameters
`a` and `w`, the transform is

```text
X[k] = sum_n x[n] a^-n w^(n k)
```

`forward_direct` evaluates this definition directly in `O(nm)` time.
`forward` and `forward_into` use Bluestein's identity
`nk = (n^2 + k^2 - (k-n)^2) / 2` to reduce the transform to one zero-padded
convolution evaluated by `apollo-fft`.

## Verification

The crate validates:

- direct and Bluestein paths agree on analytical complex inputs
- invalid lengths and mismatched execution buffers are rejected
- caller-owned output matches the allocating fast path

Production CZT code depends on Apollo-owned FFT kernels. External FFT engines
belong only in `apollo-validation`.
