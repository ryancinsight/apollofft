# Apollo Radon

`apollo-radon` owns discrete parallel-beam Radon projections and CT-style
filtered backprojection.

## Architecture

```text
src/
  domain/          geometry contracts, errors, and sinogram storage
  application/     reusable Radon plan
  infrastructure/  forward, adjoint, and ramp-filter kernels
  verification/    projection, adjoint, and filter tests
```

`RadonPlan` owns validated geometry and delegates computation to crate-local
forward, adjoint, and filtering kernels.

## Mathematical Contract

The forward model treats each pixel value as a point mass at the pixel center
and deposits it onto detector bins with linear weights. Backprojection applies
the same weights in reverse, giving the discrete adjoint identity:

```text
<R f, p> = <f, R* p>
```

Filtered backprojection applies a ramp filter to each projection before the
adjoint step.

## Verification

Tests cover axis-aligned row/column sums, adjoint identity, detector mass
conservation, ramp-filter DC removal, caller-owned filter parity, and invalid
shape/geometry contracts.
