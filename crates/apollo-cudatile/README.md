# Apollo Cudatile

`apollo-cudatile` owns the CUDA tile backend boundary for Apollo FFT.

## Architecture

```text
src/
  domain/          device and launch configuration contracts
  application/     backend adapter surface
  infrastructure/  runtime integration boundary
```

The crate preserves backend separation while CUDA runtime integration remains
behind this boundary. CPU FFT and transform crates do not depend on CUDA
details.

## Execution Contract

`CudatileBackend` reports backend unavailability unless a concrete runtime
integration is provided. This prevents silent CPU fallback from being reported
as CUDA execution.

## Verification

The current crate-level contract is compile-time API compatibility with the
shared `FftBackend` trait and explicit backend-unavailable errors.
