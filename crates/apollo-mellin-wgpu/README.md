# Apollo Mellin WGPU

`apollo-mellin-wgpu` owns the WGPU backend boundary for Apollo Mellin execution.

## Architecture

``text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors
  infrastructure/  WGPU device acquisition boundary
  verification/    capability and descriptor tests
``

The crate depends inward on `apollo-mellin` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides:

- forward Mellin log-frequency spectrum execution on WGPU for real 1D `f32` signals
- **inverse Mellin** execution on WGPU via two GPU passes:
  1. IDFT pass: spectrum → log-domain samples (`mellin_inverse_spectrum` kernel)
  2. Exp-resample pass: log-domain samples → linear output domain
     (`mellin_exp_resample` kernel)
- truthful capability reporting: both forward and inverse marked supported
- metadata-preserving plan descriptors carrying sample count and Mellin scale grid

The GPU forward path mirrors the owning CPU crate's definition: log-resampling
onto the validated scale grid followed by the direct log-frequency Mellin
spectrum sum. The GPU inverse reuses the same bind-group layout as the forward
resample pass (`binding 0`: read storage, `binding 1`: readwrite storage,
`binding 2`: uniform 32-byte params).

## Verification

Tests cover capability reporting, plan metadata preservation, invalid plan and
signal-domain rejection, CPU parity for forward spectrum execution, inverse
constant-signal roundtrip (ε < 5e-4), and invalid output-domain rejection.
