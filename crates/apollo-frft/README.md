# Apollo FrFT

`apollo-frft` owns the fractional Fourier transform reference implementation
for Apollo.

## Architecture

```text
src/
  domain/          length/order validation and error contracts
  application/     reusable FrFT plan and direct execution policy
  infrastructure/  CPU convenience wrapper
```

`FrftPlan` is the single source of truth for transform length, fractional order,
rotation parameters, and direct-kernel execution.

## Mathematical Contract

The FrFT of order `a` rotates a signal in the time-frequency plane by

```text
alpha = a*pi/2
```

Integer quarter rotations are exact discrete operators:

- `a = 0 mod 4`: identity
- `a = 1 mod 4`: centered unitary DFT
- `a = 2 mod 4`: reversal
- `a = 3 mod 4`: centered unitary inverse DFT

Non-integer orders use the direct cotangent/cosecant kernel over centered
discrete coordinates.

## Execution Surfaces

- `forward` and `inverse` allocate returned arrays.
- `forward_into` and `inverse_into` write into caller-owned output buffers.
- `forward_typed_into` and `inverse_typed_into` support `Complex64`,
  `Complex32`, and mixed `[f16; 2]` storage profiles.

## Precision Contract

Typed execution uses Apollo's shared `PrecisionProfile` contract:

- `HIGH_ACCURACY_F64`: `Complex64` storage.
- `LOW_PRECISION_F32`: `Complex32` storage.
- `MIXED_PRECISION_F16_F32`: `[f16; 2]` storage, with lane 0 as real and lane
  1 as imaginary.

Lower storage profiles reuse the authoritative `Complex64` FrFT plan and
quantize once at the storage boundary. Profile/storage mismatch is rejected
with `FrftError::PrecisionMismatch`.

The implementation is a direct `O(n^2)` reference surface. Future acceleration
must preserve the documented rotation limits and parity against this kernel.

## Verification

The crate verifies identity order, continuity near the centered DFT boundary,
integer-order inverse reconstruction, caller-owned inverse parity, and invalid
plan rejection. Typed tests cover `Complex64`, `Complex32`, mixed `[f16; 2]`,
and profile mismatch rejection.
