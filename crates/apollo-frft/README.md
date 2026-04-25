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

The implementation is a direct `O(n^2)` reference surface. Future acceleration
must preserve the documented rotation limits and parity against this kernel.

## Verification

The crate verifies identity order, continuity near the centered DFT boundary,
integer-order inverse reconstruction, caller-owned inverse parity, and invalid
plan rejection.
