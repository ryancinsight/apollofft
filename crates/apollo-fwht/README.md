# Apollo FWHT

`apollo-fwht` owns the fast Walsh-Hadamard transform implementation for Apollo.

## Architecture

```text
src/
  domain/          length validation and error contracts
  application/     reusable FWHT plan and in-place execution policy
  infrastructure/  CPU convenience wrappers
```

The plan is the single source of truth for the power-of-two length contract.
The kernel owns the butterfly schedule and selects Rayon parallel chunks for
large stages.

## Mathematical Contract

For `n = 2^m`, the transform applies the Sylvester Hadamard matrix `H_n`.
The entries are

```text
H_n[i,j] = (-1)^(popcount(i & j))
```

The identity `H_n H_n = n I` makes the unnormalized FWHT self-inverse up to
division by `n`. Apollo therefore uses the same butterfly kernel for forward
and inverse execution, with inverse scaling applied after the second transform.

## Execution Surfaces

- `forward` and `inverse` allocate returned arrays.
- `forward_into` and `inverse_into` copy into caller-owned output and then run
  in place.
- `forward_inplace` and `inverse_inplace` execute with `O(1)` auxiliary storage.
- Complex variants use the same butterfly topology over `Complex64`.

## Verification

The crate verifies analytical two-point output, real and complex roundtrips,
caller-owned parity against allocating paths, invalid length rejection, and
property-based roundtrips over power-of-two lengths.
