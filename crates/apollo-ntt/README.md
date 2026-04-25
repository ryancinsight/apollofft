# Apollo NTT

`apollo-ntt` owns the radix-2 number theoretic transform implementation for
Apollo.

## Architecture

```text
src/
  domain/          modulus, primitive-root, and error contracts
  application/     reusable NTT plan and execution policy
  infrastructure/  CPU convenience wrappers
```

`NttPlan` is the single source of truth for transform length, modulus,
primitive root, derived forward/inverse roots, inverse length factor, and stage
twiddles.

## Mathematical Contract

For a prime modulus `q`, power-of-two length `n`, and primitive root `g`, Apollo
derives

```text
omega = g^((q - 1) / n) mod q
```

The forward transform is

```text
X[k] = sum_j x[j] omega^(k j) mod q
```

The inverse uses `omega^-1` and multiplies by `n^-1 mod q`. Orthogonality of
finite root-of-unity sums gives exact recovery of each input residue modulo
`q`.

## Execution Surfaces

- `forward` and `inverse` allocate returned arrays.
- `forward_into` and `inverse_into` copy into caller-owned output and then run
  in place.
- `forward_inplace` and `inverse_inplace` execute with `O(1)` auxiliary storage.

All execution paths normalize input values into their residue class modulo the
plan modulus before applying butterflies.

## Verification

The crate verifies single-point behavior, small-vector roundtrip, caller-owned
parity, residue normalization, invalid length rejection, and property-based
roundtrips over supported power-of-two lengths.
