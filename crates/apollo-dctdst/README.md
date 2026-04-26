# Apollo DCT/DST

`apollo-dctdst` owns real-to-real cosine and sine transform plans for Apollo.

## Architecture

```text
src/
  domain/          transform kind, length validation, and errors
  application/     reusable DCT/DST plan
  infrastructure/  direct analytical kernels
  verification/    value-semantic tests
```

`DctDstPlan` is the single source of truth for transform length and transform
kind. Direct kernels remain the authoritative production implementation until a
derived FFT acceleration is proven equivalent.

Typed caller-owned paths support high-accuracy `f64`, low-precision `f32`, and
mixed `f16` storage profiles. Lower storage profiles reuse the authoritative
`f64` DCT/DST kernel and quantize once into caller-owned output, preserving one
mathematical implementation across storage formats.

## Mathematical Contract

Apollo implements unnormalized DCT-II/DCT-III and DST-II/DST-III pairs. Under
these conventions:

```text
DCT-III(DCT-II(x)) = (N / 2) x
DST-III(DST-II(x)) = (N / 2) x
```

The plan inverse applies the `2 / N` scale to recover `x` in exact arithmetic.
The direct kernels evaluate the analytical finite cosine and sine projections.

## Execution Surfaces

- `forward` and `inverse` allocate returned vectors.
- `forward_into` and `inverse_into` write into caller-owned buffers.
- `forward_typed_into` and `inverse_typed_into` write `f64`, `f32`, or mixed
  `f16` storage into caller-owned buffers.
- Production execution selects the proven direct or FFT-derived fast kernel
  through the owning `DctDstPlan`.

## Verification

The crate verifies analytical two-point projections, DCT/DST inverse-pair
scaling, caller-owned inverse parity, length mismatch errors, one-point DCT-II,
dispatch parity against direct kernels, and property-based inverse-pair
roundtrips. Typed tests cover `f64`, `f32`, mixed `f16`, and precision/profile
mismatch rejection.
