# Apollo DHT

`apollo-dht` owns the discrete Hartley transform for real-valued Apollo
signals.

## Architecture

```text
src/
  domain/          length contracts, errors, and Hartley spectrum storage
  application/     reusable DHT plan
  infrastructure/  real cas-kernel execution
  verification/    analytical, inverse, and property tests
```

`DhtPlan` is the single source of truth for validated signal length and
execution. Domain storage remains real-valued; no complex FFT dependency is
introduced for production execution.

## Mathematical Contract

The DHT computes

```text
H[k] = sum_n x[n] cas(2*pi*k*n/N), cas(theta) = cos(theta) + sin(theta)
```

The Hartley kernel is self-inverse up to scale:

```text
DHT(DHT(x)) = N x
```

so inverse execution reuses the same transform and applies `1 / N`.

## Verification

Tests cover impulse response, constant-signal DC behavior, Parseval scaling,
double-transform scaling, inverse execution, invalid contracts, and randomized
roundtrips.
