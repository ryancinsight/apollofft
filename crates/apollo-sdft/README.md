# Apollo SDFT

`apollo-sdft` owns sliding DFT state for fixed-window streaming signals.

## Architecture

```text
src/
  domain/          window-length and bin-count contracts
  application/     reusable plan and streaming state
  infrastructure/  direct initialization and recurrence kernels
  verification/    direct-vs-recurrence and contract tests
```

`SdftPlan` owns validated window metadata and twiddle factors. `SdftState` owns
the current window, tracked bins, and update count.

## Mathematical Contract

For a window of length `N`, each update removes `x_old`, appends `x_new`, and
updates tracked bins by the sliding recurrence derived from the DFT definition.
The state is equivalent to recomputing direct DFT bins over the current window
after every update.

## Verification

Tests cover initial direct-bin equivalence, update recurrence equivalence,
zero-state behavior, update counting, invalid contracts, and direct DFT parity
after a full window of pushes.
