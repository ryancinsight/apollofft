# ADR 0001: Typed Workspace Reuse and Alias Removal

## Status

Accepted for `apollo-stft` 0.2.x.

## Intent

Remove repeated typed bridge allocations from STFT typed execution while keeping
one canonical owner-precision frame/FFT/WOLA implementation. Remove deprecated
allocating alias methods that duplicate `forward` and `inverse`. Reuse inverse
WOLA work buffers across repeated calls.

## Constraints

- `forward_typed_into` and `inverse_typed_into` must remain the typed public
  entry points.
- f64/Complex64 owner arithmetic remains the current STFT numerical contract.
- Public root re-exports for typed storage traits remain available.
- Internal compatibility aliases and forwarding wrappers are not retained.
- The 1D plan module stays below the 500-line structural limit.

## Selected Approach

Typed forward and inverse use per-thread f64/Complex64 workspaces sized to the
largest signal and spectrum executed on that thread. The inverse owner path uses
per-thread frame, complex, overlap, and weight workspaces sized to the largest
WOLA execution on that thread. The generic typed wrappers only perform storage
conversion and call non-generic slice-level owner kernels, which centralizes
frame/window/FFT/WOLA logic and limits repeated monomorphized bodies.
Storage/profile traits move into `stft::storage` so the 1D plan owns execution
rather than storage representation.

## Alternatives Rejected

- Keep allocating `Array1` bridge values: preserves the old implementation but
  keeps O(N) heap allocation per typed call.
- Add specialized f32/f16 STFT kernels: duplicates frame and FFT orchestration
  without a separate native-precision STFT numerical contract.
- Retain `forward_inplace` and `inverse_inplace` as forwarding aliases:
  preserves deprecated API surface and contradicts the cleanup policy.

## Failure Modes

- Thread-local scratch retains peak per-thread capacity until thread exit.
- Recursive same-thread typed STFT calls would contend for the same scratch
  cells and panic through `RefCell` borrow checking.
- Recursive same-thread inverse STFT calls would contend for the WOLA scratch
  cells and panic through `RefCell` borrow checking.

## Verification Plan

- Check `apollo-stft` and downstream validation compilation.
- Run targeted repeated-call typed and inverse WOLA workspace tests and full
  STFT unit/property tests.
- Scan source for removed production typed bridge allocation and deprecated
  alias patterns.
- Keep crate version, README, changelog, backlog, checklist, and gap audit in
  sync with the pre-1.0 breaking change.
