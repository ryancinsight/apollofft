# ADR: Unitary Discrete FrFT Algorithm Selection

## Status
Accepted

## Context
The `FrftPlan` direct O(N²) chirp kernel evaluates the Namias continuous-FrFT definition
on discrete centred coordinates. For non-integer orders `a` (α = aπ/2 not a multiple of π),
the kernel matrix M satisfies (M†M)[j,j] = 1/|sin α| ≠ 1, violating the unitary invariant.
Applications requiring norm-preserving FrFT (e.g., time-frequency rotation, FRFT-domain
filtering with energy conservation, unitarity testing) cannot use `FrftPlan` for non-integer
orders.

## Decision
Add `UnitaryFrftPlan` implementing the eigendecomposition-based unitary DFrFT:
  DFrFT_a(x) = V · diag(exp(−i·a·k·π/2)) · V^T · x
where V is the N×N orthonormal eigenvector matrix of the Grünbaum commuting matrix S:
  S[j,j] = 2·cos(2π(j−c)/N) − 2,   c = (N−1)/2
  S[j,(j+1) mod N] = S[(j+1) mod N, j] = 1
S is real symmetric with a palindrome diagonal (S[j,j] = S[N−1−j,N−1−j]), guaranteeing
eigenvectors are symmetric or antisymmetric under index reversal (Grünbaum 1982).

Unitarity proof: V is orthogonal (V^T V = I) and |exp(−i·a·k·π/2)| = 1 for real a,
therefore ‖DFrFT_a(x)‖₂ = ‖V diag(·) V^T x‖₂ = ‖x‖₂.

The GPU implementation (`UnitaryFrftGpuKernel`) precomputes V on CPU (O(N³) via nalgebra
SymmetricEigen), uploads as f32 storage buffer, and executes 3 sequential GPU submissions
(V^T·x, phase, V·c) with device.poll between each to enforce cross-workgroup storage ordering.

## Alternatives considered

1. **Ozaktas-Kutay-Mendlovic 1996 fast FrFT** (O(N log N)): Uses DFT + chirp multiplication;
   also non-unitary for non-integer orders in the discrete setting without a correction step.
   Rejected: same unitarity defect; faster but architecturally equivalent to the existing plan.

2. **DTFT-based unitary approximation**: Approximates the continuous unitary FrFT via DTFT;
   loses the exact discrete algebraic properties (additivity, exact reversal at order 2).
   Rejected: does not satisfy DFrFT_2(x)[k] = x[N−1−k] exactly.

3. **Sampling-based approach (Ozaktas et al. 2000 book)**: Oversampled DFT-based method;
   asymptotically unitary but not exactly norm-preserving for finite N.
   Rejected: not analytically exact for small N.

## Consequences

- **Positive**: Provably unitary for all real orders and all N; algebraic properties
  (additivity, order-2 reversal, roundtrip exact inverse) hold to machine precision (rel_err < 1e-10 in f64).
- **Positive**: Construction is N-size-only (order-independent); same basis serves all orders.
- **Trade-off**: Construction O(N³) via dense SymmetricEigen; appropriate for N ≤ ~1000.
- **Trade-off**: Per-call O(N²) vs O(N log N) for the chirp alternative; for the typical N
  in FrFT applications (N ≤ ~8192) this is acceptable.
- **Test rationale**: Tests use algebraic properties (identity at a=0,4; reversal at a=2;
  additivity DFrFT_{a+b} = DFrFT_a ∘ DFrFT_b; norm preservation) rather than comparing to
  a specific centered-DFT reference, because the Grünbaum eigenbasis convention differs from
  common centered-DFT centering choices used in published tables.
- **GPU test tolerances**: f32 arithmetic with N=8..16 introduces ~N·ε_f32 ≈ 2×10⁻⁶ per
  matrix-vector product; two products compound to ~4×10⁻⁶; tests use 1e-4 for roundtrip and
  norm preservation to allow 25× safety margin.

## References
- Candan, Ç., Kutay, M. A., & Ozaktas, H. M. (2000). The discrete fractional Fourier transform.
  IEEE Trans. Signal Process., 48(5), 1329–1337.
- Grünbaum, F. A. (1982). The eigenvectors of the discrete Fourier transform.
  J. Math. Anal. Appl., 88(1), 355–363.