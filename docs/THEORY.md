# Apollo FFT Theory

Apollo uses classical FFT constructions with a shared normalization convention:

- Forward transforms are unnormalized.
- Inverse transforms apply `1 / N` over the transformed domain.

## Algorithms

- Cooley-Tukey factorization for composite sizes.
- Bluestein or Chirp-Z reduction for prime-heavy sizes in the GPU path.
- Real-to-complex half-spectrum packing that preserves Hermitian symmetry.

## Invariants

- A real-valued spatial signal produces Hermitian-symmetric spectral coefficients.
- The Nyquist bin is real-valued when the corresponding dimension is even.
- Parseval-style energy checks are evaluated under Apollo's FFTW-compatible normalization.

## Numerical Notes

- Roundtrip stability is assessed with relative and absolute tolerances.
- GPU parity compares against the CPU plan family rather than raw bitwise identity.

