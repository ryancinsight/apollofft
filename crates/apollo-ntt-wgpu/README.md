# Apollo NTT WGPU

`apollo-ntt-wgpu` owns the WGPU backend boundary for Apollo NTT execution.

## Architecture

```text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors
  infrastructure/  WGPU device acquisition boundary
  verification/    capability and descriptor tests
```

The crate depends inward on `apollo-ntt` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation provides:

- forward modular NTT execution on WGPU
- inverse modular NTT execution on WGPU
- reusable `NttGpuBuffers` for repeated forward/inverse dispatch at one length
- exact quantized `u32` residue storage forward/inverse APIs
- truthful capability reporting that marks both forward and inverse support present
- metadata-preserving plan descriptors carrying length, modulus, and primitive root

The GPU path evaluates the direct modular transform sum over the supported
32-bit modulus surface. This matches the owning CPU crate's residue-field
contract while remaining architecturally isolated inside the transform-specific
WGPU crate.

Reusable execution uses the same direct kernel and bind-group layout as the
allocating path. `NttGpuBuffers` retains canonical residue scratch storage,
input/output device buffers, a staging buffer, and the bind group for one plan
length. The caller supplies the buffer object to avoid reallocating GPU buffers
and host readback vectors across repeated transforms.

The quantized storage API is integer narrowing, not floating mixed precision.
For the current WGPU surface, plans already reject moduli larger than
`u32::MAX`; therefore `u32` host residues represent every field element
losslessly. Values are canonicalized modulo `q` before dispatch and written back
as exact `u32` residues.

## Verification

Tests cover capability reporting, plan metadata preservation, invalid-plan rejection,
CPU parity for forward and inverse execution, reusable-buffer parity against the
allocating path, quantized `u32` parity against the `u64` allocating path, and
reusable-buffer length mismatch rejection when a WGPU adapter/device is
available.
