# Apollo SFT WGPU

`apollo-sft-wgpu` owns the WGPU backend boundary for Apollo SFT execution.

## Architecture

```text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors
  infrastructure/  WGPU device acquisition and direct DFT kernel
  verification/    capability, descriptor, and CPU parity tests
```

The crate depends inward on `apollo-sft` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

The current implementation computes the dense DFT on WGPU, projects the dense
spectrum into the `apollo-sft` sparse-spectrum domain by deterministic top-k
magnitude selection, and reconstructs from sparse spectra with the normalized
inverse dense DFT.

## Verification

Tests cover capability truthfulness, descriptor metadata preservation, invalid
plan rejection, input-shape rejection, forward sparse support/coefficients
against `SparseFftPlan`, and inverse sparse reconstruction against the CPU
owner.
