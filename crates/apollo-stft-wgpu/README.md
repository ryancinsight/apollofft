# Apollo STFT WGPU

`apollo-stft-wgpu` owns the WGPU backend boundary for Apollo STFT execution.

## Architecture

```text
src/
  domain/          backend capability and error contracts
  application/     WGPU plan descriptors for frame and hop metadata
  infrastructure/  WGPU device acquisition, forward and inverse FFT-accelerated STFT kernels
  verification/    capability, contract, CPU-parity, and roundtrip tests
```

The crate depends inward on `apollo-stft` for the mathematical transform contract.
It does not move CPU algorithms or validation references out of the owning transform crate.

## Execution Contract

Both forward and inverse paths are FFT-accelerated (Radix-2 DIT, O(N log N) per frame).
`frame_len` must be a power of two; non-power-of-two values return
`WgpuError::FrameLenNotPowerOfTwo` before any GPU dispatch.

### Forward STFT

Computes `X[m, k] = Σ_{n=0}^{N-1} w_a[n] · x[m·hop − N/2 + n] · exp(−2πi·k·n/N)` for each
frame `m` and bin `k`. Analysis window: Hann (`w_a[n] = 0.5 − 0.5·cos(2π·n/(N−1))`).
Centered frame placement with zero-padding outside signal bounds.

Dispatch sequence (all in one `CommandEncoder`; implicit per-pass memory barriers):
1. `stft_fwd_pack_window`: apply Hann window; pack to split re/im scratch (f32).
2. `stft_fwd_bitrev`: Cooley-Tukey bit-reversal permutation (batched, in-place).
3. `stft_fwd_butterfly` × log₂(N): Radix-2 DIT butterfly stages, DFT twiddle `exp(−2πi·k/N)`.
4. `stft_fwd_interleave`: split re/im scratch → interleaved `ComplexValue` output.

Formal basis: Cooley & Tukey (1965).

### Inverse STFT (WOLA)

Reconstructs `y[n]` from spectrum `X[m, k]` via weighted overlap-add normalised by
the sum of squared synthesis windows (Allen-Rabiner 1977, Theorem 1).

Dispatch sequence (same `CommandEncoder`):
1. `stft_deinterleave`: interleaved spectrum f32 → split re/im scratch.
2. `stft_bitrev`: bit-reversal permutation (batched).
3. `stft_butterfly` × log₂(N): Radix-2 DIT butterfly stages, IDFT twiddle `exp(+2πi·k/N)`.
4. `stft_scale_and_window`: scale by 1/N; apply Hann synthesis window → frame_data.
5. `stft_inverse_ola`: weighted overlap-add reconstruction.

COLA condition: Hann window with `hop_len = frame_len / 2` satisfies the overlap-add
normalisation property.

## Mixed Precision

Host storage accepts `Complex32` and `[f16; 2]` input/output. GPU arithmetic is `f32`.
`f16` values are promoted to `f32` at the host–GPU boundary before dispatch.
Profile mismatch returns `WgpuError::InvalidPrecisionProfile`.

## Verification

Tests cover capability truthfulness, plan metadata preservation, invalid-plan rejection,
`FrameLenNotPowerOfTwo` rejection for both forward and inverse paths, CPU parity for
forward FFT against `apollo-stft::StftPlan::forward`, WOLA roundtrip for COLA-compliant
parameter sets, large-frame roundtrip (frame_len=1024, 10 butterfly stages), and
typed mixed-precision dispatch.
