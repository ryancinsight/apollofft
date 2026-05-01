# ADR: Bluestein/Chirp-Z Non-Power-of-Two STFT GPU Path

## Status
Accepted — 2026-05-01

## Context

`apollo-stft-wgpu` currently enforces `frame_len.is_power_of_two()` via the
`WgpuError::FrameLenNotPowerOfTwo` error variant. The CPU counterpart (`apollo-stft`)
accepts arbitrary `frame_len` via `FftPlan1D`, which auto-selects Radix-2 for PoT and
Bluestein/Chirp-Z for non-PoT. Callers that need e.g. `frame_len = 400` (20 ms at 20 kHz)
or `frame_len = 960` (20 ms at 48 kHz) cannot use the GPU path.

## Decision

Implement the Bluestein/Chirp-Z reduction for non-PoT `frame_len` in `apollo-stft-wgpu`,
mirroring the `ChirpData` / `chirp.wgsl` pattern already proven in `apollo-fft-wgpu`.

### Algorithm

Bluestein's identity maps an N-point DFT to a linear convolution (Rabiner, Schafer & Rader,
1969):

```
X[k] = W_N^{k²/2} · Σ_{n=0}^{N-1} (x[n]·W_N^{n²/2}) · W_N^{-(k-n)²/2}
```

where `W_N = exp(-2πi/N)` (forward) / `exp(+2πi/N)` (inverse).

This is a convolution `y = (premul(x)) * h`, where `h[n] = W_N^{-n²/2}`.
The convolution is computed via zero-padded Radix-2 FFT of length
`m ≥ 2N-1`, `m = 2^⌈log₂(2N-1)⌉`.

Forward five-pass GPU dispatch:
1. `stft_chirp_premul`:     x[n] ← x[n] · exp(+πi·n²/N)  (analysis window applied in same pass)
2. Radix-2 forward FFT over padded length M (reuses existing `stft_fwd_*` pipelines or a
   dedicated chirp Radix-2 pipeline with the chirp BGL).
3. `stft_chirp_pointmul`:   pointwise multiply by H (precomputed chirp kernel FFT).
4. Radix-2 inverse FFT over M.
5. `stft_chirp_postmul`:    y[k] ← y[k] · exp(+πi·k²/N)  then truncate to N outputs.

Inverse (ISTFT per frame) uses the conjugate twiddles:
`W_N = exp(+2πi/N)` → `exp(-πi·n²/N)` in premul/postmul; same convolution structure.

### Design

**`StftChirpData`** (new struct in `infrastructure/chirp.rs`):
```
StftChirpData {
    // Precomputed chirp kernel FFT stored on GPU (two f32 buffers, length M).
    h_fft_re:           wgpu::Buffer,
    h_fft_im:           wgpu::Buffer,
    // Padded working buffers (frame_count × M f32 each).
    chirp_re:           wgpu::Buffer,
    chirp_im:           wgpu::Buffer,
    // Bind group: [chirp_re rw, chirp_im rw, h_fft_re ro, h_fft_im ro] (group 0).
    data_chirp_bg:      wgpu::BindGroup,
    // Params bind group for chirp entries (group 1).
    params_bg:          wgpu::BindGroup,
    _params_buf:        wgpu::Buffer,
    // Precomputed Radix-2 stages over length M.
    radix2_fwd:         Vec<wgpu::BindGroup>,       // log₂M + 1 bind groups
    radix2_inv:         Vec<wgpu::BindGroup>,       // log₂M + 2 bind groups (with scale)
    _radix2_param_bufs: Vec<wgpu::Buffer>,
    n: u32,   // original frame_len
    m: u32,   // padded Radix-2 length
    frame_count: u32,
}
```

**Bind group layout** (group 0, 4 bindings):
- binding 0: `read_write` — chirp_re  (working re, length frame_count × M)
- binding 1: `read_write` — chirp_im  (working im, length frame_count × M)
- binding 2: `read` — h_fft_re (precomputed chirp kernel re, length M)
- binding 3: `read` — h_fft_im (precomputed chirp kernel im, length M)

**Shader** (`stft_chirp.wgsl`): four entry points:
- `stft_chirp_premul`   — premultiply + Hann window (forward) or conjugate premultiply (inverse)
- `stft_chirp_pointmul` — pointwise multiply chirp_re/im by h_fft_re/im
- `stft_chirp_postmul`  — postmultiply and truncate to N
- `stft_chirp_scale`    — 1/M normalisation (inverse path only)

**Conditional dispatch** in `StftGpuKernel::execute_forward_fft` and `execute_inverse`:
```rust
if frame_len.is_power_of_two() {
    // existing Radix-2 path (unchanged)
} else {
    // new Chirp-Z path: use StftChirpData constructed from (frame_count, frame_len)
}
```

`StftChirpData` is constructed lazily per unique `(frame_count, frame_len)` pair and cached
inside `StftGpuKernel` using a `HashMap<(usize, usize), StftChirpData>`. This mirrors the
`ChirpData` lifecycle in `apollo-fft-wgpu`'s `GpuFft3d`.

**`FrameLenNotPowerOfTwo`** error variant: **removed** from the public dispatch path (both
`device.rs` and `kernel.rs`). The variant is retained as dead-code for one release cycle
then removed in a subsequent [minor] cleanup. The variant's doc comment is updated to
indicate it is no longer returned.

### `StftGpuBuffers` impact

`StftGpuBuffers` currently sizes `re_scratch_buf` and `im_scratch_buf` for `frame_count ×
frame_len` elements. For a non-PoT frame_len, these buffers must accommodate
`frame_count × M` elements (M ≥ 2·frame_len). `StftGpuBuffers::new` must detect non-PoT
`frame_len` and allocate `frame_count × M` scratch bytes instead.

## Alternatives considered

1. **CPU fallback for non-PoT**: avoids GPU shader complexity but breaks the architectural
   contract that GPU dispatch is end-to-end GPU; mixed CPU/GPU pipelines introduce copy overhead.
2. **Expose only PoT frame_len**: preserves simplicity but limits domain coverage; the CPU
   path already supports arbitrary frame_len, creating an asymmetry.
3. **Pad frame_len to next PoT before dispatch**: simple but changes the STFT semantics
   (windowing at the padded length ≠ windowing at the requested length); breaks invariant
   `forward then inverse → original signal`.

Alternatives 1 and 3 were rejected as architectural violations. Alternative 2 is the status
quo; this ADR supersedes it.

## Consequences

- **Breaking change**: none. `FrameLenNotPowerOfTwo` was a public error variant but was only
  returned; its removal from the dispatch path is backward-compatible (callers handling it
  receive success instead of error). The variant itself remains in the enum (no removal of
  enum arm = no breaking change per SemVer).
- **API addition**: none (no new public surface; `StftChirpData` is `pub(crate)`).
- **Version**: 0.9.0 [minor] — additive non-breaking change; new trait behavior for non-PoT.
- **Verification**: property test derives expected spectrum via CPU `apollo-stft` at
  non-PoT frame_len (e.g. 400, 960) and asserts GPU output matches within TOL = 1e-4.

## References

- Rabiner, Schafer & Rader (1969) "The Chirp z-Transform Algorithm".
- Bluestein (1970) "A linear filtering approach to the computation of DFT".
- `apollo-fft-wgpu/src/infrastructure/gpu_fft/strategy.rs` — `ChirpData` definition.
- `apollo-fft-wgpu/src/infrastructure/shaders/chirp.wgsl` — GPU Bluestein kernels.
