# Apollo NUFFT WGPU

`apollo-nufft-wgpu` owns the WGPU backend boundary for Apollo NUFFT execution.

## Architecture

```text
src/
  domain/         backend capability and error contracts
  application/    NUFFT WGPU plan descriptors
  infrastructure/ WGPU device acquisition, direct kernels, and fast gridding kernels
    shaders/      WGSL compute shaders (direct, fast 1D, fast 3D)
  verification/   capability, contract, and CPU parity tests
```

The crate depends inward on `apollo-nufft` for mathematical metadata and Kaiser-Bessel spreading parameters. It does not own dense FFT contracts; oversampled FFT dispatch goes through `apollo-fft-wgpu`. CPU NUFFT math and validation references remain in `apollo-nufft`.

## Capability Records

`NufftWgpuCapabilities` is the truthful WGPU NUFFT capability descriptor with the following fields:

- `device_available: bool` — whether a WGPU adapter and device can be acquired
- `supports_mixed_precision: bool` — whether mixed-precision (f16/f32/f64) typed storage dispatch is supported
- `default_precision_profile: PrecisionProfile` — default precision profile for GPU execution

Direct execution flags:

- `supports_type1_1d: bool` — direct exact Type-1 1D
- `supports_type2_1d: bool` — direct exact Type-2 1D
- `supports_type1_3d: bool` — direct exact Type-1 3D
- `supports_type2_3d: bool` — direct exact Type-2 3D

Fast gridded execution flags:

- `supports_fast_type1_1d: bool` — fast gridded Type-1 1D
- `supports_fast_type2_1d: bool` — fast gridded Type-2 1D
- `supports_fast_type1_3d: bool` — fast gridded Type-1 3D
- `supports_fast_type2_3d: bool` — fast gridded Type-2 3D

Constructor variants encode progressively larger capability surfaces: `detected`, `direct_type1`, `direct_type1_and_type2_1d`, `direct_all`, `direct_all_fast_1d`, `direct_all_fast_all`.

## Execution Paths

### Direct Exact

- **Type-1 1D** (`execute_type1_1d`): one GPU thread per Fourier bin, sums all non-uniform sample contributions.
- **Type-2 1D** (`execute_type2_1d`): one GPU thread per non-uniform position, sums all Fourier mode contributions.
- **Type-1 3D** (`execute_type1_3d`): one GPU thread per Fourier voxel, sums all non-uniform sample contributions.
- **Type-2 3D** (`execute_type2_3d`): one GPU thread per non-uniform position, sums all Fourier mode contributions.

### Fast Gridded

- **Type-1 1D** (`execute_fast_type1_1d`): Kaiser-Bessel spreading → oversampled forward FFT → deconvolution.
- **Type-2 1D** (`execute_fast_type2_1d`): deconvolution load → oversampled inverse FFT → Kaiser-Bessel interpolation.
- **Type-1 3D** (`execute_fast_type1_3d`): GPU separable Kaiser-Bessel spreading → 3D oversampled forward FFT → deconvolution.
- **Type-2 3D** (`execute_fast_type2_3d`): deconvolution load → 3D oversampled inverse FFT → Kaiser-Bessel interpolation.

The 3D fast grid uses radix-2 oversampled dimensions large enough to avoid periodic overlap of the compact Kaiser-Bessel support. Shared fast-path bind-group layouts keep inactive entry-point bindings as device-only placeholder storage buffers; the backend does not allocate or upload host-side zero vectors for bindings that a selected shader entry point does not read.

## GPU Buffer Reuse

`NufftGpuBuffers1D` and `NufftGpuBuffers3D` are pre-allocated buffer structs that hold persistent GPU allocations for position, value, deconvolution, split-grid (re/im), output, and staging buffers across repeated fast-path invocations.

- `execute_fast_type1_1d_with_buffers` / `execute_fast_type2_1d_with_buffers` — reuse `NufftGpuBuffers1D`
- `execute_fast_type1_3d_with_buffers` / `execute_fast_type2_3d_with_buffers` — reuse `NufftGpuBuffers3D`

These methods overwrite buffer contents in place and avoid per-call allocation, suitable for repeated execution with consistent problem dimensions.

## Debug Readbacks

The `debug-readbacks` feature enables `read_grid_1d` and `read_grid_3d` on `NufftGpuKernel`, which read back the split real/imaginary grid buffers as `Vec<Complex32>`. Intended for numerical triage only; adds GPU→CPU synchronization points unacceptable in production pipelines.

The `diagnostics` feature enables `execute_fast_type2_1d_with_diagnostics` and `execute_fast_type2_3d_with_diagnostics`, which capture `NufftGridSnapshot` intermediates (`after_load`, `after_ifft`) via `NufftType2GridDiagnostics`.

## Normalization Convention

- **Type-1 fast**: uses unnormalized forward FFT (`encode_forward_split`). No compensating scale factor is needed; the deconvolution kernel divides out the Kaiser-Bessel spreading footprint directly.
- **Type-2 1D fast**: the GPU IFFT (`encode_inverse_split`) applies a normalized transform that divides by `oversampled_len`. The CPU `type2_into` path multiplies by `oversampled_len` after its normalized IFFT to recover the unnormalized IDFT required by the KB interpolation kernel. The GPU path embeds this factor by pre-scaling the deconvolution coefficients by `oversampled_len` during load (`deconv[k] × m`), so after the normalized IFFT the grid values are equivalent to the unnormalized IDFT.
- **Type-2 3D fast**: uses the normalized 3D IFFT directly without pre-scaling, matching the CPU 3D normalization path.

## Mathematical Contract

- **Type-1**: maps non-uniform sample positions and values to uniform Fourier bins. Given non-uniform positions `x_j` and values `f_j`, computes `F_k = Σ_j f_j exp(2πi k x_j)` for integer mode indices `k`.
- **Type-2**: maps uniform Fourier modes to non-uniform positions. Given Fourier coefficients `F_k` and non-uniform positions `x_j`, computes `f_j = Σ_k F_k exp(2πi k x_j)`.

## Verification

Tests cover:

- Capability truthfulness: all direct and fast execution flags reflect actual kernel availability
- Descriptor metadata preservation: plan descriptors carry validated domain, oversampling, and kernel width
- Input rejection: length mismatch and mode-shape mismatch produce descriptive errors
- Direct Type-1/Type-2 1D/3D parity against `apollo-nufft` exact references
- Fast Type-1/Type-2 1D/3D parity against `apollo-nufft` gridded references
- Fast Type-2 1D normalization invariance: delta-function input at `k=0` yields constant output across all positions, verifying that the `1/m` pre-scaling compensates the normalized IFFT correctly