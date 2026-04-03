# kwavers Migration Notes

## Phase 2

- `kwavers::math::fft::Fft1d` -> `apollofft::plan::FftPlan1D`
- `kwavers::math::fft::Fft2d` -> `apollofft::plan::FftPlan2D`
- `kwavers::math::fft::Fft3d` -> `apollofft::plan::FftPlan3D`
- `kwavers::math::fft::gpu_fft::GpuFft3d` -> `apollofft_wgpu::GpuFft3d`
- `kwavers::math::fft::get_fft_for_grid` -> `apollofft::get_fft_for_grid`
- `kwavers::math::fft::nufft::*` -> `apollofft::nufft::*`

## Compatibility Strategy

`kwavers::math::fft` remains the compatibility namespace, but reusable transform ownership moved to Apollo:

- `kwavers::math::fft::mod` re-exports Apollo FFT plans, caches, and helpers.
- `kwavers::math::fft::nufft` is now a pure Apollo re-export surface.
- `kspace` and `shift_operators` stay local because they remain solver-specific.

## WGPU Semantics

Apollo Stage 2 removes the CPU delegation path from the WGPU crate. `apollofft-wgpu` now reports truthful capability boundaries:

- 3D GPU FFT is implemented.
- 1D and 2D WGPU planning are not exposed in this stage.
- Python capability reporting only advertises `wgpu` when the host can actually acquire a usable adapter/device.
