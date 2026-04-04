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
- `kwavers::math::fft` and `kwavers::math` now also re-export Apollo precision descriptors:
  - `PrecisionMode`
  - `StoragePrecision`
  - `ComputePrecision`
  - `PrecisionProfile`
- `kspace` and `shift_operators` stay local because they remain solver-specific.

## Mixed Precision Migration

Apollo keeps the legacy default path unchanged, so existing `kwavers` call sites continue using
the `high_accuracy` profile unless they opt in.

- Existing `Fft1d::new(...)`, `Fft2d::new(...)`, and `Fft3d::new(...)` calls still mean the
  Apollo `f64` reference path.
- New precision-aware construction uses `with_precision(...)`.
- New `*_f32` helpers provide explicit `f32` storage execution without changing older APIs.
- New `*_f16` helpers provide explicit `half::f16` storage execution for mixed-precision FFT paths.
- New generic `*_typed(...)` helpers and `forward_typed` / `inverse_typed` plan methods are the
  preferred Rust integration surface going forward.

Current truthful backend support:

- CPU: `high_accuracy`, `low_precision`, `mixed_precision`
- WGPU: `low_precision` only

Current mixed-precision meaning:

- `mixed_precision` now means `half::f16` storage with `f32` compute.

## WGPU Semantics

Apollo Stage 2 removes the CPU delegation path from the WGPU crate. `apollofft-wgpu` now reports truthful capability boundaries:

- 3D GPU FFT is implemented.
- 1D and 2D WGPU planning are not exposed in this stage.
- Python capability reporting only advertises `wgpu` when the host can actually acquire a usable adapter/device.
