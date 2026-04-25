//! WGPU device acquisition for NUFFT backends.

use std::sync::Arc;

use apollo_nufft::infrastructure::kernel::kaiser_bessel::{fft_signed_index, i0, kb_kernel_ft};
use apollo_nufft::{UniformDomain1D, UniformGrid3D};
use ndarray::{Array1, Array3};
use num_complex::{Complex32, Complex64};

use crate::application::plan::{NufftWgpuPlan1D, NufftWgpuPlan3D};
use crate::domain::capabilities::NufftWgpuCapabilities;
use crate::domain::error::{NufftWgpuError, NufftWgpuResult};
use crate::infrastructure::kernel::NufftGpuKernel;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn nufft_wgpu_available() -> bool {
    NufftWgpuBackend::try_default().is_ok()
}

/// WGPU NUFFT backend descriptor.
#[derive(Debug, Clone)]
pub struct NufftWgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    kernel: Arc<NufftGpuKernel>,
}

impl NufftWgpuBackend {
    /// Create a NUFFT WGPU backend from an existing device and queue.
    #[must_use]
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            kernel: Arc::new(NufftGpuKernel::new(device.as_ref())),
            device,
            queue,
        }
    }

    /// Create a backend by requesting a default WGPU adapter and device.
    pub fn try_default() -> NufftWgpuResult<Self> {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|error| NufftWgpuError::AdapterUnavailable {
                message: error.to_string(),
            })?;
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("apollo-nufft-wgpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 8,
                ..wgpu::Limits::downlevel_defaults()
            },
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&descriptor)).map_err(|error| {
                NufftWgpuError::DeviceUnavailable {
                    message: error.to_string(),
                }
            })?;
        Ok(Self::new(Arc::new(device), Arc::new(queue)))
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> NufftWgpuCapabilities {
        NufftWgpuCapabilities::direct_all_fast_all(true)
    }

    /// Return the acquired WGPU device.
    #[must_use]
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Return the acquired WGPU queue.
    #[must_use]
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Create a 1D plan descriptor.
    #[must_use]
    pub const fn plan_1d(
        &self,
        domain: UniformDomain1D,
        oversampling: usize,
        kernel_width: usize,
    ) -> NufftWgpuPlan1D {
        NufftWgpuPlan1D::new(domain, oversampling, kernel_width)
    }

    /// Create a 3D plan descriptor.
    #[must_use]
    pub const fn plan_3d(
        &self,
        grid: UniformGrid3D,
        oversampling: usize,
        kernel_width: usize,
    ) -> NufftWgpuPlan3D {
        NufftWgpuPlan3D::new(grid, oversampling, kernel_width)
    }

    /// Execute exact direct Type-1 1D NUFFT on WGPU.
    pub fn execute_type1_1d(
        &self,
        plan: &NufftWgpuPlan1D,
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Array1<Complex64>> {
        validate_pair_lengths(positions.len(), values.len())?;
        validate_usize_to_u32(plan.domain().n)?;
        validate_usize_to_u32(positions.len())?;
        let output = self.kernel.execute_type1_1d(
            self.device.as_ref(),
            self.queue.as_ref(),
            plan.domain().n,
            plan.domain().length() as f32,
            positions,
            values,
        )?;
        Ok(Array1::from_vec(
            output
                .into_iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect(),
        ))
    }

    /// Execute exact direct Type-2 1D NUFFT on WGPU.
    pub fn execute_type2_1d(
        &self,
        plan: &NufftWgpuPlan1D,
        fourier_coeffs: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<Vec<Complex64>> {
        if fourier_coeffs.len() != plan.domain().n {
            return Err(NufftWgpuError::InputLengthMismatch {
                expected: plan.domain().n,
                actual: fourier_coeffs.len(),
            });
        }
        validate_usize_to_u32(plan.domain().n)?;
        validate_usize_to_u32(positions.len())?;
        let output = self.kernel.execute_type2_1d(
            self.device.as_ref(),
            self.queue.as_ref(),
            plan.domain().n,
            plan.domain().length() as f32,
            fourier_coeffs,
            positions,
        )?;
        Ok(output
            .into_iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect())
    }

    /// Execute fast gridded Type-1 1D NUFFT on WGPU.
    pub fn execute_fast_type1_1d(
        &self,
        plan: &NufftWgpuPlan1D,
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Array1<Complex64>> {
        validate_pair_lengths(positions.len(), values.len())?;
        validate_fast_1d_plan(plan)?;
        validate_usize_to_u32(positions.len())?;
        let fast = fast_1d_metadata(plan)?;
        let output = self.kernel.execute_fast_type1_1d(
            &self.device,
            &self.queue,
            plan.domain().n,
            fast.oversampled_len,
            plan.kernel_width(),
            plan.domain().length() as f32,
            fast.beta as f32,
            fast.i0_beta as f32,
            &fast.deconv,
            positions,
            values,
        )?;
        Ok(Array1::from_vec(
            output
                .into_iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect(),
        ))
    }

    /// Execute fast gridded Type-2 1D NUFFT on WGPU.
    pub fn execute_fast_type2_1d(
        &self,
        plan: &NufftWgpuPlan1D,
        fourier_coeffs: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<Vec<Complex64>> {
        if fourier_coeffs.len() != plan.domain().n {
            return Err(NufftWgpuError::InputLengthMismatch {
                expected: plan.domain().n,
                actual: fourier_coeffs.len(),
            });
        }
        validate_fast_1d_plan(plan)?;
        validate_usize_to_u32(positions.len())?;
        let fast = fast_1d_metadata(plan)?;
        let output = self.kernel.execute_fast_type2_1d(
            &self.device,
            &self.queue,
            plan.domain().n,
            fast.oversampled_len,
            plan.kernel_width(),
            plan.domain().length() as f32,
            fast.beta as f32,
            fast.i0_beta as f32,
            &fast.deconv,
            fourier_coeffs,
            positions,
        )?;
        Ok(output
            .into_iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect())
    }

    /// Execute fast gridded Type-1 3D NUFFT on WGPU.
    pub fn execute_fast_type1_3d(
        &self,
        plan: &NufftWgpuPlan3D,
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> NufftWgpuResult<Array3<Complex64>> {
        validate_pair_lengths(positions.len(), values.len())?;
        let grid = plan.grid();
        validate_usize_to_u32(grid.nx)?;
        validate_usize_to_u32(grid.ny)?;
        validate_usize_to_u32(grid.nz)?;
        validate_usize_to_u32(positions.len())?;
        let fast = fast_3d_metadata(plan)?;
        let (lx, ly, lz) = grid.lengths();
        let output = self.kernel.execute_fast_type1_3d(
            &self.device,
            &self.queue,
            (grid.nx, grid.ny, grid.nz),
            (fast.mx, fast.my, fast.mz),
            plan.kernel_width(),
            (lx as f32, ly as f32, lz as f32),
            fast.beta as f32,
            fast.i0_beta as f32,
            &fast.deconv_xyz,
            positions,
            values,
        )?;
        let converted: Vec<Complex64> = output
            .into_iter()
            .map(|v| Complex64::new(v.re as f64, v.im as f64))
            .collect();
        Array3::from_shape_vec((grid.nx, grid.ny, grid.nz), converted).map_err(|_| {
            NufftWgpuError::InvalidPlan {
                message: "fast 3D type1 output shape does not match grid dimensions",
            }
        })
    }

    /// Execute fast gridded Type-2 3D NUFFT on WGPU.
    pub fn execute_fast_type2_3d(
        &self,
        plan: &NufftWgpuPlan3D,
        modes: &Array3<Complex32>,
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<Vec<Complex64>> {
        let grid = plan.grid();
        if modes.dim() != (grid.nx, grid.ny, grid.nz) {
            return Err(NufftWgpuError::InvalidPlan {
                message: "mode shape must match 3D plan grid dimensions",
            });
        }
        validate_usize_to_u32(grid.nx)?;
        validate_usize_to_u32(grid.ny)?;
        validate_usize_to_u32(grid.nz)?;
        validate_usize_to_u32(positions.len())?;
        let fast = fast_3d_metadata(plan)?;
        let (lx, ly, lz) = grid.lengths();
        let flat_modes: Vec<Complex32> = modes.iter().copied().collect();
        let output = self.kernel.execute_fast_type2_3d(
            &self.device,
            &self.queue,
            (grid.nx, grid.ny, grid.nz),
            (fast.mx, fast.my, fast.mz),
            plan.kernel_width(),
            (lx as f32, ly as f32, lz as f32),
            fast.beta as f32,
            fast.i0_beta as f32,
            &fast.deconv_xyz,
            &flat_modes,
            positions,
        )?;
        Ok(output
            .into_iter()
            .map(|v| Complex64::new(v.re as f64, v.im as f64))
            .collect())
    }

    /// Execute exact direct Type-1 3D NUFFT on WGPU.
    pub fn execute_type1_3d(
        &self,
        plan: &NufftWgpuPlan3D,
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> NufftWgpuResult<Array3<Complex64>> {
        validate_pair_lengths(positions.len(), values.len())?;
        let grid = plan.grid();
        validate_usize_to_u32(grid.nx)?;
        validate_usize_to_u32(grid.ny)?;
        validate_usize_to_u32(grid.nz)?;
        validate_usize_to_u32(positions.len())?;
        let (lx, ly, lz) = grid.lengths();
        let output = self.kernel.execute_type1_3d(
            self.device.as_ref(),
            self.queue.as_ref(),
            (grid.nx, grid.ny, grid.nz),
            (lx as f32, ly as f32, lz as f32),
            positions,
            values,
        )?;
        let converted: Vec<Complex64> = output
            .into_iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        Array3::from_shape_vec((grid.nx, grid.ny, grid.nz), converted).map_err(|_| {
            NufftWgpuError::InvalidPlan {
                message: "3D output shape does not match grid dimensions",
            }
        })
    }

    /// Execute exact direct Type-2 3D NUFFT on WGPU.
    pub fn execute_type2_3d(
        &self,
        plan: &NufftWgpuPlan3D,
        modes: &Array3<Complex32>,
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<Vec<Complex64>> {
        let grid = plan.grid();
        if modes.dim() != (grid.nx, grid.ny, grid.nz) {
            return Err(NufftWgpuError::InvalidPlan {
                message: "mode shape must match 3D plan grid dimensions",
            });
        }
        validate_usize_to_u32(grid.nx)?;
        validate_usize_to_u32(grid.ny)?;
        validate_usize_to_u32(grid.nz)?;
        validate_usize_to_u32(positions.len())?;
        let (lx, ly, lz) = grid.lengths();
        let coefficients: Vec<Complex32> = modes.iter().copied().collect();
        let output = self.kernel.execute_type2_3d(
            self.device.as_ref(),
            self.queue.as_ref(),
            (grid.nx, grid.ny, grid.nz),
            (lx as f32, ly as f32, lz as f32),
            &coefficients,
            positions,
        )?;
        Ok(output
            .into_iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect())
    }
}

struct Fast3DMetadata {
    mx: usize,
    my: usize,
    mz: usize,
    beta: f64,
    i0_beta: f64,
    deconv_xyz: Vec<f32>,
}

fn fast_3d_metadata(plan: &NufftWgpuPlan3D) -> NufftWgpuResult<Fast3DMetadata> {
    let grid = plan.grid();
    let sigma = plan.oversampling();
    let w = plan.kernel_width();
    if sigma < 2 {
        return Err(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT oversampling factor must be >= 2",
        });
    }
    if w < 2 {
        return Err(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT kernel width must be >= 2",
        });
    }
    let mx_raw = grid
        .nx
        .checked_mul(sigma)
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT mx overflow",
        })?
        .max(2 * w + 1);
    let my_raw = grid
        .ny
        .checked_mul(sigma)
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT my overflow",
        })?
        .max(2 * w + 1);
    let mz_raw = grid
        .nz
        .checked_mul(sigma)
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT mz overflow",
        })?
        .max(2 * w + 1);
    let mx = mx_raw
        .checked_next_power_of_two()
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT mx radix-2 length overflow",
        })?;
    let my = my_raw
        .checked_next_power_of_two()
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT my radix-2 length overflow",
        })?;
    let mz = mz_raw
        .checked_next_power_of_two()
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT mz radix-2 length overflow",
        })?;
    validate_usize_to_u32(mx)?;
    validate_usize_to_u32(my)?;
    validate_usize_to_u32(mz)?;
    validate_usize_to_u32(
        mx.checked_mul(my)
            .and_then(|v| v.checked_mul(mz))
            .unwrap_or(usize::MAX),
    )?;

    let beta = std::f64::consts::PI * (1.0 - 1.0 / (2.0 * sigma as f64)) * (2 * w) as f64;
    let i0_beta = i0(beta);

    let deconv_x: Vec<f32> = (0..grid.nx)
        .map(|k| {
            let xi = fft_signed_index(k, grid.nx) as f64 / mx as f64;
            (1.0 / kb_kernel_ft(xi, w, beta, i0_beta)) as f32
        })
        .collect();
    let deconv_y: Vec<f32> = (0..grid.ny)
        .map(|k| {
            let xi = fft_signed_index(k, grid.ny) as f64 / my as f64;
            (1.0 / kb_kernel_ft(xi, w, beta, i0_beta)) as f32
        })
        .collect();
    let deconv_z: Vec<f32> = (0..grid.nz)
        .map(|k| {
            let xi = fft_signed_index(k, grid.nz) as f64 / mz as f64;
            (1.0 / kb_kernel_ft(xi, w, beta, i0_beta)) as f32
        })
        .collect();

    let mut deconv_xyz = Vec::with_capacity(grid.nx + grid.ny + grid.nz);
    deconv_xyz.extend_from_slice(&deconv_x);
    deconv_xyz.extend_from_slice(&deconv_y);
    deconv_xyz.extend_from_slice(&deconv_z);

    Ok(Fast3DMetadata {
        mx,
        my,
        mz,
        beta,
        i0_beta,
        deconv_xyz,
    })
}

fn validate_pair_lengths(expected: usize, actual: usize) -> NufftWgpuResult<()> {
    if expected != actual {
        return Err(NufftWgpuError::InputLengthMismatch { expected, actual });
    }
    Ok(())
}

fn validate_usize_to_u32(value: usize) -> NufftWgpuResult<()> {
    if value > u32::MAX as usize {
        return Err(NufftWgpuError::InvalidPlan {
            message: "WGPU dispatch dimension must fit in u32",
        });
    }
    Ok(())
}

struct Fast1DMetadata {
    oversampled_len: usize,
    beta: f64,
    i0_beta: f64,
    deconv: Vec<f32>,
}

fn validate_fast_1d_plan(plan: &NufftWgpuPlan1D) -> NufftWgpuResult<()> {
    if plan.oversampling() < 2 {
        return Err(NufftWgpuError::InvalidPlan {
            message: "fast 1D NUFFT oversampling factor must be >= 2",
        });
    }
    if plan.kernel_width() < 2 {
        return Err(NufftWgpuError::InvalidPlan {
            message: "fast 1D NUFFT kernel width must be >= 2",
        });
    }
    validate_usize_to_u32(plan.domain().n)?;
    let Some(oversampled_len) = plan.domain().n.checked_mul(plan.oversampling()) else {
        return Err(NufftWgpuError::InvalidPlan {
            message: "fast 1D NUFFT oversampled length overflow",
        });
    };
    validate_usize_to_u32(oversampled_len)
}

fn fast_1d_metadata(plan: &NufftWgpuPlan1D) -> NufftWgpuResult<Fast1DMetadata> {
    validate_fast_1d_plan(plan)?;
    let oversampled_len =
        plan.domain()
            .n
            .checked_mul(plan.oversampling())
            .ok_or(NufftWgpuError::InvalidPlan {
                message: "fast 1D NUFFT oversampled length overflow",
            })?;
    let beta = std::f64::consts::PI
        * (1.0 - 1.0 / (2.0 * plan.oversampling() as f64))
        * (2 * plan.kernel_width()) as f64;
    let i0_beta = i0(beta);
    let deconv = (0..plan.domain().n)
        .map(|k| {
            let xi = fft_signed_index(k, plan.domain().n) as f64 / oversampled_len as f64;
            (1.0 / kb_kernel_ft(xi, plan.kernel_width(), beta, i0_beta)) as f32
        })
        .collect();
    Ok(Fast1DMetadata {
        oversampled_len,
        beta,
        i0_beta,
        deconv,
    })
}
