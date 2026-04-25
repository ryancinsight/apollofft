//! WGPU device acquisition for NUFFT backends.

use std::sync::Arc;

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
            required_limits: wgpu::Limits::downlevel_defaults(),
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
        NufftWgpuCapabilities::direct_all(true)
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
