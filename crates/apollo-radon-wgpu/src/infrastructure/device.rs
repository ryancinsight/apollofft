//! WGPU device acquisition for this transform backend.

use std::sync::Arc;

use ndarray::Array2;

use crate::application::plan::RadonWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::RadonGpuKernel;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    RadonWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct RadonWgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    kernel: Arc<RadonGpuKernel>,
}

impl RadonWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> WgpuResult<Self> {
        Ok(Self {
            kernel: Arc::new(RadonGpuKernel::new(device.as_ref())),
            device,
            queue,
        })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|error| WgpuError::AdapterUnavailable {
                message: error.to_string(),
            })?;
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("apollo-radon-wgpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&descriptor)).map_err(|error| {
                WgpuError::DeviceUnavailable {
                    message: error.to_string(),
                }
            })?;
        Self::new(Arc::new(device), Arc::new(queue))
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::forward_only(true)
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

    /// Create a metadata-only plan descriptor.
    #[must_use]
    pub fn plan(
        &self,
        rows: usize,
        cols: usize,
        angle_count: usize,
        detector_count: usize,
        detector_spacing: f64,
    ) -> RadonWgpuPlan {
        RadonWgpuPlan::new(
            rows,
            cols,
            angle_count,
            detector_count,
            detector_spacing.to_bits(),
        )
    }

    /// Execute the forward parallel-beam Radon projection.
    pub fn execute_forward(
        &self,
        plan: &RadonWgpuPlan,
        image: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        Self::validate_inputs(plan, image, angles)?;
        self.kernel.execute(
            self.device.as_ref(),
            self.queue.as_ref(),
            plan,
            image,
            angles,
        )
    }

    /// Inverse or adjoint execution is unsupported until the owning Radon crate defines the WGPU surface.
    pub fn execute_inverse(&self) -> WgpuResult<()> {
        Err(WgpuError::UnsupportedExecution {
            operation: "inverse",
        })
    }

    fn validate_inputs(
        plan: &RadonWgpuPlan,
        image: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<()> {
        if plan.rows() == 0
            || plan.cols() == 0
            || plan.angle_count() == 0
            || plan.detector_count() == 0
        {
            return Err(WgpuError::InvalidPlan {
                rows: plan.rows(),
                cols: plan.cols(),
                angle_count: plan.angle_count(),
                detector_count: plan.detector_count(),
                detector_spacing: plan.detector_spacing(),
                message: "geometry dimensions must be greater than zero",
            });
        }
        if !plan.detector_spacing().is_finite() || plan.detector_spacing() <= 0.0 {
            return Err(WgpuError::InvalidPlan {
                rows: plan.rows(),
                cols: plan.cols(),
                angle_count: plan.angle_count(),
                detector_count: plan.detector_count(),
                detector_spacing: plan.detector_spacing(),
                message: "detector spacing must be finite and positive",
            });
        }
        let (actual_rows, actual_cols) = image.dim();
        if (actual_rows, actual_cols) != (plan.rows(), plan.cols()) {
            return Err(WgpuError::ImageShapeMismatch {
                expected_rows: plan.rows(),
                expected_cols: plan.cols(),
                actual_rows,
                actual_cols,
            });
        }
        if angles.len() != plan.angle_count() {
            return Err(WgpuError::AngleCountMismatch {
                expected: plan.angle_count(),
                actual: angles.len(),
            });
        }
        Ok(())
    }
}
