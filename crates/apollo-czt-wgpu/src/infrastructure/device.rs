//! WGPU device acquisition for this transform backend.

use std::sync::Arc;

use num_complex::Complex32;

use crate::application::plan::CztWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::CztGpuKernel;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    CztWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct CztWgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    kernel: Arc<CztGpuKernel>,
}

impl CztWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> WgpuResult<Self> {
        Ok(Self {
            kernel: Arc::new(CztGpuKernel::new(device.as_ref())),
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
            label: Some("apollo-czt-wgpu"),
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
        input_len: usize,
        output_len: usize,
        a: Complex32,
        w: Complex32,
    ) -> CztWgpuPlan {
        CztWgpuPlan::new(
            input_len,
            output_len,
            [a.re.to_bits(), a.im.to_bits()],
            [w.re.to_bits(), w.im.to_bits()],
        )
    }

    /// Execute the direct forward CZT for a complex-valued `f32` signal.
    pub fn execute_forward(
        &self,
        plan: &CztWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_plan_input(plan, input)?;
        self.kernel
            .execute(self.device.as_ref(), self.queue.as_ref(), plan, input)
    }

    /// Inverse or adjoint execution is unsupported until the owning CZT crate defines it.
    pub fn execute_inverse(&self) -> WgpuResult<()> {
        Err(WgpuError::UnsupportedExecution {
            operation: "inverse",
        })
    }

    fn validate_plan_input(plan: &CztWgpuPlan, input: &[Complex32]) -> WgpuResult<()> {
        let input_len = plan.input_len();
        let output_len = plan.output_len();
        if input_len == 0 || output_len == 0 {
            return Err(WgpuError::InvalidLength {
                input_len,
                output_len,
                message: "lengths must be greater than zero",
            });
        }
        if input.len() != input_len {
            return Err(WgpuError::LengthMismatch {
                expected: input_len,
                actual: input.len(),
            });
        }
        let a = plan.a();
        let w = plan.w();
        let a_norm = a.norm();
        let w_norm = w.norm();
        if !a_norm.is_finite() || !w_norm.is_finite() || a_norm == 0.0 || w_norm == 0.0 {
            return Err(WgpuError::InvalidParameters {
                message: "spiral parameters must have finite non-zero magnitude",
            });
        }
        Ok(())
    }
}
