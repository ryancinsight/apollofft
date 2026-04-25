//! WGPU device acquisition for this transform backend.

use std::sync::Arc;

use num_complex::Complex32;

use crate::application::plan::MellinWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::MellinGpuKernel;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    MellinWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct MellinWgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    kernel: Arc<MellinGpuKernel>,
}

impl MellinWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> WgpuResult<Self> {
        Ok(Self {
            kernel: Arc::new(MellinGpuKernel::new(device.as_ref())),
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
            label: Some("apollo-mellin-wgpu"),
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
    pub fn plan(&self, samples: usize, min_scale: f64, max_scale: f64) -> MellinWgpuPlan {
        MellinWgpuPlan::new(samples, min_scale.to_bits(), max_scale.to_bits())
    }

    /// Execute the forward Mellin log-frequency spectrum for a real-valued `f32` signal.
    pub fn execute_forward(
        &self,
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f64,
        signal_max: f64,
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_inputs(plan, signal, signal_min, signal_max)?;
        self.kernel.execute(
            self.device.as_ref(),
            self.queue.as_ref(),
            plan,
            signal,
            signal_min,
            signal_max,
        )
    }

    /// Inverse or adjoint execution is unsupported until the owning Mellin crate defines it.
    pub fn execute_inverse(&self) -> WgpuResult<()> {
        Err(WgpuError::UnsupportedExecution {
            operation: "inverse",
        })
    }

    fn validate_inputs(
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f64,
        signal_max: f64,
    ) -> WgpuResult<()> {
        if plan.samples() == 0 {
            return Err(WgpuError::InvalidPlan {
                samples: plan.samples(),
                min_scale: plan.min_scale(),
                max_scale: plan.max_scale(),
                message: "sample count must be greater than zero",
            });
        }
        if !plan.min_scale().is_finite()
            || !plan.max_scale().is_finite()
            || plan.min_scale() <= 0.0
            || plan.max_scale() <= 0.0
        {
            return Err(WgpuError::InvalidPlan {
                samples: plan.samples(),
                min_scale: plan.min_scale(),
                max_scale: plan.max_scale(),
                message: "plan scales must be finite and positive",
            });
        }
        if plan.min_scale() >= plan.max_scale() {
            return Err(WgpuError::InvalidPlan {
                samples: plan.samples(),
                min_scale: plan.min_scale(),
                max_scale: plan.max_scale(),
                message: "min_scale must be less than max_scale",
            });
        }
        if signal.is_empty() {
            return Err(WgpuError::LengthMismatch {
                expected: 1,
                actual: 0,
            });
        }
        if !signal_min.is_finite()
            || !signal_max.is_finite()
            || signal_min <= 0.0
            || signal_max <= 0.0
        {
            return Err(WgpuError::InvalidSignalDomain {
                signal_min,
                signal_max,
                message: "signal bounds must be finite and positive",
            });
        }
        if signal_min >= signal_max {
            return Err(WgpuError::InvalidSignalDomain {
                signal_min,
                signal_max,
                message: "signal_min must be less than signal_max",
            });
        }
        Ok(())
    }
}
