//! WGPU device acquisition and backend orchestration for the Haar DWT.

use std::sync::Arc;

use crate::application::plan::WaveletWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::WaveletGpuKernel;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    WaveletWgpuBackend::try_default().is_ok()
}

/// WGPU backend for the Haar DWT.
///
/// Owns an acquired device/queue pair and a cached kernel pipeline.
#[derive(Debug, Clone)]
pub struct WaveletWgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    kernel: Arc<WaveletGpuKernel>,
}

impl WaveletWgpuBackend {
    /// Create a backend from an existing device and queue.
    #[must_use]
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let kernel = Arc::new(WaveletGpuKernel::new(&device));
        Self {
            device,
            queue,
            kernel,
        }
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|e| WgpuError::AdapterUnavailable {
                message: e.to_string(),
            })?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("apollo-wavelet-wgpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| WgpuError::DeviceUnavailable {
            message: e.to_string(),
        })?;
        Ok(Self::new(Arc::new(device), Arc::new(queue)))
    }

    /// Return truthful forward+inverse capability descriptor.
    #[must_use]
    pub fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::implemented(true)
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

    /// Create a plan descriptor for the given signal length and decomposition levels.
    #[must_use]
    pub const fn plan(&self, len: usize, levels: usize) -> WaveletWgpuPlan {
        WaveletWgpuPlan::new(len, levels)
    }

    /// Execute the forward multi-level Haar DWT on .
    ///
    /// Returns a flat coefficient buffer in Mallat ordering:
    /// .
    ///
    /// Validation:  must be a non-zero power of two,  must be
    /// non-zero, , and .
    pub fn execute_forward(&self, plan: &WaveletWgpuPlan, signal: &[f32]) -> WgpuResult<Vec<f32>> {
        Self::validate_plan(plan)?;
        if signal.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: signal.len(),
            });
        }
        self.kernel
            .execute_forward(&self.device, &self.queue, signal, plan.len(), plan.levels())
    }

    /// Execute the inverse multi-level Haar DWT on .
    ///
    /// Expects input in Mallat ordering (output of ).
    /// Returns the reconstructed signal of length .
    ///
    /// Validation mirrors .
    pub fn execute_inverse(
        &self,
        plan: &WaveletWgpuPlan,
        coefficients: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        Self::validate_plan(plan)?;
        if coefficients.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: coefficients.len(),
            });
        }
        self.kernel.execute_inverse(
            &self.device,
            &self.queue,
            coefficients,
            plan.len(),
            plan.levels(),
        )
    }

    /// Validate plan parameters before GPU dispatch.
    ///
    /// Invariants:
    /// -  and  is a power of two (Haar requires dyadic length).
    /// -  (at least one decomposition pass).
    /// -  (each level halves the approximation subband).
    fn validate_plan(plan: &WaveletWgpuPlan) -> WgpuResult<()> {
        let len = plan.len();
        let levels = plan.levels();
        if len == 0 || !len.is_power_of_two() {
            return Err(WgpuError::InvalidLength {
                len,
                levels,
                message: "len must be a non-zero power of two",
            });
        }
        if levels == 0 {
            return Err(WgpuError::InvalidLength {
                len,
                levels,
                message: "levels must be non-zero",
            });
        }
        if (1usize << levels) > len {
            return Err(WgpuError::InvalidLength {
                len,
                levels,
                message: "2^levels must not exceed len",
            });
        }
        Ok(())
    }
}
