//! WGPU device acquisition and backend orchestration for the STFT.

use std::sync::Arc;

use num_complex::Complex32;

use crate::application::plan::StftWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::StftGpuKernel;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    StftWgpuBackend::try_default().is_ok()
}

/// WGPU backend for the STFT.
///
/// Owns an acquired device/queue pair and a cached kernel pipeline.
#[derive(Debug, Clone)]
pub struct StftWgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    kernel: Arc<StftGpuKernel>,
}

impl StftWgpuBackend {
    /// Create a backend from an existing device and queue.
    #[must_use]
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let kernel = Arc::new(StftGpuKernel::new(&device));
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
            label: Some("apollo-stft-wgpu"),
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

    /// Return truthful forward-only capability descriptor.
    #[must_use]
    pub fn capabilities(&self) -> WgpuCapabilities {
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

    /// Execute the forward STFT on `signal` using the supplied plan.
    ///
    /// Returns `frame_count * frame_len` complex values in row-major order:
    /// element `m * frame_len + k` is X[frame_m=m, bin_k=k].
    ///
    /// Frame count: `1 + signal.len().div_ceil(hop_len)` when signal is long enough.
    pub fn execute_forward(
        &self,
        plan: &StftWgpuPlan,
        signal: &[f32],
    ) -> WgpuResult<Vec<Complex32>> {
        if plan.frame_len() == 0 {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "frame_len must be non-zero",
            });
        }
        if plan.hop_len() == 0 {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "hop_len must be non-zero",
            });
        }
        if plan.hop_len() > plan.frame_len() {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "hop_len must not exceed frame_len",
            });
        }
        if signal.len() < plan.frame_len() {
            return Err(WgpuError::InputTooShort {
                min: plan.frame_len(),
                actual: signal.len(),
            });
        }
        // Mirror the CPU StftPlan::frame_count formula exactly:
        //   1 + signal_len.div_ceil(hop_len)
        let frame_count = 1 + signal.len().div_ceil(plan.hop_len());
        self.kernel.execute(
            &self.device,
            &self.queue,
            signal,
            plan.frame_len(),
            plan.hop_len(),
            frame_count,
        )
    }

    /// Inverse STFT is not implemented on the GPU.
    pub fn execute_inverse(
        &self,
        _plan: &StftWgpuPlan,
        _spectrum: &[Complex32],
    ) -> WgpuResult<Vec<f32>> {
        Err(WgpuError::UnsupportedExecution {
            operation: "inverse",
        })
    }
}
