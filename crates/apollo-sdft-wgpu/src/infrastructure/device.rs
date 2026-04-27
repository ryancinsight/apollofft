//! WGPU device acquisition for the SDFT transform backend.

use std::sync::Arc;

use num_complex::{Complex32, Complex64};

use apollo_fft::PrecisionProfile;
use apollo_sdft::{SdftBinStorage, SdftRealStorage};

use crate::application::plan::SdftWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::SdftGpuKernel;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    SdftWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct SdftWgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    kernel: Arc<SdftGpuKernel>,
}

impl SdftWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> WgpuResult<Self> {
        Ok(Self {
            kernel: Arc::new(SdftGpuKernel::new(device.as_ref())),
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
            label: Some("apollo-sdft-wgpu"),
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
    pub const fn plan(&self, window_len: usize, bin_count: usize) -> SdftWgpuPlan {
        SdftWgpuPlan::new(window_len, bin_count)
    }

    /// Execute the direct SDFT bins computation for a real-valued window.
    ///
    /// Returns `Vec<Complex32>` with `plan.bin_count()` complex outputs.
    pub fn execute_forward(
        &self,
        plan: &SdftWgpuPlan,
        window: &[f32],
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_plan_window(plan, window)?;
        self.kernel.execute(
            self.device.as_ref(),
            self.queue.as_ref(),
            window,
            plan.window_len(),
            plan.bin_count(),
        )
    }

    /// Execute the direct SDFT bins computation with typed real input and typed complex output.
    ///
    /// `input_precision` must match `I::PROFILE`; `output_precision` must match `O::PROFILE`.
    /// WGPU arithmetic is `f32`; host storage is promoted/quantized at the dispatch boundary.
    pub fn execute_forward_typed_into<I: SdftRealStorage, O: SdftBinStorage>(
        &self,
        plan: &SdftWgpuPlan,
        input_precision: PrecisionProfile,
        output_precision: PrecisionProfile,
        window: &[I],
        output: &mut [O],
    ) -> WgpuResult<()> {
        let expected_in = I::PROFILE;
        if input_precision.storage != expected_in.storage
            || input_precision.compute != expected_in.compute
        {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        let expected_out = O::PROFILE;
        if output_precision.storage != expected_out.storage
            || output_precision.compute != expected_out.compute
        {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        if output.len() != plan.bin_count() {
            return Err(WgpuError::WindowLengthMismatch {
                expected: plan.bin_count(),
                actual: output.len(),
            });
        }
        let represented: Vec<f32> = window.iter().map(|v| v.to_f64() as f32).collect();
        let computed = self.execute_forward(plan, &represented)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = O::from_complex64(Complex64::new(f64::from(value.re), f64::from(value.im)));
        }
        Ok(())
    }

    /// Inverse execution is unsupported: SDFT is a direct DFT snapshot with no GPU inverse.
    pub fn execute_inverse(&self) -> WgpuResult<()> {
        Err(WgpuError::UnsupportedExecution {
            operation: "inverse",
        })
    }

    fn validate_plan_window(plan: &SdftWgpuPlan, window: &[f32]) -> WgpuResult<()> {
        if plan.window_len() == 0 || plan.bin_count() == 0 {
            return Err(WgpuError::InvalidPlan {
                window_len: plan.window_len(),
                bin_count: plan.bin_count(),
                message: "window_len and bin_count must each be greater than zero",
            });
        }
        if plan.bin_count() > plan.window_len() {
            return Err(WgpuError::InvalidPlan {
                window_len: plan.window_len(),
                bin_count: plan.bin_count(),
                message: "bin_count must not exceed window_len",
            });
        }
        if window.len() != plan.window_len() {
            return Err(WgpuError::WindowLengthMismatch {
                expected: plan.window_len(),
                actual: window.len(),
            });
        }
        Ok(())
    }
}
