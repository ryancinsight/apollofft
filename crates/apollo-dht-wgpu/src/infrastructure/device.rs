//! WGPU device acquisition for this transform backend.

use std::sync::Arc;

use apollo_dht::application::execution::plan::dht::HartleyStorage;
use apollo_fft::PrecisionProfile;

use crate::application::plan::DhtWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::DhtGpuKernel;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    DhtWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct DhtWgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    kernel: Arc<DhtGpuKernel>,
}

impl DhtWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> WgpuResult<Self> {
        let kernel = Arc::new(DhtGpuKernel::new(device.as_ref()));
        Ok(Self {
            device,
            queue,
            kernel,
        })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
                .map_err(|error| WgpuError::AdapterUnavailable {
                message: error.to_string(),
            })?;
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("apollo-dht-wgpu"),
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

    /// Create a metadata-only plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize) -> DhtWgpuPlan {
        DhtWgpuPlan::new(len)
    }

    /// Execute the unnormalized forward 1D DHT for a real-valued `f32` signal.
    pub fn execute_forward(&self, plan: &DhtWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        Self::validate_plan_input(plan, input)?;
        self.kernel
            .execute(self.device.as_ref(), self.queue.as_ref(), input, false)
    }

    /// Execute the unnormalized forward 1D DHT with caller-owned typed storage.
    ///
    /// WGPU arithmetic remains `f32`; mixed `f16` storage is promoted once to
    /// represented `f32` before dispatch and quantized at the output boundary.
    pub fn execute_forward_typed_into<T: HartleyStorage>(
        &self,
        plan: &DhtWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed_plan_input::<T>(plan, precision, input, output)?;
        let represented = typed_to_f32(input);
        let computed = self.execute_forward(plan, &represented)?;
        write_typed_output(&computed, output);
        Ok(())
    }

    /// Execute the normalized inverse 1D DHT for a real-valued `f32` spectrum.
    pub fn execute_inverse(&self, plan: &DhtWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        Self::validate_plan_input(plan, input)?;
        self.kernel
            .execute(self.device.as_ref(), self.queue.as_ref(), input, true)
    }

    /// Execute the normalized inverse 1D DHT with caller-owned typed storage.
    pub fn execute_inverse_typed_into<T: HartleyStorage>(
        &self,
        plan: &DhtWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed_plan_input::<T>(plan, precision, input, output)?;
        let represented = typed_to_f32(input);
        let computed = self.execute_inverse(plan, &represented)?;
        write_typed_output(&computed, output);
        Ok(())
    }

    fn validate_plan_input(plan: &DhtWgpuPlan, input: &[f32]) -> WgpuResult<()> {
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidLength {
                len,
                message: "length must be greater than zero",
            });
        }
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        Ok(())
    }

    fn validate_typed_plan_input<T: HartleyStorage>(
        plan: &DhtWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &[T],
    ) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidLength {
                len,
                message: "length must be greater than zero",
            });
        }
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        Ok(())
    }
}

fn typed_to_f32<T: HartleyStorage>(input: &[T]) -> Vec<f32> {
    input.iter().map(|value| value.to_f64() as f32).collect()
}

fn write_typed_output<T: HartleyStorage>(source: &[f32], output: &mut [T]) {
    for (slot, value) in output.iter_mut().zip(source.iter().copied()) {
        *slot = T::from_f64(f64::from(value));
    }
}
