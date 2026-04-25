//! WGPU device acquisition for this transform backend.

use std::sync::Arc;

use apollo_ntt::{DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT};

use crate::application::plan::NttWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::{NttGpuKernel, NttMode};

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    NttWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct NttWgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    kernel: Arc<NttGpuKernel>,
}

impl NttWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> WgpuResult<Self> {
        Ok(Self {
            kernel: Arc::new(NttGpuKernel::new(device.as_ref())),
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
            label: Some("apollo-ntt-wgpu"),
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
        WgpuCapabilities::full(true)
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
    pub const fn plan(&self, len: usize) -> NttWgpuPlan {
        NttWgpuPlan::new(len, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT)
    }

    /// Create a plan with an explicit modulus and primitive root.
    #[must_use]
    pub const fn plan_with_modulus(
        &self,
        len: usize,
        modulus: u64,
        primitive_root: u64,
    ) -> NttWgpuPlan {
        NttWgpuPlan::new(len, modulus, primitive_root)
    }

    /// Execute the direct forward NTT over the configured residue field.
    pub fn execute_forward(&self, plan: &NttWgpuPlan, input: &[u64]) -> WgpuResult<Vec<u64>> {
        let root = Self::validate_plan_and_input(plan, input)?;
        self.kernel.execute(
            self.device.as_ref(),
            self.queue.as_ref(),
            input,
            plan.len(),
            plan.modulus(),
            root,
            NttMode::Forward,
        )
    }

    /// Execute the direct inverse NTT over the configured residue field.
    pub fn execute_inverse(&self, plan: &NttWgpuPlan, input: &[u64]) -> WgpuResult<Vec<u64>> {
        let root = Self::validate_plan_and_input(plan, input)?;
        self.kernel.execute(
            self.device.as_ref(),
            self.queue.as_ref(),
            input,
            plan.len(),
            plan.modulus(),
            root,
            NttMode::Inverse,
        )
    }

    fn validate_plan_and_input(plan: &NttWgpuPlan, input: &[u64]) -> WgpuResult<u64> {
        let len = plan.len();
        let modulus = plan.modulus();
        let primitive_root = plan.primitive_root();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                len,
                modulus,
                primitive_root,
                message: "length must be greater than zero",
            });
        }
        if !len.is_power_of_two() {
            return Err(WgpuError::InvalidPlan {
                len,
                modulus,
                primitive_root,
                message: "length must be a power of two",
            });
        }
        if modulus < 2 {
            return Err(WgpuError::InvalidPlan {
                len,
                modulus,
                primitive_root,
                message: "modulus must be at least 2",
            });
        }
        if modulus > u32::MAX as u64 || primitive_root > u32::MAX as u64 {
            return Err(WgpuError::InvalidPlan {
                len,
                modulus,
                primitive_root,
                message: "current WGPU NTT surface supports 32-bit modulus and primitive root",
            });
        }
        if (modulus - 1) % len as u64 != 0 {
            return Err(WgpuError::InvalidPlan {
                len,
                modulus,
                primitive_root,
                message: "transform length is not supported by the modulus",
            });
        }
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        let root = mod_pow_u64(primitive_root, (modulus - 1) / len as u64, modulus);
        Ok(root)
    }
}

fn mod_pow_u64(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1_u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exp >>= 1;
    }
    result
}
