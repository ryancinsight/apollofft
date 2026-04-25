//! WGPU device acquisition and FrFT execution backend.

use std::sync::Arc;

use num_complex::Complex32;

use crate::application::plan::FrftWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::FrftGpuKernel;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    FrftWgpuBackend::try_default().is_ok()
}

/// WGPU backend for FrFT execution.
#[derive(Debug, Clone)]
pub struct FrftWgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    kernel: Arc<FrftGpuKernel>,
}

impl FrftWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> WgpuResult<Self> {
        Ok(Self {
            kernel: Arc::new(FrftGpuKernel::new(device.as_ref())),
            device,
            queue,
        })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|e| WgpuError::AdapterUnavailable {
                message: e.to_string(),
            })?;
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("apollo-frft-wgpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&descriptor)).map_err(|e| {
                WgpuError::DeviceUnavailable {
                    message: e.to_string(),
                }
            })?;
        Self::new(Arc::new(device), Arc::new(queue))
    }

    /// Return truthful current capabilities (forward and inverse both implemented).
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

    /// Create a metadata plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize, order: f32) -> FrftWgpuPlan {
        FrftWgpuPlan::new(len, order)
    }

    /// Execute the forward FrFT for a complex-valued f32 signal.
    ///
    /// Validates input length, determines the dispatch mode from the plan order,
    /// precomputes cot/csc/scale for non-integer orders, then calls the kernel.
    pub fn execute_forward(
        &self,
        plan: &FrftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate(plan, input)?;
        let (mode, cot, csc, scale_re, scale_im) = mode_params(plan);
        self.kernel.execute(
            self.device.as_ref(),
            self.queue.as_ref(),
            input,
            plan.len(),
            mode,
            cot,
            csc,
            scale_re,
            scale_im,
        )
    }

    /// Execute the inverse FrFT, equivalent to the forward FrFT of order -a.
    pub fn execute_inverse(
        &self,
        plan: &FrftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        let inv_plan = FrftWgpuPlan::new(plan.len(), -plan.order());
        Self::validate(&inv_plan, input)?;
        let (mode, cot, csc, scale_re, scale_im) = mode_params(&inv_plan);
        self.kernel.execute(
            self.device.as_ref(),
            self.queue.as_ref(),
            input,
            inv_plan.len(),
            mode,
            cot,
            csc,
            scale_re,
            scale_im,
        )
    }

    fn validate(plan: &FrftWgpuPlan, input: &[Complex32]) -> WgpuResult<()> {
        if plan.len() == 0 {
            return Err(WgpuError::InvalidPlan {
                len: 0,
                message: "length must be greater than zero",
            });
        }
        if !plan.order().is_finite() {
            return Err(WgpuError::NonFiniteOrder);
        }
        if input.len() != plan.len() {
            return Err(WgpuError::InputLengthMismatch {
                expected: plan.len(),
                actual: input.len(),
            });
        }
        Ok(())
    }
}

/// Determine the dispatch mode and trigonometric parameters from a plan.
///
/// Returns (mode, cot, csc, scale_re, scale_im) where:
/// - Integer rotations (frac < 1e-5): mode in {0,1,2,3}, chirp params zeroed.
/// - Non-integer order: mode=4, cot/csc/scale computed from alpha=reduced*pi/2.
fn mode_params(plan: &FrftWgpuPlan) -> (u32, f32, f32, f32, f32) {
    let order = plan.order();
    let reduced = ((order % 4.0_f32) + 4.0_f32) % 4.0_f32;
    let rounded = reduced.round();
    let frac = (reduced - rounded).abs();
    if frac < 1.0e-5_f32 {
        let mode = if reduced < 0.5_f32 || reduced > 3.5_f32 {
            0_u32
        } else if reduced < 1.5_f32 {
            1_u32
        } else if reduced < 2.5_f32 {
            2_u32
        } else {
            3_u32
        };
        (mode, 0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32)
    } else {
        let alpha = reduced * std::f32::consts::FRAC_PI_2;
        let sin_a = alpha.sin();
        let cos_a = alpha.cos();
        let cot = cos_a / sin_a;
        let csc = 1.0_f32 / sin_a;
        let n_f = plan.len() as f32;
        // scale = sqrt(1 - i*cot) / sqrt(n)
        // = sqrt((1-i*cot)/n) via polar form
        let z_norm = (1.0_f32 + cot * cot).sqrt();
        let z_arg = (-cot).atan2(1.0_f32);
        let sr = z_norm.sqrt() / n_f.sqrt();
        let sa = z_arg * 0.5_f32;
        let scale_re = sr * sa.cos();
        let scale_im = sr * sa.sin();
        (4_u32, cot, csc, scale_re, scale_im)
    }
}
