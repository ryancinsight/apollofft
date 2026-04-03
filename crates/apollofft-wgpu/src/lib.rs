#![warn(missing_docs)]
//! WGPU backend surface for Apollo FFT.
//!
//! The current adapter validates device availability and exposes the same
//! numerical contract as the CPU backend while the long-form GPU kernels and
//! shader pipelines stabilize behind this crate boundary.

pub mod application;
pub mod domain;
pub mod infrastructure;

use apollofft::backend::FftBackend;
use apollofft::domain::backend::BackendCapabilities;
use apollofft::error::{ApolloError, ApolloResult};
use apollofft::types::{BackendKind, Normalization, Shape1D, Shape2D, Shape3D};
pub use infrastructure::gpu_fft::{gpu_fft_available, GpuFft3d};

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct WgpuBackend {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
}

impl WgpuBackend {
    /// Create a backend from an existing device and queue.
    #[must_use]
    pub fn new(device: std::sync::Arc<wgpu::Device>, queue: std::sync::Arc<wgpu::Queue>) -> Self {
        Self { device, queue }
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> ApolloResult<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .map_err(|error| ApolloError::BackendUnavailable {
                backend: format!("wgpu adapter unavailable: {error}"),
            })?;
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("apollofft-wgpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::default(),
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&descriptor))
            .map_err(|error| ApolloError::Wgpu {
                message: error.to_string(),
            })?;
        Ok(Self::new(
            std::sync::Arc::new(device),
            std::sync::Arc::new(queue),
        ))
    }
}

impl FftBackend for WgpuBackend {
    type Plan1D = ();
    type Plan2D = ();
    type Plan3D = GpuFft3d;

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Wgpu
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            kind: BackendKind::Wgpu,
            normalization: Normalization::FftwCompatible,
            supports_1d: false,
            supports_2d: false,
            supports_3d: true,
            supports_real_to_complex: false,
        }
    }

    fn plan_1d(&self, _shape: Shape1D) -> ApolloResult<Self::Plan1D> {
        Err(ApolloError::BackendUnavailable {
            backend: "wgpu 1D plans are not exposed in v1".to_string(),
        })
    }

    fn plan_2d(&self, _shape: Shape2D) -> ApolloResult<Self::Plan2D> {
        Err(ApolloError::BackendUnavailable {
            backend: "wgpu 2D plans are not exposed in v1".to_string(),
        })
    }

    fn plan_3d(&self, shape: Shape3D) -> ApolloResult<Self::Plan3D> {
        GpuFft3d::new(
            std::sync::Arc::clone(&self.device),
            std::sync::Arc::clone(&self.queue),
            shape.nx,
            shape.ny,
            shape.nz,
        )
        .map_err(|message| ApolloError::Wgpu { message })
    }
}

