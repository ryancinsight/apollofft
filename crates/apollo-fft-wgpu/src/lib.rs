#![warn(missing_docs)]
//! WGPU dense FFT backend surface for Apollo.
//!
//! The current adapter validates device availability and exposes the same
//! numerical contract as the CPU dense FFT backend. NUFFT-specific GPU
//! execution is intentionally owned by `apollo-nufft-wgpu`.

pub mod application;
pub mod domain;
pub mod infrastructure;

use apollo_fft::backend::FftBackend;
use apollo_fft::domain::backend::BackendCapabilities;
use apollo_fft::error::{ApolloError, ApolloResult};
use apollo_fft::types::{BackendKind, Normalization, PrecisionProfile, Shape1D, Shape2D, Shape3D};
pub use infrastructure::gpu_fft::{gpu_fft_available, GpuFft3d, GpuFft3dBuffers};

#[cfg(feature = "native-f16")]
pub use infrastructure::gpu_fft::GpuFft3dF16Native;

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
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|error| ApolloError::BackendUnavailable {
            backend: format!("wgpu adapter unavailable: {error}"),
        })?;
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("apollo-fft-wgpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&descriptor)).map_err(|error| {
                ApolloError::Wgpu {
                    message: error.to_string(),
                }
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
            supports_mixed_precision: true,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
            supported_precision_profiles: vec![
                PrecisionProfile::LOW_PRECISION_F32,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
            ],
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
