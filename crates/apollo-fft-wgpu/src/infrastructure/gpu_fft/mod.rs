//! Shader-backed 3D FFT implementation for the Apollo WGPU backend.

pub mod dispatch;
pub mod pipeline;
pub mod strategy;
pub mod workspace;

pub use pipeline::{gpu_fft_available, GpuFft3d};
pub use workspace::GpuFft3dBuffers;
