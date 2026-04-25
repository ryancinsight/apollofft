
pub struct StftGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer:     wgpu::Buffer,
    forward_pipeline:  wgpu::ComputePipeline,
}

impl StftGpuKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("apollo-stft-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stft.wgsl").into()),
        });
        Self { bind_group_layout: todo!(), params_buffer: todo!(), forward_pipeline: todo!() }
    }
}
