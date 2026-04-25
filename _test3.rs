    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("apollo-stft-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stft.wgsl").into()),
        });
