
        let level_lens: Vec<usize> = if inverse {
            (0..levels).rev().map(|l| len >> l).collect()
        } else {
            (0..levels).map(|l| len >> l).collect()
        };

        for &current_len in &level_lens {
            let half       = (current_len / 2) as u32;
            let pass_bytes = (current_len * std::mem::size_of::<f32>()) as u64;
            queue.write_buffer(
                &self.params_buffer, 0,
                bytemuck::bytes_of(&WaveletParams {
                    len: current_len as u32, _p0: 0, _p1: 0, _p2: 0,
                }),
            );
            let pipeline = if inverse { &self.synthesis_pipeline } else { &self.analysis_pipeline };
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-wavelet-wgpu pass encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("apollo-wavelet-wgpu pass"), timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(dispatch_count(half), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&temp_buf, 0, &main_buf, 0, pass_bytes);
            queue.submit(std::iter::once(encoder.finish()));
        }
