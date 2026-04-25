
    fn run_passes(
        &self,
        device:  &wgpu::Device,
        queue:   &wgpu::Queue,
        input:   &[f32],
        len:     usize,
        levels:  usize,
        inverse: bool,
    ) -> WgpuResult<Vec<f32>> {
        let byte_len = (len * std::mem::size_of::<f32>()) as u64;
        let level_lens: Vec<usize> = if inverse {
            (0..levels).rev().map(|l| len >> l).collect()
        } else {
            (0..levels).map(|l| len >> l).collect()
        };
        Ok(output)
    }
