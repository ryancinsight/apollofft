//! `rustfft`-based reference transforms used by validation.

use ndarray::{Array1, Array3};
use num_complex::Complex64;
use rustfft::FftPlanner;

/// Execute a reference 1D FFT using `rustfft`.
pub fn fft1_real(input: &Array1<f64>) -> Vec<Complex64> {
    let mut buffer: Vec<Complex64> = input
        .iter()
        .map(|value| Complex64::new(*value, 0.0))
        .collect();
    FftPlanner::<f64>::new()
        .plan_fft_forward(buffer.len())
        .process(&mut buffer);
    buffer
}

/// Execute a separable 3D FFT using `rustfft`.
pub fn fft3_real(input: &Array3<f64>) -> Array3<Complex64> {
    let shape = input.dim();
    let mut data = Array3::from_shape_fn(shape, |(i, j, k)| Complex64::new(input[(i, j, k)], 0.0));
    let mut planner = FftPlanner::<f64>::new();
    let fft_x = planner.plan_fft_forward(shape.0);
    let fft_y = planner.plan_fft_forward(shape.1);
    let fft_z = planner.plan_fft_forward(shape.2);

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            let mut lane: Vec<Complex64> = (0..shape.2).map(|k| data[(i, j, k)]).collect();
            fft_z.process(&mut lane);
            for (k, value) in lane.into_iter().enumerate() {
                data[(i, j, k)] = value;
            }
        }
    }

    for i in 0..shape.0 {
        for k in 0..shape.2 {
            let mut lane: Vec<Complex64> = (0..shape.1).map(|j| data[(i, j, k)]).collect();
            fft_y.process(&mut lane);
            for (j, value) in lane.into_iter().enumerate() {
                data[(i, j, k)] = value;
            }
        }
    }

    for j in 0..shape.1 {
        for k in 0..shape.2 {
            let mut lane: Vec<Complex64> = (0..shape.0).map(|i| data[(i, j, k)]).collect();
            fft_x.process(&mut lane);
            for (i, value) in lane.into_iter().enumerate() {
                data[(i, j, k)] = value;
            }
        }
    }

    data
}
