//! Debug binary for f32 FFT parity inspection.

use apollo_fft::application::execution::kernel::{direct::dft_forward_32, fft_forward};
use num_complex::Complex32;

fn main() {
    let n = 45usize;
    let mut input = vec![Complex32::new(0.0, 0.0); n];
    for i in 0..n {
        input[i] = Complex32::new(i as f32, (i * 2) as f32);
    }
    let mut generic = input.clone();
    fft_forward(&mut generic);
    let direct = dft_forward_32(&input);

    let mut found = 0;
    for i in 0..n {
        let diff = (generic[i] - direct[i]).norm();
        if diff > 1.0e-3 || generic[i].re.is_nan() {
            println!(
                "Mismatch at {}: generic={:?}, direct={:?}",
                i, generic[i], direct[i]
            );
            found += 1;
            if found > 10 {
                break;
            }
        }
    }
    if found == 0 {
        println!("SUCCESS!");
    }
}
