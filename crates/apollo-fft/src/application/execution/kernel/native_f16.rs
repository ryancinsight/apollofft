//! Native-f16 execution gate.

use super::radix2_f16::Cf16;
use num_complex::Complex32;

const NATIVE_F16_UNAVAILABLE: &str =
    "native f16 FFT requires a native f16 CPU backend; f32 widening is forbidden";

/// Reject f16 execution for kernels that would otherwise widen through f32.
pub(crate) fn run_f16_native_only<F>(data: &mut [Cf16], _kernel: F)
where
    F: FnOnce(&mut [Complex32]),
{
    reject_native_f16(data);
}

/// Reject direct f16 kernel execution on unsupported stable CPU backends.
pub(crate) fn reject_native_f16(data: &[Cf16]) {
    if data.len() <= 1 {
        return;
    }
    panic!("{NATIVE_F16_UNAVAILABLE}");
}
