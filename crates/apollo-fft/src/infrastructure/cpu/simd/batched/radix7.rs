//! Batched radix-7 native-precision kernel entry points.

use crate::domain::storage::FftStorage;

/// Execute eight independent planar radix-7 DFT instances in f32.
pub fn radix7_batched_f32<S: FftStorage<f32>>(storage: &mut S, inverse: bool) {
    super::prime::prime_batched_f32::<7, S>(storage, inverse);
}
