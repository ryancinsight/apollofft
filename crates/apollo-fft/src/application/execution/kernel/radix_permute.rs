#[inline]
fn reverse_base_radix(mut value: usize, radix: usize, digits: u32) -> usize {
    let mut reversed = 0usize;
    for _ in 0..digits {
        reversed = reversed * radix + (value % radix);
        value /= radix;
    }
    reversed
}

/// O(N) in-place bit-reversal permutation for any element type.
///
/// Uses the iterative XOR / binary-counter-in-reverse technique: maintains
/// variable `j` such that after step `i`, `j = bit_reverse(i, log₂N)`.
/// Each bit of `j` is updated by propagating carry from MSB toward LSB.
/// Amortised cost ≈ 2 operations per element (geometric series), making
/// this O(N) rather than the O(N log N) of the digit-wise reverse approach.
///
/// # Precondition
///
/// `data.len()` must be a power of two (debug-asserted).
#[inline]
pub(crate) fn bit_reverse_permute<T>(data: &mut [T]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    if n <= 1 {
        return;
    }
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

#[inline]
pub(crate) fn digit_reverse_permute_pow2_radix<const RADIX: usize, T>(data: &mut [T]) {
    debug_assert!(RADIX.is_power_of_two(), "RADIX must be a power of two");
    if data.len() <= 1 {
        return;
    }
    let radix_log2 = RADIX.trailing_zeros() as usize;
    debug_assert!(radix_log2 > 0, "RADIX must be greater than 1");
    debug_assert!(
        data.len().is_power_of_two(),
        "digit-reversal expects power-of-two length"
    );
    debug_assert!(
        (data.len().trailing_zeros() as usize) % radix_log2 == 0,
        "length must be an exact power of RADIX"
    );

    // Radix-2 digit-reversal is bit-reversal; use the O(N) amortised XOR algorithm.
    if RADIX == 2 {
        bit_reverse_permute(data);
        return;
    }

    let digits = (data.len().trailing_zeros() as usize) / radix_log2;
    for index in 0..data.len() {
        let reversed = reverse_base_radix(index, RADIX, digits as u32);
        if reversed > index {
            data.swap(index, reversed);
        }
    }
}