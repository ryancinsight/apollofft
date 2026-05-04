#[inline]
pub(crate) fn is_power_of_pow2_radix(n: usize, radix_log2: u32) -> bool {
    n.is_power_of_two() && (n.trailing_zeros() % radix_log2 == 0)
}

#[inline]
pub(crate) fn is_power_of_four(n: usize) -> bool {
    is_power_of_pow2_radix(n, 2)
}

#[inline]
pub(crate) fn is_power_of_eight(n: usize) -> bool {
    is_power_of_pow2_radix(n, 3)
}

#[inline]
pub(crate) fn stage_twiddle_slice<T>(twiddles: Option<&[T]>, half: usize) -> Option<&[T]> {
    twiddles.map(|t| &t[(half - 1)..(half - 1 + half)])
}

/// Look up the radix-4 stage twiddle at position `exponent` within a stage slice
/// of nominal length `half`.
///
/// ## Mathematical contract
///
/// In a radix-4 stage with group length `len = 4·half`, the DIT butterfly uses
/// twiddles `W^{j}`, `W^{2j}`, `W^{3j}` for j = 0..half. The exponent may reach
/// `3·half - 1`. For exponents beyond `half - 1` the symmetry
/// `W^{half} = W^{half mod half}·(-1) = -W^{exponent mod half}` holds because
/// `W^{half} = exp(2πi·half/len) = exp(πi) = -1`. Hence:
///
/// ```text
/// stage_twiddle(stage, half, exponent) =
///     if exponent < half:  stage[exponent]
///     else:               -stage[exponent - half]
/// ```
///
/// This avoids storing more than `half` entries per stage while supporting all
/// three twiddle exponents.
///
/// ## Generic bound
///
/// `T` requires `Copy` (to read without consuming) and `Neg<Output = T>`
/// (to negate without allocation). Both `Complex64` and `Complex32` satisfy this.
#[inline]
pub(crate) fn stage_twiddle<T>(stage: &[T], half: usize, exponent: usize) -> T
where
    T: Copy + std::ops::Neg<Output = T>,
{
    if exponent < half {
        stage[exponent]
    } else {
        -stage[exponent - half]
    }
}