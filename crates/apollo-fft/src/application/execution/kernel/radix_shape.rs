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
