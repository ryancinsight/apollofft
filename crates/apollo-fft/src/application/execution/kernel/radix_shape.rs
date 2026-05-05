/// Factorize `n` into a sequence of small radices suitable for the
/// mixed-radix Cooley-Tukey DIT algorithm, or return `None` if `n` has
/// a prime factor outside the supported set {2, 3, 5}.
///
/// ## Radix selection
///
/// - Powers of 2 are packed greedily into radix-8 (3 factors), then radix-4
///   (2 factors), then radix-2 (1 factor).
/// - Primes 3 and 5 appear individually in the sequence.
///
/// The sequence is ordered **innermost first** (smallest sub-transform first):
/// prime radices precede power-of-two radices so that the smallest butterfly
/// groups are processed in the initial stages, improving cache locality.
///
/// ## Contract
///
/// For the returned `radices`, `radices.iter().product::<usize>() == n` and
/// every element ∈ {2, 3, 4, 5, 8}.
///
/// Returns `None` (caller should fall back to Bluestein) when `n` has any
/// prime factor other than 2, 3, or 5.
#[inline]
pub(crate) fn factorize_composite(n: usize) -> Option<Vec<usize>> {
    if n <= 1 {
        return Some(Vec::new());
    }
    let mut remaining = n;
    let mut count2 = 0u32;
    let mut count3 = 0u32;
    let mut count5 = 0u32;
    while remaining % 2 == 0 { count2 += 1; remaining /= 2; }
    while remaining % 3 == 0 { count3 += 1; remaining /= 3; }
    while remaining % 5 == 0 { count5 += 1; remaining /= 5; }
    if remaining > 1 {
        return None; // has prime factor > 5
    }
    // Pure power-of-two sizes are handled by the existing pow2_dispatch! routing;
    // return None so those sizes continue using the optimised PoT kernels.
    if count3 == 0 && count5 == 0 {
        return None;
    }
    let mut radices = Vec::new();
    // Innermost stages: odd prime factors first (5 > 3 ordering for outer loop
    // peeling matches cache footprint: largest prime groups process last).
    for _ in 0..count5 { radices.push(5usize); }
    for _ in 0..count3 { radices.push(3usize); }
    // Outermost stages: powers of 2 packed greedily into 8 > 4 > 2.
    let mut p2 = count2;
    while p2 >= 3 { radices.push(8); p2 -= 3; }
    if p2 >= 2 { radices.push(4); p2 -= 2; }
    if p2 == 1 { radices.push(2); }
    Some(radices)
}

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
