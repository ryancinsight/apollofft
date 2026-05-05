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

/// Compute the digit-reversed index for `i` in the mixed-radix system defined by
/// `radices[0]` (innermost, least-significant) to `radices[L-1]` (outermost).
///
/// ## Mathematical contract
///
/// Represent `i` in the mixed-radix positional system:
/// `i = d₀ + r₀·(d₁ + r₁·(d₂ + …))` where `dₖ = (i / ∏_{j<k} rⱼ) mod rₖ`.
/// The reversed index is `i_rev = d_{L-1} + r_{L-1}·(d_{L-2} + r_{L-2}·(…))`.
/// For the power-of-two case (all rₖ = 2), this reduces to standard bit-reversal.
#[inline]
pub(crate) fn digit_reverse_mixed(mut i: usize, radices: &[usize]) -> usize {
    let mut result = 0;
    for &r in radices {
        result = result * r + (i % r);
        i /= r;
    }
    result
}

/// In-place digit-reversal permutation for a mixed-radix system.
///
/// ## Correctness
///
/// Let σ(i) = `digit_reverse_mixed(i, radices)`.  We want to achieve
/// `data[i] := old_data[σ(i)]` — a **gather** from σ(i) into position i.
///
/// For radix-2, σ is self-inverse so the naive "swap when σ(i) > i" works.
/// For heterogeneous radices (e.g., [3, 2]), σ can contain cycles of length > 2
/// and the naive algorithm silently produces the wrong permutation.
///
/// This implementation uses **cycle-following** for the gather form:
///   for cycle (c₀ → c₁ → c₂ → … → c_{k-1} → c₀) under σ:
///   - save tmp = data[c₀]
///   - data[c₀] = data[c₁]   (still old_data[c₁] — not yet overwritten)
///   - data[c₁] = data[c₂]
///   - …
///   - data[c_{k-1}] = tmp    (= old_data[c₀])
///
/// Each position is read before it is written, so no auxiliary element is
/// needed beyond `tmp`.  A `visited` bitmap prevents re-processing cycles.
///
/// ## Complexity
///
/// O(N·L) time, O(N) space for the `visited` bitmap (one bit per position).
/// L = radices.len().
pub(crate) fn digit_reverse_permute_mixed<T: Copy>(data: &mut [T], radices: &[usize]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let mut visited = vec![false; n];
    for start in 0..n {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let j0 = digit_reverse_mixed(start, radices); // σ(start)
        if j0 == start {
            // Fixed point: position start maps to itself.
            continue;
        }
        // Gather cycle: data[i] := old_data[σ(i)].
        // Traverse forward in the cycle chain (start → σ(start) → σ²(start) → …).
        // Each data[i] is read from σ(i) before position i is overwritten.
        let saved = data[start];
        let mut i = start;
        let mut j = j0; // j = σ(i)
        while j != start {
            data[i] = data[j]; // gather: position i receives old_data at σ(i) = j
            visited[i] = true;
            i = j;
            j = digit_reverse_mixed(i, radices); // σ(new i)
        }
        data[i] = saved; // close the cycle: last position gets old_data[start]
        visited[i] = true;
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
