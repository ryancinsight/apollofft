# Hybrid Stockham Codelet Leaves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first production-candidate hybrid FFT path by composing Stockham's natural-order contract with fixed-size codelet leaves for `N = 4096 = 8 * 512`.

**Architecture:** Keep `stockham.rs` as the current authoritative implementation file for this vertical slice to avoid a broad move refactor. Add a private mixed-radix-8-by-512 hybrid path that performs a radix-8 column codelet, runs existing fixed-512 Stockham leaves on contiguous rows, then transposes to natural order. Route the path only after value tests and Criterion prove it beats the retained Stockham schedule.

**Tech Stack:** Rust, `num_complex::Complex32/Complex64`, existing Apollo Stockham twiddle tables, AVX/FMA target-feature gates, Criterion, `cargo asm`.

---

## File Structure

- Modify `crates/apollo-fft/src/application/execution/kernel/stockham.rs`: add private hybrid codelets, tests, and gated routing in `forward64_avx_with_scratch`, `inverse64_avx_with_scratch`, `forward32_avx_with_scratch`, and `inverse32_avx_with_scratch`.
- Modify `crates/apollo-fft/benches/kernel_strategy.rs`: add stage-isolation benchmarks for the hybrid column pass, row leaves, transpose, and full hybrid plan.
- Modify `crates/apollo-fft/benches/vs_rustfft.rs`: ensure focused `N=4096` and `N=8192` rows remain covered for same-run RustFFT comparison.
- Modify `backlog.md`, `checklist.md`, `gap_audit.md`, and `CHANGELOG.md`: record promotion or rejection evidence.

## Task 1: Failing Hybrid Correctness Tests

**Files:**
- Modify: `crates/apollo-fft/src/application/execution/kernel/stockham.rs`

- [ ] **Step 1: Add f64 and f32 tests that call the private hybrid functions**

Add these tests inside the existing `#[cfg(test)] mod tests` block:

```rust
#[cfg(target_arch = "x86_64")]
#[test]
fn f64_hybrid_radix8x512_matches_stockham_n4096() {
    if !std::arch::is_x86_feature_detected!("avx")
        || !std::arch::is_x86_feature_detected!("fma")
    {
        return;
    }

    let n = 4096usize;
    let mut expected: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.007).sin(), (k as f64 * 0.011).cos()))
        .collect();
    let mut actual = expected.clone();
    let twiddles =
        crate::infrastructure::cpu::simd::power_of_two::radix2::build_forward_twiddle_table_64(n);
    let mut expected_scratch = vec![Complex64::new(0.0, 0.0); n];
    let mut actual_scratch = vec![Complex64::new(0.0, 0.0); n];

    transform::<F64StockhamAvxFma>(&mut expected, &mut expected_scratch, &twiddles, None);
    unsafe {
        hybrid_radix8x512_64_avx_fma(&mut actual, &mut actual_scratch, &twiddles);
    }

    let err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(err < 1.0e-10, "f64 hybrid radix8x512 err={err:.2e}");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn f32_hybrid_radix8x512_matches_stockham_n4096() {
    if !std::arch::is_x86_feature_detected!("avx")
        || !std::arch::is_x86_feature_detected!("fma")
    {
        return;
    }

    let n = 4096usize;
    let mut expected: Vec<Complex32> = (0..n)
        .map(|k| Complex32::new((k as f32 * 0.007).sin(), (k as f32 * 0.011).cos()))
        .collect();
    let mut actual = expected.clone();
    let twiddles =
        crate::infrastructure::cpu::simd::power_of_two::radix2::build_forward_twiddle_table_32(n);
    let mut expected_scratch = vec![Complex32::new(0.0, 0.0); n];
    let mut actual_scratch = vec![Complex32::new(0.0, 0.0); n];

    transform::<F32StockhamAvxFma>(&mut expected, &mut expected_scratch, &twiddles, None);
    unsafe {
        hybrid_radix8x512_32_avx_fma(&mut actual, &mut actual_scratch, &twiddles);
    }

    let err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f32::max);
    assert!(err < 1.0e-4, "f32 hybrid radix8x512 err={err:.2e}");
}
```

- [ ] **Step 2: Run tests and verify they fail because functions are absent**

Run:

```powershell
cargo test -p apollo-fft hybrid_radix8x512 -- --nocapture
```

Expected result: compile failure naming `hybrid_radix8x512_64_avx_fma` and `hybrid_radix8x512_32_avx_fma` as missing.

## Task 2: Scalar-Correct Hybrid Function

**Files:**
- Modify: `crates/apollo-fft/src/application/execution/kernel/stockham.rs`

- [ ] **Step 1: Add scalar radix-8 column helpers**

Insert before `forward64_avx_with_scratch`:

```rust
#[inline(always)]
fn column_radix8_forward_64(x: [Complex64; 8]) -> [Complex64; 8] {
    let w1 = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, -std::f64::consts::FRAC_1_SQRT_2);
    let w2 = Complex64::new(0.0, -1.0);
    let w3 = Complex64::new(-std::f64::consts::FRAC_1_SQRT_2, -std::f64::consts::FRAC_1_SQRT_2);
    dft8_from_roots_64(x, w1, w2, w3)
}

#[inline(always)]
fn column_radix8_forward_32(x: [Complex32; 8]) -> [Complex32; 8] {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let w1 = Complex32::new(s, -s);
    let w2 = Complex32::new(0.0, -1.0);
    let w3 = Complex32::new(-s, -s);
    dft8_from_roots_32(x, w1, w2, w3)
}

#[inline(always)]
fn dft8_from_roots_64(
    x: [Complex64; 8],
    w1: Complex64,
    w2: Complex64,
    w3: Complex64,
) -> [Complex64; 8] {
    let roots = [
        Complex64::new(1.0, 0.0),
        w1,
        w2,
        w3,
        Complex64::new(-1.0, 0.0),
        -w1,
        -w2,
        -w3,
    ];
    let mut out = [Complex64::new(0.0, 0.0); 8];
    let mut k = 0;
    while k < 8 {
        let mut r = 0;
        while r < 8 {
            out[k] += x[r] * roots[(k * r) & 7];
            r += 1;
        }
        k += 1;
    }
    out
}

#[inline(always)]
fn dft8_from_roots_32(
    x: [Complex32; 8],
    w1: Complex32,
    w2: Complex32,
    w3: Complex32,
) -> [Complex32; 8] {
    let roots = [
        Complex32::new(1.0, 0.0),
        w1,
        w2,
        w3,
        Complex32::new(-1.0, 0.0),
        -w1,
        -w2,
        -w3,
    ];
    let mut out = [Complex32::new(0.0, 0.0); 8];
    let mut k = 0;
    while k < 8 {
        let mut r = 0;
        while r < 8 {
            out[k] += x[r] * roots[(k * r) & 7];
            r += 1;
        }
        k += 1;
    }
    out
}
```

- [ ] **Step 2: Add scalar hybrid bodies**

Add:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn hybrid_radix8x512_64_avx_fma(
    data: &mut [Complex64],
    scratch: &mut [Complex64],
    twiddles: &[Complex64],
) {
    const ROWS: usize = 8;
    const COLS: usize = 512;
    debug_assert_eq!(data.len(), ROWS * COLS);
    debug_assert_eq!(scratch.len(), ROWS * COLS);
    debug_assert!(twiddles.len() >= ROWS * COLS - 1);

    for c in 0..COLS {
        let y = column_radix8_forward_64([
            data[c],
            data[COLS + c],
            data[2 * COLS + c],
            data[3 * COLS + c],
            data[4 * COLS + c],
            data[5 * COLS + c],
            data[6 * COLS + c],
            data[7 * COLS + c],
        ]);
        scratch[c] = y[0];
        for r in 1..ROWS {
            let angle = -std::f64::consts::TAU * (r * c) as f64 / (ROWS * COLS) as f64;
            scratch[r * COLS + c] = y[r] * Complex64::new(angle.cos(), angle.sin());
        }
    }

    for r in 0..ROWS {
        let row = &mut scratch[r * COLS..(r + 1) * COLS];
        let row_scratch = &mut data[r * COLS..(r + 1) * COLS];
        fixed_len512_avx_fma(row, row_scratch, &twiddles[..COLS - 1]);
    }

    for r in 0..ROWS {
        for c in 0..COLS {
            data[c * ROWS + r] = scratch[r * COLS + c];
        }
    }
}
```

Add the f32 hybrid body:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn hybrid_radix8x512_32_avx_fma(
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    twiddles: &[Complex32],
) {
    const ROWS: usize = 8;
    const COLS: usize = 512;
    debug_assert_eq!(data.len(), ROWS * COLS);
    debug_assert_eq!(scratch.len(), ROWS * COLS);
    debug_assert!(twiddles.len() >= ROWS * COLS - 1);

    for c in 0..COLS {
        let y = column_radix8_forward_32([
            data[c],
            data[COLS + c],
            data[2 * COLS + c],
            data[3 * COLS + c],
            data[4 * COLS + c],
            data[5 * COLS + c],
            data[6 * COLS + c],
            data[7 * COLS + c],
        ]);
        scratch[c] = y[0];
        for r in 1..ROWS {
            let angle = -std::f32::consts::TAU * (r * c) as f32 / (ROWS * COLS) as f32;
            scratch[r * COLS + c] = y[r] * Complex32::new(angle.cos(), angle.sin());
        }
    }

    for r in 0..ROWS {
        let row = &mut scratch[r * COLS..(r + 1) * COLS];
        let row_scratch = &mut data[r * COLS..(r + 1) * COLS];
        fixed_len512_32_avx_fma(row, row_scratch, &twiddles[..COLS - 1]);
    }

    for r in 0..ROWS {
        for c in 0..COLS {
            data[c * ROWS + r] = scratch[r * COLS + c];
        }
    }
}
```

- [ ] **Step 3: Run correctness tests**

Run:

```powershell
cargo test -p apollo-fft hybrid_radix8x512 -- --nocapture
```

Expected result: both tests pass or produce a value-mismatch that identifies a factorization/order defect. If a mismatch occurs, fix the factorization before adding any routing.

## Task 3: Remove Per-Column Transcendentals

**Files:**
- Modify: `crates/apollo-fft/src/application/execution/kernel/stockham.rs`

- [ ] **Step 1: Replace `angle.cos()/angle.sin()` with twiddle-table indexing**

In the f64 hybrid body, replace:

```rust
let angle = -std::f64::consts::TAU * (r * c) as f64 / (ROWS * COLS) as f64;
scratch[r * COLS + c] = y[r] * Complex64::new(angle.cos(), angle.sin());
```

with:

```rust
let twiddle = stockham_mixed_twiddle_64::<ROWS, COLS>(twiddles, r, c);
scratch[r * COLS + c] = y[r] * twiddle;
```

Add:

```rust
#[inline(always)]
fn stockham_mixed_twiddle_64<const ROWS: usize, const COLS: usize>(
    twiddles: &[Complex64],
    row: usize,
    col: usize,
) -> Complex64 {
    if row == 0 || col == 0 {
        return Complex64::new(1.0, 0.0);
    }
    let stage_base = (ROWS * COLS) / 2 - 1;
    let exponent = row * col;
    twiddles[stage_base + exponent]
}
```

Add the f32 twiddle lookup:

```rust
#[inline(always)]
fn stockham_mixed_twiddle_32<const ROWS: usize, const COLS: usize>(
    twiddles: &[Complex32],
    row: usize,
    col: usize,
) -> Complex32 {
    if row == 0 || col == 0 {
        return Complex32::new(1.0, 0.0);
    }
    let stage_base = (ROWS * COLS) / 2 - 1;
    let exponent = row * col;
    twiddles[stage_base + exponent]
}
```

- [ ] **Step 2: Run correctness tests**

Run:

```powershell
cargo test -p apollo-fft hybrid_radix8x512 -- --nocapture
```

Expected result: both tests pass.

## Task 4: Add Source Guards

**Files:**
- Modify: `crates/apollo-fft/src/application/execution/kernel/stockham.rs`

- [ ] **Step 1: Add no-bit-reversal/no-allocation hybrid source test**

Add:

```rust
#[test]
fn hybrid_radix8x512_source_has_no_bit_reversal_or_allocation() {
    let source = include_str!("stockham.rs");
    let body = source
        .split_once("unsafe fn hybrid_radix8x512_64_avx_fma")
        .and_then(|(_, tail)| tail.split_once("#[cfg(target_arch = \"x86_64\")]"))
        .map(|(body, _)| body)
        .expect("hybrid radix8x512 body must be present");
    assert!(!body.contains("bit_reverse"));
    assert!(!body.contains("reverse_bits"));
    assert!(!body.contains("bitrev"));
    assert!(!body.contains("Vec<"));
    assert!(!body.contains("vec!"));
    assert!(!body.contains("Box<"));
}
```

- [ ] **Step 2: Run the source guard**

Run:

```powershell
cargo test -p apollo-fft hybrid_radix8x512_source -- --nocapture
```

Expected result: pass.

## Task 5: Benchmark Before Routing

**Files:**
- Modify: `crates/apollo-fft/benches/kernel_strategy.rs`

- [ ] **Step 1: Add benchmark-only public wrappers under `#[cfg(test)]` is rejected**

Do not expose production wrappers for benchmarks. Instead benchmark end-to-end through temporary internal routing in Task 6 only after correctness passes.

- [ ] **Step 2: Run focused current baseline**

Run:

```powershell
cargo bench -p apollo-fft --bench vs_rustfft -- "apollo_fft_vs_rustfft_(f32|f64)/(apollo_zero_alloc_reused|rustfft_zero_alloc_reused)/4096" --warm-up-time 0.2 --measurement-time 0.5 --sample-size 10
```

Expected result: record same-run Apollo and RustFFT medians before routing.

## Task 6: Route Only N=4096 If Correctness Holds

**Files:**
- Modify: `crates/apollo-fft/src/application/execution/kernel/stockham.rs`

- [ ] **Step 1: Gate f64 N=4096 route**

In `forward64_avx_with_scratch`, insert before the generic transform:

```rust
if data.len() == 4096 {
    hybrid_radix8x512_64_avx_fma(data, scratch, twiddles);
    return;
}
```

In `inverse64_avx_with_scratch`, insert the same shape only after adding inverse column roots in Task 7. Do not route inverse earlier.

- [ ] **Step 2: Gate f32 N=4096 route**

In `forward32_avx_with_scratch`, insert before the generic transform:

```rust
if data.len() == 4096 {
    hybrid_radix8x512_32_avx_fma(data, scratch, twiddles);
    return;
}
```

- [ ] **Step 3: Run correctness tests**

Run:

```powershell
cargo test -p apollo-fft stockham -- --nocapture
```

Expected result: all stockham tests pass.

- [ ] **Step 4: Run focused Criterion A/B**

Run:

```powershell
cargo bench -p apollo-fft --bench vs_rustfft -- "apollo_fft_vs_rustfft_(f32|f64)/(apollo_zero_alloc_reused|rustfft_zero_alloc_reused)/4096" --warm-up-time 0.2 --measurement-time 0.5 --sample-size 10
```

Expected result: keep the route only if Apollo N=4096 is faster than the pre-route Apollo baseline and RustFFT in the same run. If it is slower, remove the routing and keep the hybrid function as direct-test-only evidence.

## Task 7: Inverse Path

**Files:**
- Modify: `crates/apollo-fft/src/application/execution/kernel/stockham.rs`

- [ ] **Step 1: Add inverse column roots**

Add inverse variants:

```rust
#[inline(always)]
fn column_radix8_inverse_64(x: [Complex64; 8]) -> [Complex64; 8] {
    let w1 = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2);
    let w2 = Complex64::new(0.0, 1.0);
    let w3 = Complex64::new(-std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2);
    dft8_from_roots_64(x, w1, w2, w3)
}
```

Add the f32 inverse root function:

```rust
#[inline(always)]
fn column_radix8_inverse_32(x: [Complex32; 8]) -> [Complex32; 8] {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let w1 = Complex32::new(s, s);
    let w2 = Complex32::new(0.0, 1.0);
    let w3 = Complex32::new(-s, s);
    dft8_from_roots_32(x, w1, w2, w3)
}
```

- [ ] **Step 2: Generalize hybrid body with `const INVERSE: bool`**

Replace each forward-only helper signature with the const-generic form and
replace the column DFT call site shown here:

```rust
unsafe fn hybrid_radix8x512_64_avx_fma<const INVERSE: bool>(
    data: &mut [Complex64],
    scratch: &mut [Complex64],
    twiddles: &[Complex64],
) {
    let column = [
        data[c],
        data[COLS + c],
        data[2 * COLS + c],
        data[3 * COLS + c],
        data[4 * COLS + c],
        data[5 * COLS + c],
        data[6 * COLS + c],
        data[7 * COLS + c],
    ];
    let y = if INVERSE {
        column_radix8_inverse_64(column)
    } else {
        column_radix8_forward_64(column)
    };
}
```

For f32, replace the helper signature and column DFT call with:

```rust
unsafe fn hybrid_radix8x512_32_avx_fma<const INVERSE: bool>(
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    twiddles: &[Complex32],
) {
    let column = [
        data[c],
        data[COLS + c],
        data[2 * COLS + c],
        data[3 * COLS + c],
        data[4 * COLS + c],
        data[5 * COLS + c],
        data[6 * COLS + c],
        data[7 * COLS + c],
    ];
    let y = if INVERSE {
        column_radix8_inverse_32(column)
    } else {
        column_radix8_forward_32(column)
    };
}
```

Update tests and forward routes to call `<false>`.

- [ ] **Step 3: Add inverse roundtrip tests**

Add:

```rust
#[cfg(target_arch = "x86_64")]
#[test]
fn f64_hybrid_radix8x512_inverse_roundtrip_n4096() {
    if !std::arch::is_x86_feature_detected!("avx")
        || !std::arch::is_x86_feature_detected!("fma")
    {
        return;
    }
    let n = 4096usize;
    let mut data: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.017).sin(), (k as f64 * 0.023).cos()))
        .collect();
    let original = data.clone();
    let fwd =
        crate::infrastructure::cpu::simd::power_of_two::radix2::build_forward_twiddle_table_64(n);
    let inv =
        crate::infrastructure::cpu::simd::power_of_two::radix2::build_inverse_twiddle_table_64(n);
    let mut scratch = vec![Complex64::new(0.0, 0.0); n];
    unsafe {
        hybrid_radix8x512_64_avx_fma::<false>(&mut data, &mut scratch, &fwd);
        hybrid_radix8x512_64_avx_fma::<true>(&mut data, &mut scratch, &inv);
    }
    data.iter_mut().for_each(|z| *z *= 1.0 / n as f64);
    let err = data
        .iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(err < 1.0e-10, "f64 hybrid inverse roundtrip err={err:.2e}");
}
```

Add this f32 roundtrip test:

```rust
#[cfg(target_arch = "x86_64")]
#[test]
fn f32_hybrid_radix8x512_inverse_roundtrip_n4096() {
    if !std::arch::is_x86_feature_detected!("avx")
        || !std::arch::is_x86_feature_detected!("fma")
    {
        return;
    }
    let n = 4096usize;
    let mut data: Vec<Complex32> = (0..n)
        .map(|k| Complex32::new((k as f32 * 0.017).sin(), (k as f32 * 0.023).cos()))
        .collect();
    let original = data.clone();
    let fwd =
        crate::infrastructure::cpu::simd::power_of_two::radix2::build_forward_twiddle_table_32(n);
    let inv =
        crate::infrastructure::cpu::simd::power_of_two::radix2::build_inverse_twiddle_table_32(n);
    let mut scratch = vec![Complex32::new(0.0, 0.0); n];
    unsafe {
        hybrid_radix8x512_32_avx_fma::<false>(&mut data, &mut scratch, &fwd);
        hybrid_radix8x512_32_avx_fma::<true>(&mut data, &mut scratch, &inv);
    }
    data.iter_mut().for_each(|z| *z *= 1.0 / n as f32);
    let err = data
        .iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f32::max);
    assert!(err < 1.0e-4, "f32 hybrid inverse roundtrip err={err:.2e}");
}
```

- [ ] **Step 4: Route inverse only after tests pass**

In `inverse64_avx_with_scratch` and `inverse32_avx_with_scratch`, route `data.len() == 4096` to `hybrid_radix8x512_*_avx_fma::<true>`.

## Task 8: Assembly and Promotion Gate

**Files:**
- Modify: `backlog.md`
- Modify: `checklist.md`
- Modify: `gap_audit.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Inspect assembly**

Run:

```powershell
cargo asm -p apollo-fft --lib --build-type release --no-color apollo_fft::application::execution::kernel::stockham::hybrid_radix8x512_64_avx_fma
```

Expected result: record stack frame size and `%rsp` references. If non-ABI vector spills dominate, reject default routing.

- [ ] **Step 2: Run full local verification**

Run:

```powershell
cargo fmt --check
cargo check -p apollo-fft --benches
cargo test -p apollo-fft stockham -- --nocapture
cargo test -p apollo-fft --lib -- --nocapture
```

Expected result: all pass.

- [ ] **Step 3: Update artifacts with measured decision**

Record one of these exact outcomes:

```markdown
- [x] [patch] Promoted hybrid Stockham/fixed-leaf N=4096 route after same-run Criterion beat RustFFT and current Apollo.
```

or:

```markdown
- [x] [patch] Rejected default hybrid Stockham/fixed-leaf N=4096 route after same-run Criterion or assembly showed a regression; retained correctness-tested codelet for direct diagnostics only.
```

Do not leave the route enabled if it fails the promotion gate.
