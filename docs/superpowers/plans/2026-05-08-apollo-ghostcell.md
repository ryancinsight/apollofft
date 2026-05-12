# Apollo GhostCell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add an Apollo-owned GhostCell crate and use it to separate cached Stockham runtime permission from runtime data without adding hot-path borrow checks.

**Architecture:** `apollo-ghostcell` owns branded permission primitives. `apollo-fft` depends on that crate only in orchestration code; Stockham SIMD kernels continue to receive plain slices and raw pointers after precondition checks.

**Tech Stack:** Rust 2021, `UnsafeCell`, higher-ranked lifetime brands, Cargo workspace path dependency, Apollo FFT Criterion/TDD verification.

---

### Task 1: Add Apollo GhostCell Contract Tests

**Files:**
- Create: `crates/apollo-ghostcell/Cargo.toml`
- Create: `crates/apollo-ghostcell/src/lib.rs`
- Create: `crates/apollo-ghostcell/tests/brand.rs`
- Modify: `Cargo.toml`

- [x] **Step 1: Add the crate manifest and empty library target**

Create `crates/apollo-ghostcell/Cargo.toml` with package metadata and no dependencies. Add `crates/apollo-ghostcell` to the workspace members. Create an empty `src/lib.rs`.

- [x] **Step 2: Write failing tests**

Add tests proving:
- scoped `GhostCell` mutation requires a token;
- `LocalGhostCell` exposes mutation only through an unsafe TLS-style permission root;
- the local root does not expose `RefCell` or `borrow_mut` runtime checks.

- [x] **Step 3: Run red verification**

Run: `cargo test -p apollo-ghostcell --tests -- --nocapture`

Expected: fail with unresolved `GhostCell`, `GhostToken`, or `LocalGhostCell` imports.

### Task 2: Implement Apollo GhostCell

**Files:**
- Modify: `crates/apollo-ghostcell/src/lib.rs`

- [x] **Step 1: Implement `GhostToken<'brand>` and `GhostCell<'brand, T>`**

Use invariant lifetime brands and `UnsafeCell<T>`. Provide `GhostToken::scope`, `GhostCell::new`, `borrow`, `borrow_mut`, `get_mut`, and `into_inner`.

- [x] **Step 2: Implement `LocalGhostCell<T>` and `LocalGhostToken<'brand>`**

Expose a documented unsafe root for thread-local statics:
`unsafe fn with_token<R>(&self, f: impl for<'brand> FnOnce(&mut LocalGhostToken<'brand>, &'brand LocalGhostCell<T>) -> R) -> R`.

- [x] **Step 3: Run green verification**

Run: `cargo test -p apollo-ghostcell --tests -- --nocapture`

Expected: all `apollo-ghostcell` tests pass.

### Task 3: Incorporate Into Stockham Cached Runtime

**Files:**
- Modify: `crates/apollo-fft/Cargo.toml`
- Modify: `crates/apollo-fft/src/application/execution/kernel/mixed_radix.rs`

- [x] **Step 1: Add failing source guard**

Update the cached runtime source guard so production code must contain `LocalGhostCell<StockhamRuntime<Complex64>>` and `LocalGhostCell<StockhamRuntime<Complex32>>`, and must not contain `UnsafeCell<StockhamRuntime`.

- [x] **Step 2: Run red verification**

Run: `cargo test -p apollo-fft cached_stockham_runtime_tls_has_no_refcell_borrow_checks -- --nocapture`

Expected: fail because production still uses `UnsafeCell<StockhamRuntime<_>>`.

- [x] **Step 3: Replace the local unsafe shim**

Import `apollo_ghostcell::LocalGhostCell`, change TLS statics to `LocalGhostCell`, and route `with_stockham_runtime_64/32` through `LocalGhostCell::with_token`.

- [x] **Step 4: Run green verification**

Run: `cargo test -p apollo-fft cached_stockham_runtime_tls_has_no_refcell_borrow_checks -- --nocapture`

Expected: pass.

### Task 4: Verify FFT Behavior And Performance

**Files:**
- Modify: `backlog.md`
- Modify: `checklist.md`
- Modify: `gap_audit.md`
- Modify: `CHANGELOG.md`

- [x] **Step 1: Run compile and targeted tests**

Run:
- `cargo check -p apollo-fft --benches`
- `cargo test -p apollo-ghostcell --tests -- --nocapture`
- `cargo test -p apollo-fft stockham -- --nocapture`

- [x] **Step 2: Run focused benchmark**

Run:
`cargo bench -p apollo-fft --bench vs_rustfft -- "apollo_fft_vs_rustfft_f32/(apollo_zero_alloc_reused|rustfft_zero_alloc_reused)/(64|256|512)" --sample-size 20 --warm-up-time 1 --measurement-time 2`

- [x] **Step 3: Sync sprint artifacts**

Record implementation, verification, and residual performance risk in `backlog.md`, `checklist.md`, `gap_audit.md`, and `CHANGELOG.md`.

