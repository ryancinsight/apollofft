#![allow(missing_docs)]

use apollo_ghostcell::{GhostCell, GhostToken, LocalGhostCell};

#[test]
fn scoped_ghost_cell_mutates_through_brand_token() {
    let result = GhostToken::scope(|token| {
        let cell = GhostCell::new(3usize);
        *cell.borrow_mut(token) = 13;
        *cell.borrow(token)
    });

    assert_eq!(result, 13);
}

#[test]
fn scoped_ghost_cell_consumes_inner_value() {
    let value = GhostToken::scope(|token| {
        let mut cell = GhostCell::new(String::from("fft"));
        cell.get_mut().push_str("-stockham");
        assert_eq!(cell.borrow(token).as_str(), "fft-stockham");
        cell.into_inner()
    });

    assert_eq!(value, "fft-stockham");
}

#[test]
fn local_ghost_cell_mutates_only_under_permission_root() {
    let cell = LocalGhostCell::new(5usize);

    let observed = unsafe {
        cell.with_token(|token, branded| {
            *branded.borrow_mut(token) = 21;
            *branded.borrow(token)
        })
    };

    assert_eq!(observed, 21);
}

#[test]
fn local_ghost_cell_source_has_no_runtime_borrow_checks() {
    let source = std::fs::read_to_string("src/lib.rs").expect("read source");

    assert!(
        !source.contains("RefCell"),
        "apollo-ghostcell must not implement branding through RefCell"
    );
    assert!(
        !source.contains(".borrow_mut()"),
        "apollo-ghostcell must not use runtime borrow_mut checks"
    );
}
