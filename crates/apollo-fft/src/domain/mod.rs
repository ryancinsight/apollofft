//! Domain primitives and contracts.

/// Domain contracts.
pub mod contracts;
/// Domain metadata.
pub mod metadata;

/// Compatibility module for backend contracts.
pub mod backend {
    pub use crate::domain::contracts::backend::*;
}

/// Compatibility module for error contracts.
pub mod error {
    pub use crate::domain::contracts::error::*;
}

/// Compatibility module for precision metadata.
pub mod precision {
    pub use crate::domain::metadata::precision::*;
}

/// Compatibility module for shape metadata.
pub mod shape {
    pub use crate::domain::metadata::shape::*;
}
