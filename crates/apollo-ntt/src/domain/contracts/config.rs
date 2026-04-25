//! Default configuration mappings.

/// NTT-friendly prime 998_244_353 = 119 * 2^23 + 1. Has multiplicative order 2^23, supporting NTT lengths up to 2^23.
pub const DEFAULT_MODULUS: u64 = 998_244_353;

/// 3 is a primitive root modulo 998_244_353. Order = 998_244_352 = (p-1).
pub const DEFAULT_PRIMITIVE_ROOT: u64 = 3;
