//! 1D Number Theoretic Transform Plan

use crate::application::execution::kernel::direct::ntt_kernel;
use crate::domain::contracts::config::{DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT};
use crate::domain::contracts::error::NttError;
use crate::domain::contracts::math::{mod_inv, mod_mul, mod_pow};
use ndarray::Array1;
use serde::{Deserialize, Serialize};