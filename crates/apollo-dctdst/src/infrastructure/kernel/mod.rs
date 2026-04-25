/// Direct analytical DCT/DST kernels (O(N²)).
pub mod direct;
/// Fast FFT-based DCT/DST kernels (O(N log N)).
pub mod fast;

pub use fast::FAST_THRESHOLD;
