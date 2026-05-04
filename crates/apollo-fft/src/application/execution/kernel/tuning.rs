/// Chunk-size threshold above which high-radix kernels switch to rayon parallel chunk execution.
pub(crate) const RADIX_PARALLEL_CHUNK_THRESHOLD: usize = 32_768;