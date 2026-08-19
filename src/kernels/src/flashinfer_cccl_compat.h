#pragma once
// FlashInfer TRT-LLM quantization.cu uses cuda::maximum<> without including it.
// Newer CUB (CUDA 12.8+/13, SM120) no longer transitively provides that name.
#if __has_include(<cuda/functional>)
#include <cuda/functional>
#endif
#if __has_include(<cuda/std/functional>)
#include <cuda/std/functional>
#endif
