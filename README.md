**turboquant.hpp**⚡

**Zero-Dependency**, **SIMD-Accelerated KV Cache Compression**

A bare-metal, pure **C++** implementation of the **TurboQuant (ICLR 2026) paper**.

While the open-source community is flooded with heavy PyTorch wrappers and massive framework integrations, this repository goes the opposite direction. No Python. No external dependencies. Just raw math and SIMD intrinsics to crush Large Language Model (LLM) KV caches by 7.8x with near-zero accuracy loss.
🚀 The Metrics (128-dim head, 4-bit)

    Memory: 7.8x compression (e.g., 256 KB KV cache reduced to 33 KB).

    Speed: ~3.20 µs decode latency per vector (AVX2).

    Accuracy: 0.989 attention logit cosine similarity


  **Why This Exists (The Math)**

Standard linear quantization fails on LLM KV caches because neural network activations contain massive, unpredictable "outliers."

Instead of building complex outlier-handling logic, this library uses a Fast Randomized Orthogonal Transform. By applying a Walsh-Hadamard Transform (WHT) combined with a deterministic sign-flip matrix, we mathematically "smear" the vector energy uniformly across all dimensions. The outliers disappear into a predictable Beta distribution.

Once uniform, we apply Lloyd-Max quantization to find the mathematically optimal centroids for 3-bit or 4-bit packing.

The best part? Because the WHT preserves inner products, you can compute attention dot products directly in the quantized domain using Fused Multiply-Add (FMA) instructions, without ever dequantizing the Key cache.


 **Features**

    1.Single Header: Drop turboquant.hpp into your project and you're done.

    2.SIMD Tiers: Auto-detects and compiles for AVX2+FMA (Tier 1), SSE4.1 (Tier 2), or Scalar C++ (Tier 0).

    3.Data Oblivious: The codebooks are generated algorithmically during initialization based on the Beta distribution; no offline calibration data needed.

💻 Quick Start

1. Include the header
C++
```


#include "turboquant.hpp"
#include <vector>

int main() {
    int dim = 128;
    int bits = 4; // 4-bit compression
    
    // Initialize the quantizer (pre-computes Lloyd-Max codebooks)
    turboquant::TurboQuantizer tq(dim, bits);
    
    std::vector<float> my_vector(dim, 0.5f); // Your FP32 data
    std::vector<uint8_t> compressed_block(tq.block_size());
    std::vector<float> decompressed_vector(dim);
    
    // Encode to compressed bytes
    tq.encode(my_vector.data(), compressed_block.data());
    
    // Decode back to FP32
    tq.decode(compressed_block.data(), decompressed_vector.data());
    
    // Fast Attention: Compute dot product directly on compressed data
    float fast_dot = tq.dot(compressed_block.data(), compressed_block.data());
    
    return 0;
}
```

2. Compile with SIMD flags
To get the sub-microsecond speeds, ensure you compile with AVX2 and FMA enabled:
Bash
```
g++ -O3 -std=c++17 -mavx2 -mfma your_file.cpp -o your_app
```

🗺️ Roadmap

    [x] Fast Walsh-Hadamard Transform (AVX2/SSE4)

    [x] Lloyd-Max Codebook Generation

    [x] Quantized-Domain Dot Products

    [ ] ARM Neon Support (for Apple Silicon / M-series Macs)

    [ ] llama.cpp GGML Tensor Bridge Example