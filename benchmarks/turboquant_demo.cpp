/**
 * turboquant_demo.cpp
 *
 * Build:  g++ -O3 -std=c++17 turboquant_demo.cpp -o turboquant_demo
 * Run:    ./turboquant_demo
 */

#include "turboquant.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static float mse(const float* a, const float* b, int n) {
    float e = 0;
    for (int i = 0; i < n; ++i) { float d = a[i]-b[i]; e += d*d; }
    return e / n;
}

static float cosine_sim(const float* a, const float* b, int n) {
    float dot=0, na=0, nb=0;
    for (int i=0;i<n;++i){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
    return dot / (std::sqrt(na)*std::sqrt(nb)+1e-12f);
}

static std::vector<float> rand_vec(int d, std::mt19937& rng) {
    std::normal_distribution<float> nd(0,1);
    std::vector<float> v(d);
    for (float& x : v) x = nd(rng);
    return v;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    using namespace turboquant;
    using clk = std::chrono::high_resolution_clock;

    std::mt19937 rng(42);

    printf("=== TurboQuant C++ Demo ===\n\n");

    // ------------------------------------------------------------------
    // 1. Basic encode/decode correctness
    // ------------------------------------------------------------------
    printf("--- Correctness (d=128) ---\n");
    for (int bits : {3, 4}) {
        TurboQuantizer tq(128, bits);
        int bs = tq.block_size();

        // FP16 baseline (2 bytes per float)
        float fp16_bytes = 128 * 2.0f;
        float compression = fp16_bytes / bs;

        float total_mse = 0, total_cos = 0;
        int N = 1000;
        std::vector<uint8_t> block(bs);
        std::vector<float> rec(128);

        for (int i = 0; i < N; ++i) {
            auto v = rand_vec(128, rng);
            tq.encode(v.data(), block.data());
            tq.decode(block.data(), rec.data());
            total_mse += mse(v.data(), rec.data(), 128);
            total_cos += cosine_sim(v.data(), rec.data(), 128);
        }

        printf("  TQ%d | block=%d bytes | compression=%.1fx | MSE=%.5f | cosine=%.5f\n",
               bits, bs, compression, total_mse/N, total_cos/N);
    }

    // ------------------------------------------------------------------
    // 2. Quantized dot product vs FP32 dot product
    // ------------------------------------------------------------------
    printf("\n--- Quantized inner product accuracy (d=128, 4-bit) ---\n");
    {
        TurboQuantizer tq(128, 4);
        int bs = tq.block_size();
        float dot_err = 0;
        int N = 5000;
        std::vector<uint8_t> ba(bs), bb(bs);

        for (int i = 0; i < N; ++i) {
            auto a = rand_vec(128, rng);
            auto b = rand_vec(128, rng);

            // FP32 reference dot
            float ref = 0;
            for (int j = 0; j < 128; ++j) ref += a[j]*b[j];

            tq.encode(a.data(), ba.data());
            tq.encode(b.data(), bb.data());
            float approx = tq.dot(ba.data(), bb.data());

            dot_err += std::abs(approx - ref) / (std::abs(ref) + 1e-6f);
        }
        printf("  Mean relative inner-product error: %.4f%%\n", dot_err/N*100);
    }

    // ------------------------------------------------------------------
    // 3. KV cache simulation: attention logits
    // ------------------------------------------------------------------
    printf("\n--- KV cache attention logit accuracy (seq=512, d=128, 4-bit) ---\n");
    {
        int seq = 512, d = 128;
        TurboQuantizer tq(d, 4);

        // Generate random keys and query
        std::vector<float> keys(seq*d), query(d);
        for (float& v : keys)  { std::normal_distribution<float> nd(0,1); v=nd(rng); }
        for (float& v : query) { std::normal_distribution<float> nd(0,1); v=nd(rng); }

        // Encode key cache
        std::vector<uint8_t> kenc(seq * tq.block_size());
        tq.encode_cache(keys.data(), seq, kenc.data());

        // FP32 reference logits
        std::vector<float> ref_logits(seq), tq_logits(seq);
        for (int t = 0; t < seq; ++t) {
            float dot = 0;
            for (int j = 0; j < d; ++j) dot += query[j] * keys[t*d+j];
            ref_logits[t] = dot;
        }

        // Quantized logits
        tq.attention_logits(query.data(), kenc.data(), seq, tq_logits.data());

        // Compute cosine between logit vectors (what attention softmax sees)
        float cs = cosine_sim(ref_logits.data(), tq_logits.data(), seq);
        printf("  Attention logit cosine similarity: %.6f\n", cs);

        // Memory: FP32 KV cache vs TurboQuant
        float fp32_kb = seq * d * 4.0f / 1024;
        float tq_kb   = seq * tq.block_size() / 1024.0f;
        printf("  KV cache: FP32=%.1f KB  TQ4=%.1f KB  (%.1fx smaller)\n",
               fp32_kb, tq_kb, fp32_kb/tq_kb);
    }

    // ------------------------------------------------------------------
    // 4. Throughput benchmark
    // ------------------------------------------------------------------
    printf("\n--- Throughput (d=128, 4-bit, 10k encode+decode cycles) ---\n");
    {
        TurboQuantizer tq(128, 4);
        int N = 10000;
        std::vector<uint8_t> block(tq.block_size());
        std::vector<float> rec(128);
        auto v = rand_vec(128, rng);

        auto t0 = clk::now();
        for (int i = 0; i < N; ++i) tq.encode(v.data(), block.data());
        auto t1 = clk::now();
        for (int i = 0; i < N; ++i) tq.decode(block.data(), rec.data());
        auto t2 = clk::now();

        double enc_us = std::chrono::duration<double,std::micro>(t1-t0).count() / N;
        double dec_us = std::chrono::duration<double,std::micro>(t2-t1).count() / N;
        printf("  Encode: %.2f µs/vec   Decode: %.2f µs/vec\n", enc_us, dec_us);
    }

    // ------------------------------------------------------------------
    // 5. Codebook inspection
    // ------------------------------------------------------------------
    printf("\n--- 3-bit codebook centroids (d=128) ---\n");
    {
        TurboQuantizer tq(128, 3);
        printf("  [");
        for (float c : tq.cb.centroids) printf("%.4f ", c);
        printf("]\n");
    }

    printf("\nDone.\n");
    return 0;
}