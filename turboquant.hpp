/**
 * turboquant.hpp  —  TurboQuant (ICLR 2026), header-only C++
 *
 * SIMD acceleration:
 *   Tier 1 (AVX2+FMA, -mavx2 -mfma)    : WHT, sign-flip, dot product
 *   Tier 2 (SSE4.1, -msse4.1)           : partial SIMD, 4-wide
 *   Tier 0 (scalar)                      : always-available fallback
 *
 * Build examples:
 *   g++ -O3 -std=c++17 -mavx2 -mfma    demo.cpp   # full SIMD
 *   g++ -O3 -std=c++17 -msse4.1        demo.cpp   # SSE fallback
 *   g++ -O3 -std=c++17                  demo.cpp   # scalar fallback
 *
 * No external dependencies.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

// ── SIMD capability detection ─────────────────────────────────────────────────
#if defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
  #define TQ_HAVE_AVX2 1
#elif defined(__SSE4_1__)
  #include <smmintrin.h>
  #define TQ_HAVE_SSE4 1
#endif

namespace turboquant {

// =============================================================================
// §1  FP16 helpers  (storage only — no arithmetic)
// =============================================================================

static inline uint16_t f32_to_f16(float v) {
    uint32_t u; std::memcpy(&u, &v, 4);
    uint16_t sign = (u >> 16) & 0x8000;
    int32_t  exp  = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x7FFFFF;
    if (exp <= 0)  return sign;
    if (exp >= 31) return sign | 0x7C00;
    return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0)  return 0.0f;
    if (exp == 31) return std::numeric_limits<float>::infinity();
    uint32_t u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f; std::memcpy(&f, &u, 4);
    return f;
}

// =============================================================================
// §2  Walsh-Hadamard Transform
//
//  Math recap:
//    WHT is its own inverse (self-adjoint unitary over R).
//    We normalize by 1/sqrt(n) so H·Ht = I → length-preserving.
//    Combined with the sign-flip diagonal D, R = D·H is the cheap
//    random orthogonal transform that spreads vector energy uniformly.
//
//    Butterfly structure:  for stride s = 1,2,4,...,n/2
//      for each pair (i, i+s):
//        (a, b) -> (a+b, a-b)
//    After log2(n) passes: O(n log n) total.  No trig, no complex numbers.
//
//  SIMD strategy (AVX2):
//    Process 8 floats per register.  In each butterfly pass, load pairs
//    of 256-bit lanes, compute sum and difference simultaneously using
//    _mm256_add_ps / _mm256_sub_ps, store back.  For strides < 8 we
//    fall through to a scalar sub-loop.
// =============================================================================

#ifdef TQ_HAVE_AVX2

static inline void wht_pass_avx2(float* __restrict__ x, int n, int stride) {
    for (int i = 0; i < n; i += stride << 1) {
        float* a = x + i;
        float* b = x + i + stride;
        for (int j = 0; j < stride; j += 8) {
            __m256 va = _mm256_loadu_ps(a + j);
            __m256 vb = _mm256_loadu_ps(b + j);
            _mm256_storeu_ps(a + j, _mm256_add_ps(va, vb));
            _mm256_storeu_ps(b + j, _mm256_sub_ps(va, vb));
        }
    }
}

static void wht(float* x, int n) {
    int stride = 1;
    // Scalar passes for stride 1, 2, 4 (< one AVX2 register width)
    for (; stride < 8 && stride < n; stride <<= 1)
        for (int i = 0; i < n; i += stride << 1)
            for (int j = i; j < i + stride; ++j) {
                float a = x[j], b = x[j + stride];
                x[j] = a + b;  x[j + stride] = a - b;
            }
    // AVX2 passes for stride >= 8
    for (; stride < n; stride <<= 1)
        wht_pass_avx2(x, n, stride);
    // Normalize 1/sqrt(n) using AVX2
    __m256 scale = _mm256_set1_ps(1.0f / std::sqrt(static_cast<float>(n)));
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), scale));
    float s = 1.0f / std::sqrt(static_cast<float>(n));
    for (; i < n; ++i) x[i] *= s;
}

#elif defined(TQ_HAVE_SSE4)

static inline void wht_pass_sse(float* __restrict__ x, int n, int stride) {
    for (int i = 0; i < n; i += stride << 1) {
        float* a = x + i, *b = x + i + stride;
        for (int j = 0; j < stride; j += 4) {
            __m128 va = _mm_loadu_ps(a + j);
            __m128 vb = _mm_loadu_ps(b + j);
            _mm_storeu_ps(a + j, _mm_add_ps(va, vb));
            _mm_storeu_ps(b + j, _mm_sub_ps(va, vb));
        }
    }
}

static void wht(float* x, int n) {
    int stride = 1;
    for (; stride < 4 && stride < n; stride <<= 1)
        for (int i = 0; i < n; i += stride << 1)
            for (int j = i; j < i + stride; ++j) {
                float a = x[j], b = x[j + stride];
                x[j] = a + b;  x[j + stride] = a - b;
            }
    for (; stride < n; stride <<= 1)
        wht_pass_sse(x, n, stride);
    __m128 scale = _mm_set1_ps(1.0f / std::sqrt(static_cast<float>(n)));
    int i = 0;
    for (; i + 4 <= n; i += 4)
        _mm_storeu_ps(x + i, _mm_mul_ps(_mm_loadu_ps(x + i), scale));
    float s = 1.0f / std::sqrt(static_cast<float>(n));
    for (; i < n; ++i) x[i] *= s;
}

#else  // scalar fallback

static void wht(float* x, int n) {
    for (int stride = 1; stride < n; stride <<= 1)
        for (int i = 0; i < n; i += stride << 1)
            for (int j = i; j < i + stride; ++j) {
                float a = x[j], b = x[j + stride];
                x[j] = a + b;  x[j + stride] = a - b;
            }
    float s = 1.0f / std::sqrt(static_cast<float>(n));
    for (int i = 0; i < n; ++i) x[i] *= s;
}

#endif  // SIMD WHT

// Self-inverse: decode just calls wht again
static inline void iwht(float* x, int n) { wht(x, n); }


// =============================================================================
// §3  Deterministic sign flips  D = diag(+-1)
//
//  A diagonal matrix of +-1 is orthogonal (Dt = D^-1 = D).
//  Combined with WHT: R = D*H is a fast randomized orthogonal transform.
//  The sign pattern comes from a xorshift64 PRNG seeded by `seed`.
//  Same seed -> same signs -> decode undoes encode by applying them again
//  (since +-1 * +-1 = +1, the double application cancels).
//
//  SIMD strategy (AVX2):
//    We generate 64 sign bits at a time via xorshift64.
//    _mm256_set1_ps(-0.0f) gives the IEEE sign bit mask (0x80000000).
//    We expand each bit of the PRNG output to a full 32-bit lane:
//      bit=0 -> 0x00000000 (XOR leaves sign unchanged -> positive)
//      bit=1 -> 0x80000000 (XOR flips sign bit -> negative)
//    Then _mm256_xor_ps flips the sign bits in one shot:
//    8 sign flips per instruction instead of 8 multiplications.
// =============================================================================

#ifdef TQ_HAVE_AVX2

static void apply_signs(float* __restrict__ x, int n, uint64_t seed) {
    uint64_t state = seed ^ 0x9E3779B97F4A7C15ULL;
    const __m256i sign_bit = _mm256_set1_epi32(static_cast<int>(0x80000000u));
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        uint8_t byte = static_cast<uint8_t>(state);
        // expand 8 bits into 8 full-lane masks
        __m256i mask = _mm256_set_epi32(
            (byte >> 7) & 1 ? -1 : 0,  (byte >> 6) & 1 ? -1 : 0,
            (byte >> 5) & 1 ? -1 : 0,  (byte >> 4) & 1 ? -1 : 0,
            (byte >> 3) & 1 ? -1 : 0,  (byte >> 2) & 1 ? -1 : 0,
            (byte >> 1) & 1 ? -1 : 0,  (byte >> 0) & 1 ? -1 : 0
        );
        __m256i flip = _mm256_and_si256(mask, sign_bit);
        __m256 vx    = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_xor_ps(vx, _mm256_castsi256_ps(flip)));
    }
    // scalar tail
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    for (int j = i; j < n; ++j)
        if ((state >> (j - i)) & 1) x[j] = -x[j];
}

#else  // scalar fallback

static void apply_signs(float* x, int n, uint64_t seed) {
    uint64_t state = seed ^ 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < n; i += 64) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        uint64_t bits = state;
        int end = std::min(i + 64, n);
        for (int j = i; j < end; ++j)
            if ((bits >> (j - i)) & 1) x[j] = -x[j];
    }
}

#endif // SIMD signs


// =============================================================================
// §4  Lloyd-Max codebook computation
//
//  After WHT+signs, each coordinate of a unit vector in R^d follows:
//    f(x) ∝ (1 - x^2)^((d-3)/2)   x in [-1, 1]
//  (Beta((d-1)/2, (d-1)/2) distribution on [-1,1]).
//  For d=128 this is extremely concentrated around 0 (exponent 62.5).
//
//  Lloyd-Max iteration:
//    Given k = 2^bits quantization levels, find centroids c0...c_{k-1}
//    and boundaries t0...t_{k-2} minimizing E[(X - Q(X))^2].
//
//    At optimum, two conditions must hold simultaneously:
//      (A) boundaries are midpoints:  t_i = (c_i + c_{i+1}) / 2
//      (B) centroids are cond. means: c_i = E[X | t_{i-1} < X <= t_i]
//
//    Algorithm:
//      1. Initialize centroids uniformly on (-0.9, 0.9)
//      2. Compute boundaries from (A)
//      3. Compute new centroids from (B) via numerical integration
//      4. Repeat until convergence (~150 iterations typically)
//
//  The codebook is computed once at TurboQuantizer construction.
//  It only depends on (dim, bits) — fully data-oblivious.
// =============================================================================

struct Codebook {
    std::vector<float> centroids;    // k = 2^bits values
    std::vector<float> boundaries;   // k-1 decision thresholds (sorted)
    int bits = 0;
};

static std::pair<double,double>
integrate_beta_region(double lo, double hi, double alpha, int steps = 512) {
    double sum_f = 0, sum_xf = 0, dx = (hi - lo) / steps;
    for (int i = 0; i < steps; ++i) {
        double mid = lo + (i + 0.5) * dx;
        double f   = std::pow(1.0 - mid * mid, alpha);
        sum_f  += f;
        sum_xf += mid * f;
    }
    return {sum_f * dx, sum_xf * dx};
}

static Codebook compute_lloyd_max(int dim, int bits, int max_iter = 300) {
    int    k     = 1 << bits;
    double alpha = std::max(0.0, (dim - 3) * 0.5);

    std::vector<double> c(k);
    for (int i = 0; i < k; ++i)
        c[i] = -0.9 + 1.8 * (i + 0.5) / k;

    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<double> b(k - 1);
        for (int i = 0; i < k - 1; ++i) b[i] = (c[i] + c[i+1]) * 0.5;
        bool converged = true;
        for (int i = 0; i < k; ++i) {
            double lo = (i == 0)     ? -1.0 : b[i-1];
            double hi = (i == k - 1) ?  1.0 : b[i];
            auto [mass, moment] = integrate_beta_region(lo, hi, alpha);
            double nc = (mass > 1e-15) ? moment / mass : (lo + hi) * 0.5;
            if (std::abs(nc - c[i]) > 1e-7) converged = false;
            c[i] = nc;
        }
        if (converged) break;
    }

    Codebook cb; cb.bits = bits;
    cb.centroids.resize(k);
    cb.boundaries.resize(k - 1);
    for (int i = 0; i < k;     ++i) cb.centroids[i]  = static_cast<float>(c[i]);
    for (int i = 0; i < k - 1; ++i) cb.boundaries[i] = static_cast<float>((c[i]+c[i+1])*0.5);
    return cb;
}


// =============================================================================
// §5  Scalar quantization / dequantization
// =============================================================================

static inline int quantize_scalar(float v, const Codebook& cb) {
    int lo = 0, hi = static_cast<int>(cb.boundaries.size());
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        (v > cb.boundaries[mid]) ? lo = mid + 1 : hi = mid;
    }
    return lo;
}

static inline float dequantize_scalar(int idx, const Codebook& cb) {
    return cb.centroids[idx];
}


// =============================================================================
// §6  Bit packing / unpacking
//
//  Pack n indices of `bits` bits each into a compact byte array.
//  No alignment padding — bits stream consecutively.
//
//  Example (3-bit, 4 indices = [5,2,7,0]):
//    binary: 101 010 111 000  ->  bytes: [0b01_010_101, 0b0000_0_111]
// =============================================================================

static void pack_bits(const int* indices, int n, int bits, uint8_t* out) {
    int total_bytes = (n * bits + 7) / 8;
    std::memset(out, 0, total_bytes);
    int bit_pos = 0;
    for (int i = 0; i < n; ++i) {
        int val = indices[i];
        for (int b = 0; b < bits; ++b) {
            if ((val >> b) & 1)
                out[bit_pos >> 3] |= static_cast<uint8_t>(1u << (bit_pos & 7));
            ++bit_pos;
        }
    }
}

static void unpack_bits(const uint8_t* in, int n, int bits, int* out) {
    int bit_pos = 0;
    for (int i = 0; i < n; ++i) {
        int val = 0;
        for (int b = 0; b < bits; ++b) {
            if ((in[bit_pos >> 3] >> (bit_pos & 7)) & 1)
                val |= (1 << b);
            ++bit_pos;
        }
        out[i] = val;
    }
}


// =============================================================================
// §7  Quantized dot product  (core attention speedup)
//
//  Key insight: WHT is orthogonal, so it preserves inner products:
//    <a, b> = <R*a, R*b>  for any orthogonal R.
//
//  After encoding, both vectors are stored in the WHT-rotated domain
//  as centroid indices. We can compute the dot product directly there.
//
//  <a, b> ~= norm_a * norm_b * sum_i centroids[idx_a[i]] * centroids[idx_b[i]]
//
//  This is where the 8x attention speedup comes from: no dequant needed,
//  centroid LUT is tiny (8 floats for 3-bit = 32 bytes, fits in L1),
//  and the inner loop is a pure FMA-able dot product.
//
//  SIMD strategy (AVX2):
//    _mm256_i32gather_ps  — gather 8 float centroids by index in one shot.
//    _mm256_fmadd_ps      — fused multiply-add for the accumulation.
//    Horizontal reduce    — hadd across 8 lanes at the end.
// =============================================================================

#ifdef TQ_HAVE_AVX2

static float dot_avx2(const int* __restrict__ ia,
                      const int* __restrict__ ib,
                      const float* __restrict__ centroids, int d)
{
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= d; i += 8) {
        // Gather: load centroids[ia[i..i+7]] and centroids[ib[i..i+7]]
        // scale=4 means byte offset = index * 4 (float = 4 bytes)
        __m256i va_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ia + i));
        __m256i vb_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ib + i));
        __m256  ca     = _mm256_i32gather_ps(centroids, va_idx, 4);
        __m256  cb_v   = _mm256_i32gather_ps(centroids, vb_idx, 4);
        // FMA: acc += ca * cb
        acc = _mm256_fmadd_ps(ca, cb_v, acc);
    }
    // Horizontal reduce: sum all 8 lanes
    __m128 lo   = _mm256_castps256_ps128(acc);
    __m128 hi   = _mm256_extractf128_ps(acc, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    __m128 sum1 = _mm_add_ss(sum2, _mm_movehl_ps(sum2, sum2));
    float  total = _mm_cvtss_f32(sum1);
    // scalar tail
    for (; i < d; ++i) total += centroids[ia[i]] * centroids[ib[i]];
    return total;
}

#elif defined(TQ_HAVE_SSE4)

static float dot_sse4(const int* ia, const int* ib,
                      const float* centroids, int d)
{
    __m128 acc = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= d; i += 4) {
        // No gather in SSE4 — manual loads (still vectorizes the mul/add)
        __m128 ca = _mm_set_ps(centroids[ia[i+3]], centroids[ia[i+2]],
                               centroids[ia[i+1]], centroids[ia[i+0]]);
        __m128 cb = _mm_set_ps(centroids[ib[i+3]], centroids[ib[i+2]],
                               centroids[ib[i+1]], centroids[ib[i+0]]);
        acc = _mm_add_ps(acc, _mm_mul_ps(ca, cb));
    }
    __m128 shuf = _mm_movehdup_ps(acc);
    __m128 sum2 = _mm_add_ps(acc, shuf);
    __m128 sum1 = _mm_add_ss(sum2, _mm_movehl_ps(sum2, sum2));
    float total = _mm_cvtss_f32(sum1);
    for (; i < d; ++i) total += centroids[ia[i]] * centroids[ib[i]];
    return total;
}

#endif // SIMD dot


// =============================================================================
// §8  Block encode / decode
//
//  Block layout:  [uint16_t norm_f16 (2 bytes)] [packed indices]
//  Total:         2 + ceil(d * bits / 8) bytes
// =============================================================================

static int block_bytes(int d, int bits) { return 2 + (d * bits + 7) / 8; }

static void encode(const float* vec, int d, int bits,
                   const Codebook& cb, uint64_t seed, uint8_t* out)
{
    std::vector<float> x(vec, vec + d);

    float norm = 0.0f;
    for (float v : x) norm += v * v;
    norm = std::sqrt(norm);

    uint16_t norm_f16 = f32_to_f16(norm > 0 ? norm : 1.0f);
    std::memcpy(out, &norm_f16, 2);  out += 2;

    float inv = (norm > 1e-12f) ? 1.0f / norm : 1.0f;
    for (float& v : x) v *= inv;

    apply_signs(x.data(), d, seed);
    wht(x.data(), d);

    std::vector<int> idx(d);
    for (int i = 0; i < d; ++i) idx[i] = quantize_scalar(x[i], cb);

    pack_bits(idx.data(), d, bits, out);
}

static void decode(const uint8_t* in, int d, int bits,
                   const Codebook& cb, uint64_t seed, float* out)
{
    uint16_t norm_f16;
    std::memcpy(&norm_f16, in, 2);
    float norm = f16_to_f32(norm_f16);
    in += 2;

    std::vector<int>   idx(d);
    std::vector<float> x(d);
    unpack_bits(in, d, bits, idx.data());
    for (int i = 0; i < d; ++i) x[i] = cb.centroids[idx[i]];

    iwht(x.data(), d);
    apply_signs(x.data(), d, seed);
    for (int i = 0; i < d; ++i) out[i] = x[i] * norm;
}

static float quantized_dot(const uint8_t* a, const uint8_t* b,
                            int d, int bits, const Codebook& cb)
{
    uint16_t na16, nb16;
    std::memcpy(&na16, a, 2); std::memcpy(&nb16, b, 2);
    float norm_a = f16_to_f32(na16), norm_b = f16_to_f32(nb16);
    a += 2; b += 2;

    std::vector<int> ia(d), ib(d);
    unpack_bits(a, d, bits, ia.data());
    unpack_bits(b, d, bits, ib.data());

    float dot;
#if defined(TQ_HAVE_AVX2)
    dot = dot_avx2(ia.data(), ib.data(), cb.centroids.data(), d);
#elif defined(TQ_HAVE_SSE4)
    dot = dot_sse4(ia.data(), ib.data(), cb.centroids.data(), d);
#else
    dot = 0.0f;
    for (int i = 0; i < d; ++i) dot += cb.centroids[ia[i]] * cb.centroids[ib[i]];
#endif
    return norm_a * norm_b * dot;
}


// =============================================================================
// §9  TurboQuantizer  —  public API
// =============================================================================

struct TurboQuantizer {
    int      dim;
    int      bits;
    Codebook cb;
    uint64_t seed;

    /**
     * @param dim   Head dimension — power-of-2, typically 64 or 128.
     * @param bits  Bit width per coordinate.
     *              3 -> ~5x compression, marginal quality loss on models >=8B
     *              4 -> ~4x compression, quality indistinguishable from FP16
     * @param seed  Rotation seed — arbitrary but must match encode/decode.
     */
    TurboQuantizer(int dim, int bits, uint64_t seed = 0xCAFEBABEDEADBEEFULL)
        : dim(dim), bits(bits), seed(seed)
    {
        if (dim < 8 || (dim & (dim-1)))
            throw std::invalid_argument("dim must be a power of 2, >= 8");
        if (bits < 1 || bits > 8)
            throw std::invalid_argument("bits must be in [1, 8]");
        cb = compute_lloyd_max(dim, bits);
    }

    int block_size() const { return block_bytes(dim, bits); }

    void encode(const float* vec, uint8_t* out) const {
        turboquant::encode(vec, dim, bits, cb, seed, out);
    }

    void decode(const uint8_t* in, float* out) const {
        turboquant::decode(in, dim, bits, cb, seed, out);
    }

    /**
     * Approximate inner product of two encoded blocks.
     * Equivalent to dot(decode(a), decode(b)) but without dequantizing.
     */
    float dot(const uint8_t* a, const uint8_t* b) const {
        return quantized_dot(a, b, dim, bits, cb);
    }

    void encode_cache(const float* vecs, int seq_len, uint8_t* out) const {
        int bs = block_size();
        for (int t = 0; t < seq_len; ++t)
            encode(vecs + t * dim, out + t * bs);
    }

    /**
     * Compute attention logits scores[t] = <query, K[t]> without dequantizing K.
     * This is the hot path for attention — everything stays quantized.
     */
    void attention_logits(const float* query, const uint8_t* keys_enc,
                          int seq_len, float* scores) const {
        int bs = block_size();
        std::vector<uint8_t> qenc(bs);
        encode(query, qenc.data());
        for (int t = 0; t < seq_len; ++t)
            scores[t] = dot(qenc.data(), keys_enc + t * bs);
    }

    static const char* simd_tier() {
#if   defined(TQ_HAVE_AVX2)
        return "AVX2+FMA";
#elif defined(TQ_HAVE_SSE4)
        return "SSE4.1";
#else
        return "scalar";
#endif
    }
};

} // namespace turboquant