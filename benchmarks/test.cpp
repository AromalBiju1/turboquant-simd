#include "turboquant.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <iomanip>

// Helper: Calculate Cosine Similarity
float calc_cosine(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0, norm_a = 0, norm_b = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-12f);
}

// Helper: Load binary file
std::vector<float> load_real_cache(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Could not open file. Did you run the Python script?");
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<float> buffer(size / sizeof(float));
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    }
    throw std::runtime_error("Failed to read binary data.");
}

int main() {
    std::cout << "==== TurboQuant Real LLM Test ====\n";

    // 1. Load the real Qwen weights
    std::vector<float> qwen_cache;
    try {
        qwen_cache = load_real_cache("qwen_cache.bin");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    int d = 128; // Qwen 1.5B head dimension
    int seq_len = qwen_cache.size() / d;
    std::cout << "Loaded Qwen KV Cache: " << seq_len << " tokens\n\n";

    turboquant::TurboQuantizer tq(d, 4); // 4-bit compression
    
    std::vector<uint8_t> compressed_cache(seq_len * tq.block_size());
    std::vector<float> decoded_cache(qwen_cache.size());

    // 2. Compress the entire sequence
    for (int t = 0; t < seq_len; ++t) {
        tq.encode(qwen_cache.data() + t * d, compressed_cache.data() + t * tq.block_size());
    }

    // 3. Decompress the entire sequence to measure accuracy
    for (int t = 0; t < seq_len; ++t) {
        tq.decode(compressed_cache.data() + t * tq.block_size(), decoded_cache.data() + t * d);
    }

    // 4. Calculate total Cosine Similarity across the whole sequence
    float total_cosine = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
        std::vector<float> orig_token(qwen_cache.begin() + t * d, qwen_cache.begin() + (t + 1) * d);
        std::vector<float> dec_token(decoded_cache.begin() + t * d, decoded_cache.begin() + (t + 1) * d);
        total_cosine += calc_cosine(orig_token, dec_token);
    }
    float avg_cosine = total_cosine / seq_len;

    // 5. Output Results
    std::cout << "--- Real LLM Results ---\n";
    std::cout << "Original Size   : " << (qwen_cache.size() * 4) / 1024.0f << " KB\n";
    std::cout << "Compressed Size : " << compressed_cache.size() / 1024.0f << " KB\n";
    std::cout << "Compression     : " << (qwen_cache.size() * 4.0f) / compressed_cache.size() << "x\n";
    
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "\nAverage Token Fidelity (Cosine): " << avg_cosine << "\n";

    return 0;
}