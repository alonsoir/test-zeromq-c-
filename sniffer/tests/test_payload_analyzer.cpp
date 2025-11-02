#include "payload_analyzer.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>

using namespace sniffer;

// ===== Test Fixture =====

class PayloadAnalyzerTest : public ::testing::Test {
protected:
    PayloadAnalyzer analyzer_;
    
    // Helper: Create random payload
    std::vector<uint8_t> create_random_payload(size_t len, unsigned int seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> dis(0, 255);
        
        std::vector<uint8_t> payload(len);
        for (size_t i = 0; i < len; ++i) {
            payload[i] = static_cast<uint8_t>(dis(gen));
        }
        return payload;
    }
    
    // Helper: Create low-entropy payload (plain text)
    std::vector<uint8_t> create_text_payload(const std::string& text) {
        return std::vector<uint8_t>(text.begin(), text.end());
    }
    
    // Helper: Create PE header (minimal valid PE)
    std::vector<uint8_t> create_pe_header() {
        std::vector<uint8_t> pe(512, 0);
        
        // DOS Header
        pe[0] = 'M';
        pe[1] = 'Z';
        
        // PE offset at 0x3C (pointing to offset 128)
        uint32_t pe_offset = 128;
        pe[0x3C] = pe_offset & 0xFF;
        pe[0x3D] = (pe_offset >> 8) & 0xFF;
        pe[0x3E] = (pe_offset >> 16) & 0xFF;
        pe[0x3F] = (pe_offset >> 24) & 0xFF;
        
        // PE Signature at offset 128
        pe[128] = 'P';
        pe[129] = 'E';
        pe[130] = 0;
        pe[131] = 0;
        
        // COFF Header (Machine Type: x64 = 0x8664)
        pe[132] = 0x64;
        pe[133] = 0x86;
        
        // Timestamp (example: 0x12345678)
        pe[136] = 0x78;
        pe[137] = 0x56;
        pe[138] = 0x34;
        pe[139] = 0x12;
        
        return pe;
    }
};

// ===== Basic Functionality Tests =====

TEST_F(PayloadAnalyzerTest, NullPayloadReturnsZeroFeatures) {
    auto features = analyzer_.analyze(nullptr, 0);
    
    EXPECT_FALSE(features.is_pe_executable);
    EXPECT_EQ(features.entropy, 0.0f);
    EXPECT_FALSE(features.high_entropy);
    EXPECT_FALSE(features.ransom_note_pattern);
    EXPECT_EQ(features.analyzed_bytes, 0);
}

TEST_F(PayloadAnalyzerTest, EmptyPayloadReturnsZeroFeatures) {
    std::vector<uint8_t> empty;
    auto features = analyzer_.analyze(empty.data(), 0);
    
    EXPECT_FALSE(features.is_pe_executable);
    EXPECT_EQ(features.entropy, 0.0f);
    EXPECT_EQ(features.analyzed_bytes, 0);
}

// ===== PE Header Detection Tests =====

TEST_F(PayloadAnalyzerTest, ValidPEHeaderDetected) {
    auto pe = create_pe_header();
    auto features = analyzer_.analyze(pe.data(), pe.size());
    
    EXPECT_TRUE(features.is_pe_executable) << "Should detect valid PE header";
    EXPECT_EQ(features.pe_machine_type, 0x8664) << "Should extract x64 machine type";
    EXPECT_EQ(features.pe_timestamp, 0x12345678) << "Should extract timestamp";
}

TEST_F(PayloadAnalyzerTest, InvalidMZSignatureNotDetected) {
    auto pe = create_pe_header();
    pe[0] = 'X';  // Corrupt MZ signature
    
    auto features = analyzer_.analyze(pe.data(), pe.size());
    EXPECT_FALSE(features.is_pe_executable);
}

TEST_F(PayloadAnalyzerTest, InvalidPESignatureNotDetected) {
    auto pe = create_pe_header();
    pe[128] = 'X';  // Corrupt PE signature
    
    auto features = analyzer_.analyze(pe.data(), pe.size());
    EXPECT_FALSE(features.is_pe_executable);
}

TEST_F(PayloadAnalyzerTest, TruncatedPEHeaderNotDetected) {
    auto pe = create_pe_header();
    
    // Too short for full DOS header
    auto features = analyzer_.analyze(pe.data(), 32);
    EXPECT_FALSE(features.is_pe_executable);
}

TEST_F(PayloadAnalyzerTest, PEMachineTypesCorrectlyIdentified) {
    auto pe = create_pe_header();
    
    // Test x86 (0x014c)
    pe[132] = 0x4c;
    pe[133] = 0x01;
    auto features_x86 = analyzer_.analyze(pe.data(), pe.size());
    EXPECT_EQ(features_x86.pe_machine_type, 0x014c);
    
    // Test ARM64 (0xaa64)
    pe[132] = 0x64;
    pe[133] = 0xaa;
    auto features_arm64 = analyzer_.analyze(pe.data(), pe.size());
    EXPECT_EQ(features_arm64.pe_machine_type, 0xaa64);
}

// ===== Entropy Analysis Tests =====

TEST_F(PayloadAnalyzerTest, PlainTextHasLowEntropy) {
    auto text = create_text_payload("Hello World! This is a plain text message.");
    auto features = analyzer_.analyze(text.data(), text.size());
    
    EXPECT_LT(features.entropy, 5.0f) << "Plain text should have entropy < 5.0";
    EXPECT_FALSE(features.high_entropy);
}

TEST_F(PayloadAnalyzerTest, RandomDataHasHighEntropy) {
    auto random = create_random_payload(512);
    auto features = analyzer_.analyze(random.data(), random.size());
    
    EXPECT_GT(features.entropy, 7.0f) << "Random data should have entropy > 7.0";
    EXPECT_TRUE(features.high_entropy);
}

TEST_F(PayloadAnalyzerTest, AllSameByteHasZeroEntropy) {
    std::vector<uint8_t> zeros(100, 0);
    auto features = analyzer_.analyze(zeros.data(), zeros.size());
    
    EXPECT_FLOAT_EQ(features.entropy, 0.0f);
    EXPECT_FALSE(features.high_entropy);
}

TEST_F(PayloadAnalyzerTest, BalancedDistributionHasMaxEntropy) {
    // Create payload with exactly 2 occurrences of each byte 0-255
    std::vector<uint8_t> balanced(512);
    for (int i = 0; i < 256; ++i) {
        balanced[i * 2] = i;
        balanced[i * 2 + 1] = i;
    }
    
    auto features = analyzer_.analyze(balanced.data(), balanced.size());
    
    // Should be very close to 8.0 bits (max entropy)
    EXPECT_GT(features.entropy, 7.9f);
    EXPECT_LT(features.entropy, 8.1f);
    EXPECT_TRUE(features.high_entropy);
}

// ===== Pattern Matching Tests =====

TEST_F(PayloadAnalyzerTest, RansomNotePatternDetected) {
    auto text = create_text_payload(
        "YOUR FILES HAVE BEEN ENCRYPTED! "
        "Pay 0.5 BITCOIN to wallet: bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
    );
    
    auto features = analyzer_.analyze(text.data(), text.size());
    
    EXPECT_TRUE(features.ransom_note_pattern);
    EXPECT_GT(features.suspicious_strings, 3) << "Should detect ENCRYPTED, BITCOIN, etc.";
}

TEST_F(PayloadAnalyzerTest, CryptoAPIPatternDetected) {
    auto text = create_text_payload(
        "Calling CryptEncrypt with AES-256 algorithm"
    );
    
    auto features = analyzer_.analyze(text.data(), text.size());
    
    EXPECT_TRUE(features.crypto_api_pattern);
    EXPECT_GT(features.suspicious_strings, 1);
}

TEST_F(PayloadAnalyzerTest, OnionAddressDetected) {
    auto text = create_text_payload(
        "Visit http://3g2upl4pq6kufc4m.onion for instructions"
    );
    
    auto features = analyzer_.analyze(text.data(), text.size());
    
    EXPECT_TRUE(features.ransom_note_pattern);
}

TEST_F(PayloadAnalyzerTest, CaseInsensitivePatternMatching) {
    auto text_upper = create_text_payload("YOUR FILES ARE ENCRYPTED");
    auto text_lower = create_text_payload("your files are encrypted");
    auto text_mixed = create_text_payload("YoUr FiLeS aRe EnCrYpTeD");
    
    EXPECT_TRUE(analyzer_.analyze(text_upper.data(), text_upper.size()).ransom_note_pattern);
    EXPECT_TRUE(analyzer_.analyze(text_lower.data(), text_lower.size()).ransom_note_pattern);
    EXPECT_TRUE(analyzer_.analyze(text_mixed.data(), text_mixed.size()).ransom_note_pattern);
}

TEST_F(PayloadAnalyzerTest, NoFalsePositivesOnNormalData) {
    auto text = create_text_payload(
        "GET /index.html HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "User-Agent: Mozilla/5.0\r\n"
    );
    
    auto features = analyzer_.analyze(text.data(), text.size());
    
    EXPECT_FALSE(features.ransom_note_pattern);
    EXPECT_FALSE(features.crypto_api_pattern);
    EXPECT_EQ(features.suspicious_strings, 0);
}

// ===== Combined Detection Tests =====

TEST_F(PayloadAnalyzerTest, RansomwarePayloadFullDetection) {
    // Simulate ransomware payload: PE + high entropy + ransom note
    auto payload = create_pe_header();
    
    // Add high-entropy section (encrypted data)
    auto random = create_random_payload(256);
    payload.insert(payload.end(), random.begin(), random.end());
    
    // Add ransom note
    std::string note = "DECRYPT YOUR FILES WITH BITCOIN";
    payload.insert(payload.end(), note.begin(), note.end());
    
    auto features = analyzer_.analyze(payload.data(), payload.size());
    
    // Should detect all ransomware indicators
    EXPECT_TRUE(features.is_pe_executable);
    EXPECT_TRUE(features.high_entropy);  // Overall entropy affected by random section
    EXPECT_TRUE(features.ransom_note_pattern);
    EXPECT_GT(features.suspicious_strings, 2);
}

// ===== Performance Tests =====

TEST_F(PayloadAnalyzerTest, PerformanceUnder10Microseconds) {
    auto payload = create_random_payload(512);
    
    // Warm-up
    for (int i = 0; i < 100; ++i) {
        analyzer_.analyze(payload.data(), payload.size());
    }
    
    // Benchmark
    const int iterations = 10000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        auto features = analyzer_.analyze(payload.data(), payload.size());
        // Prevent optimization
        volatile auto dummy = features.entropy;
        (void)dummy;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double avg_us = static_cast<double>(duration_us) / iterations;
    
    std::cout << "[PERF] Average analysis time: " << avg_us << " μs" << std::endl;
    
    EXPECT_LT(avg_us, 10.0) << "Analysis should complete in <10μs";
}

TEST_F(PayloadAnalyzerTest, ThreadLocalStateIsolated) {
    // This test verifies thread_local behavior by running multiple
    // analyses and ensuring state doesn't leak
    
    auto payload1 = create_random_payload(256, 111);
    auto payload2 = create_random_payload(256, 222);
    
    auto f1_first = analyzer_.analyze(payload1.data(), payload1.size());
    auto f2 = analyzer_.analyze(payload2.data(), payload2.size());
    auto f1_second = analyzer_.analyze(payload1.data(), payload1.size());
    
    // Same payload should give same results
    EXPECT_FLOAT_EQ(f1_first.entropy, f1_second.entropy);
    EXPECT_EQ(f1_first.suspicious_strings, f1_second.suspicious_strings);
    
    // Different payloads should give different results
    EXPECT_NE(f1_first.entropy, f2.entropy);
}

// ===== Edge Cases =====

TEST_F(PayloadAnalyzerTest, SingleBytePayload) {
    uint8_t single_byte = 0x42;
    auto features = analyzer_.analyze(&single_byte, 1);
    
    EXPECT_EQ(features.analyzed_bytes, 1);
    EXPECT_FLOAT_EQ(features.entropy, 0.0f);  // Single byte = zero entropy
}

TEST_F(PayloadAnalyzerTest, MaxPayloadSize) {
    auto large = create_random_payload(512);  // Max eBPF payload size
    auto features = analyzer_.analyze(large.data(), large.size());
    
    EXPECT_EQ(features.analyzed_bytes, 512);
    EXPECT_GT(features.entropy, 7.0f);
}

TEST_F(PayloadAnalyzerTest, ResetClearsState) {
    auto payload = create_random_payload(256);
    
    analyzer_.analyze(payload.data(), payload.size());
    analyzer_.reset();
    
    // After reset, internal state should be cleared
    // (This is mainly for testing; reset() is rarely needed in production)
    auto features = analyzer_.analyze(payload.data(), payload.size());
    EXPECT_GT(features.entropy, 0.0f);  // Should still work after reset
}

// ===== Main =====

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
