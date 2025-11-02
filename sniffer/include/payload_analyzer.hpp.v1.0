#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>

namespace sniffer {

/**
 * @brief Thread-safe payload analyzer for malware detection
 * 
 * Analyzes packet payloads for:
 * - PE executable headers (Windows malware)
 * - Shannon entropy (encrypted/packed content)
 * - Suspicious patterns (ransom notes, crypto APIs)
 * 
 * Design:
 * - Thread-local state (no locks needed)
 * - Performance target: <10μs per 512-byte analysis
 * - Compatible with Layer 1 FastDetector architecture
 */
class PayloadAnalyzer {
public:
    /**
     * @brief Analysis results for a single payload
     */
    struct Features {
        // PE Executable Detection
        bool is_pe_executable{false};      ///< MZ/PE header detected
        uint16_t pe_machine_type{0};       ///< CPU architecture (x86, x64, ARM)
        uint32_t pe_timestamp{0};          ///< PE compilation timestamp
        
        // Entropy Analysis
        float entropy{0.0f};               ///< Shannon entropy (0.0-8.0 bits)
        bool high_entropy{false};          ///< >7.0 = likely encrypted/packed
        
        // Pattern Matching
        bool ransom_note_pattern{false};   ///< Ransom message detected
        bool crypto_api_pattern{false};    ///< Crypto API strings found
        uint16_t suspicious_strings{0};    ///< Count of suspicious patterns
        
        // Metadata
        uint16_t analyzed_bytes{0};        ///< Actual bytes analyzed
    };
    
    PayloadAnalyzer() = default;
    ~PayloadAnalyzer() = default;
    
    // Non-copyable (thread_local instance)
    PayloadAnalyzer(const PayloadAnalyzer&) = delete;
    PayloadAnalyzer& operator=(const PayloadAnalyzer&) = delete;
    
    /**
     * @brief Analyze a packet payload
     * 
     * @param payload Raw packet data (can be nullptr if len=0)
     * @param len Payload length (max 512 bytes typically)
     * @return Features struct with analysis results
     * 
     * Thread-safe: Uses only thread_local state
     * Performance: <10μs for 512 bytes
     */
    Features analyze(const uint8_t* payload, uint16_t len);
    
    /**
     * @brief Reset internal state (optional, for testing)
     */
    void reset();
    
private:
    // ===== PE Header Detection =====
    
    /**
     * @brief Check for PE executable signature
     * 
     * Validates:
     * - "MZ" magic (DOS header)
     * - PE signature location
     * - "PE\0\0" magic
     * - COFF header structure
     * 
     * @return true if valid PE file detected
     */
    bool check_pe_header(const uint8_t* data, uint16_t len);
    
    /**
     * @brief Extract PE machine type from COFF header
     * 
     * Common values:
     * - 0x014c: Intel 386 (x86)
     * - 0x8664: x64 (AMD64)
     * - 0x01c0: ARM
     * - 0xaa64: ARM64
     * 
     * @return Machine type or 0 if not PE
     */
    uint16_t extract_pe_machine_type(const uint8_t* data, uint16_t len);
    
    /**
     * @brief Extract PE compilation timestamp
     * @return Unix timestamp or 0 if not PE
     */
    uint32_t extract_pe_timestamp(const uint8_t* data, uint16_t len);
    
    // ===== Entropy Analysis =====
    
    /**
     * @brief Fast Shannon entropy calculation
     * 
     * Uses lookup table optimization for log2.
     * Formula: H = -Σ(p_i * log2(p_i))
     * 
     * Interpretation:
     * - 0.0-4.0: Plain text / low entropy
     * - 4.0-6.0: Compressed / medium entropy
     * - 6.0-7.0: Well compressed
     * - 7.0-8.0: Encrypted / highly random
     * 
     * @return Entropy in bits (0.0-8.0)
     */
    float calculate_entropy_fast(const uint8_t* data, uint16_t len);
    
    // ===== Pattern Matching =====
    
    /**
     * @brief Search for ransom note patterns
     * 
     * Patterns:
     * - "YOUR FILES HAVE BEEN ENCRYPTED"
     * - "DECRYPT", "RANSOM", "BITCOIN"
     * - ".onion" addresses
     * - Wallet addresses
     * 
     * @return true if ransom-related content found
     */
    bool match_ransom_patterns(const uint8_t* data, uint16_t len);
    
    /**
     * @brief Search for cryptographic API strings
     * 
     * Patterns:
     * - "CryptEncrypt", "CryptDecrypt"
     * - "AES", "RSA", "ChaCha"
     * - OpenSSL function names
     * 
     * @return true if crypto API strings found
     */
    bool match_crypto_api_patterns(const uint8_t* data, uint16_t len);
    
    /**
     * @brief Count all suspicious string patterns
     * @return Total count of suspicious strings
     */
    uint16_t count_suspicious_strings(const uint8_t* data, uint16_t len);
    
    /**
     * @brief Case-insensitive substring search
     * 
     * Uses fast Boyer-Moore-like approach for long patterns.
     * Falls back to simple search for short patterns (<4 bytes).
     * 
     * @param haystack Data to search in
     * @param haystack_len Length of data
     * @param needle Pattern to find
     * @param needle_len Length of pattern
     * @return true if pattern found
     */
    bool find_pattern_nocase(const uint8_t* haystack, uint16_t haystack_len,
                             const char* needle, uint16_t needle_len);
    
    // ===== Thread-local State =====
    
    /// Byte frequency table for entropy calculation (reused across calls)
    std::array<uint32_t, 256> byte_freq_{};
    
    /// Pre-computed log2 lookup table for entropy (lazy init)
    static constexpr std::array<float, 257> log2_table_ = []() constexpr {
        std::array<float, 257> table{};
        table[0] = 0.0f;  // log2(0) = 0 by convention
        for (int i = 1; i <= 256; ++i) {
            // log2(x) = ln(x) / ln(2)
            table[i] = std::log(static_cast<float>(i)) / std::log(2.0f);
        }
        return table;
    }();
};

} // namespace sniffer
