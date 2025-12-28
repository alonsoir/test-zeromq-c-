#pragma once

#include <string>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cstring>
#include "crypto.hpp"
#include "compression.hpp"

namespace crypto {

/**
 * @brief High-level manager combining encryption and compression
 *
 * Provides a simple interface for the encrypt-compress / decompress-decrypt pipeline
 * Used by ml-detector, firewall-acl-agent, and sniffer components.
 */
class CryptoManager {
public:
    /**
     * @brief Construct with encryption seed (32 bytes)
     * @param seed Encryption seed as string (32 bytes)
     */
    explicit CryptoManager(const std::string& seed) {
        // Convert seed string to vector<uint8_t>
        key_ = std::vector<uint8_t>(seed.begin(), seed.end());

        // Validate key size (should be 32 bytes for ChaCha20-Poly1305)
        if (key_.size() != 32) {
            throw std::runtime_error("Invalid encryption key size: expected 32 bytes, got " +
                                   std::to_string(key_.size()));
        }
    }

    /**
     * @brief Encrypt data
     * @param plaintext Data to encrypt (string)
     * @return Encrypted data (string with binary data)
     */
    std::string encrypt(const std::string& plaintext) {
        // Convert string to vector<uint8_t>
        std::vector<uint8_t> data(plaintext.begin(), plaintext.end());

        // Encrypt using crypto_transport
        auto encrypted = crypto_transport::encrypt(data, key_);

        // Return as string (contains binary data)
        return std::string(encrypted.begin(), encrypted.end());
    }

    /**
     * @brief Decrypt data
     * @param ciphertext Encrypted data (string with binary data)
     * @return Decrypted plaintext (string)
     */
    std::string decrypt(const std::string& ciphertext) {
        // Convert string to vector<uint8_t>
        std::vector<uint8_t> encrypted_data(ciphertext.begin(), ciphertext.end());

        // Decrypt using crypto_transport
        auto decrypted = crypto_transport::decrypt(encrypted_data, key_);

        // Convert vector<uint8_t> back to string
        return std::string(decrypted.begin(), decrypted.end());
    }

    /**
     * @brief Compress data using LZ4
     * @param data Data to compress (string)
     * @return Compressed data (string with binary data)
     */
    std::string compress(const std::string& data) {
        // Convert string to vector<uint8_t>
        std::vector<uint8_t> input(data.begin(), data.end());

        // Compress using crypto_transport
        auto compressed = crypto_transport::compress(input);

        // Return as string (contains binary data)
        return std::string(compressed.begin(), compressed.end());
    }

    /**
     * @brief Decompress LZ4 data
     * @param compressed Compressed data (string with binary data)
     * @return Decompressed data (string)
     */
    std::string decompress(const std::string& compressed) {
        // Convert string to vector<uint8_t>
        std::vector<uint8_t> compressed_data(compressed.begin(), compressed.end());

        // Decompress using crypto_transport
        // Estimate max decompressed size (10x compression ratio)
        size_t max_size = compressed_data.size() * 10;
        if (max_size < 1024) max_size = 1024; // Minimum 1KB

        auto decompressed = crypto_transport::decompress(compressed_data, max_size);

        // Convert vector<uint8_t> back to string
        return std::string(decompressed.begin(), decompressed.end());
    }

    /**
     * @brief Get current encryption key
     * @return Encryption key as string (32 bytes)
     */
    std::string get_encryption_key() const {
        return std::string(key_.begin(), key_.end());
    }

private:
    std::vector<uint8_t> key_;  // 32-byte encryption key
};

} // namespace crypto