// /vagrant/etcd-client/include/etcd_client/etcd_client.hpp
// Day 54: etcd Client with HMAC Grace Period Validation
//
// Co-authored-by: Claude (Anthropic)
// Co-authored-by: Alonso

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "spdlog/spdlog.h"

namespace etcd_client {

/**
 * @brief Etcd client with HMAC utilities and grace period support
 *
 * Features:
 * - HMAC-SHA256 computation and validation
 * - Grace period key fallback validation
 * - Hex encoding/decoding
 *
 * Co-authors: Claude (Anthropic), Alonso
 */
class EtcdClient {
public:
    /**
     * @brief Constructor
     *
     * @param etcd_endpoint Etcd server endpoint (e.g., "localhost:2380")
     */
    explicit EtcdClient(const std::string& etcd_endpoint);

    // ========================================================================
    // HMAC Methods
    // ========================================================================

    /**
     * @brief Compute HMAC-SHA256
     *
     * @param data Input data
     * @param key HMAC key (hex-encoded)
     * @return std::string HMAC digest (hex-encoded)
     */
    std::string compute_hmac_sha256(
        const std::string& data,
        const std::string& key
    );

    /**
     * @brief Validate HMAC-SHA256 (single key)
     *
     * @param data Input data
     * @param hmac_hex Expected HMAC (hex-encoded)
     * @param key HMAC key (hex-encoded)
     * @return bool True if HMAC matches
     */
    bool validate_hmac_sha256(
        const std::string& data,
        const std::string& hmac_hex,
        const std::string& key
    );

    /**
     * @brief Validate HMAC with grace period fallback (Day 54)
     *
     * Algorithm:
     * 1. Try validation with active key first (fastest path)
     * 2. If fails, try each grace period key
     * 3. Log warning if grace key succeeds (operational insight)
     *
     * @param data Input data
     * @param hmac_hex Expected HMAC (hex-encoded)
     * @param valid_keys Vector of valid keys (active + grace period)
     * @return bool True if HMAC validates with ANY valid key
     */
    bool validate_hmac_sha256_with_grace(
        const std::string& data,
        const std::string& hmac_hex,
        const std::vector<std::string>& valid_keys
    );

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /**
     * @brief Convert bytes to hex string
     *
     * @param bytes Input bytes
     * @return std::string Hex-encoded string (lowercase)
     */
    static std::string bytes_to_hex(const std::vector<uint8_t>& bytes);

    /**
     * @brief Convert hex string to bytes
     *
     * @param hex Hex-encoded string
     * @return std::vector<uint8_t> Decoded bytes
     */
    static std::vector<uint8_t> hex_to_bytes(const std::string& hex);

private:
    std::string etcd_endpoint_;
    std::shared_ptr<spdlog::logger> logger_;
};

} // namespace etcd_client