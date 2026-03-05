// /vagrant/etcd-client/src/etcd_client.cpp
// Day 54: etcd Client Implementation
//
// Co-authored-by: Claude (Anthropic)
// Co-authored-by: Alonso

#include "etcd_client/etcd_client.hpp"
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace etcd_client {

EtcdClient::EtcdClient(const std::string& etcd_endpoint)
    : etcd_endpoint_(etcd_endpoint),
      logger_(spdlog::get("etcd_client") ? spdlog::get("etcd_client") : spdlog::default_logger())
{
    logger_->info("EtcdClient initialized with endpoint: {}", etcd_endpoint_);
}

// ============================================================================
// HMAC Methods
// ============================================================================

std::string EtcdClient::compute_hmac_sha256(
    const std::string& data,
    const std::string& key
) {
    // Decode hex key
    auto key_bytes = hex_to_bytes(key);

    // Compute HMAC
    unsigned char hmac_result[EVP_MAX_MD_SIZE];
    unsigned int hmac_len;

    HMAC(EVP_sha256(),
         key_bytes.data(), key_bytes.size(),
         reinterpret_cast<const unsigned char*>(data.data()), data.size(),
         hmac_result, &hmac_len);

    // Convert to hex
    std::vector<uint8_t> hmac_vec(hmac_result, hmac_result + hmac_len);
    return bytes_to_hex(hmac_vec);
}

bool EtcdClient::validate_hmac_sha256(
    const std::string& data,
    const std::string& hmac_hex,
    const std::string& key
) {
    std::string computed = compute_hmac_sha256(data, key);

    // Constant-time comparison
    bool match = (computed == hmac_hex);

    if (!match) {
        logger_->debug("HMAC validation failed with this key");
    }

    return match;
}

bool EtcdClient::validate_hmac_sha256_with_grace(
    const std::string& data,
    const std::string& hmac_hex,
    const std::vector<std::string>& valid_keys
) {
    if (valid_keys.empty()) {
        logger_->error("No valid HMAC keys provided for validation");
        return false;
    }

    logger_->debug("Validating HMAC with {} valid keys (active + grace)", valid_keys.size());

    // Try active key first (index 0 - should be sorted by caller)
    if (validate_hmac_sha256(data, hmac_hex, valid_keys[0])) {
        logger_->debug("HMAC validated with active key");
        return true;
    }

    // Fallback: try grace period keys
    for (size_t i = 1; i < valid_keys.size(); ++i) {
        if (validate_hmac_sha256(data, hmac_hex, valid_keys[i])) {
            logger_->warn("HMAC validated with grace period key (index {})", i);
            logger_->warn("Component should sync to latest key");
            return true;
        }
    }

    logger_->error("HMAC validation failed with all {} valid keys", valid_keys.size());
    return false;
}

// ============================================================================
// Utility Methods
// ============================================================================

std::string EtcdClient::bytes_to_hex(const std::vector<uint8_t>& bytes) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');

    for (uint8_t byte : bytes) {
        oss << std::setw(2) << static_cast<int>(byte);
    }

    return oss.str();
}

std::vector<uint8_t> EtcdClient::hex_to_bytes(const std::string& hex) {
    if (hex.length() % 2 != 0) {
        throw std::invalid_argument("Hex string must have even length");
    }

    std::vector<uint8_t> bytes;
    bytes.reserve(hex.length() / 2);

    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byte_str = hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoi(byte_str, nullptr, 16));
        bytes.push_back(byte);
    }

    return bytes;
}

} // namespace etcd_client