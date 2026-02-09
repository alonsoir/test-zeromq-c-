// etcd-server/src/secrets_manager.cpp
#include "etcd_server/secrets_manager.hpp"
#include <sodium.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace etcd {

SecretsManager::SecretsManager(const Config& config)
    : config_(config) {
}

SecretsManager::SecretsManager()
    : config_(Config{}) {
}

bool SecretsManager::initialize() {
    std::cout << "[SecretsManager] Initializing..." << std::endl;

    // Initialize libsodium for cryptographic random number generation
    if (sodium_init() < 0) {
        std::cerr << "[SecretsManager] ❌ Failed to initialize libsodium" << std::endl;
        return false;
    }

    if (!config_.enabled) {
        std::cout << "[SecretsManager] ⚠️  Secrets management DISABLED" << std::endl;
        return true;
    }

    std::cout << "[SecretsManager] ✅ Initialized successfully" << std::endl;
    std::cout << "[SecretsManager]    Default key length: " << config_.default_key_length << " bytes" << std::endl;
    std::cout << "[SecretsManager]    Rotation interval: " << config_.rotation_interval_hours << " hours" << std::endl;

    // Auto-generate default keys if configured
    if (config_.auto_generate_on_startup) {
        std::cout << "[SecretsManager] Auto-generating default HMAC keys..." << std::endl;

        // Generate HMAC key for RAG logs
        if (generate_hmac_key("/secrets/rag/log_hmac_key")) {
            std::cout << "[SecretsManager] ✅ Generated /secrets/rag/log_hmac_key" << std::endl;
        } else {
            std::cerr << "[SecretsManager] ⚠️  Failed to generate /secrets/rag/log_hmac_key" << std::endl;
        }
    }

    return true;
}

bool SecretsManager::generate_hmac_key(const std::string& path, size_t key_length) {
    if (!validate_key_path(path)) {
        std::cerr << "[SecretsManager] ❌ Invalid key path: " << path << std::endl;
        increment_stat("failed_requests");
        return false;
    }

    // Check if key already exists
    {
        std::lock_guard<std::mutex> lock(keys_mutex_);
        if (keys_.find(path) != keys_.end()) {
            std::cout << "[SecretsManager] ⚠️  Key already exists: " << path
                      << " (use rotate_key to replace)" << std::endl;
            return false;
        }
    }

    // Generate random key bytes
    std::vector<uint8_t> key_data = generate_random_bytes(key_length);
    if (key_data.empty()) {
        std::cerr << "[SecretsManager] ❌ Failed to generate random bytes" << std::endl;
        increment_stat("failed_requests");
        return false;
    }

    // Create metadata
    KeyMetadata metadata;
    metadata.path = path;
    metadata.key_data = std::move(key_data);
    metadata.key_length = key_length;
    metadata.algorithm = "hmac-sha256";
    metadata.created_at = std::chrono::system_clock::now();
    metadata.last_rotated_at = metadata.created_at;
    metadata.rotation_count = 0;
    metadata.active = true;

    // Store key
    {
        std::lock_guard<std::mutex> lock(keys_mutex_);
        keys_[path] = std::move(metadata);
    }

    increment_stat("keys_generated");

    std::cout << "[SecretsManager] ✅ Generated HMAC key: " << path
              << " (" << key_length << " bytes)" << std::endl;

    return true;
}

std::optional<std::vector<uint8_t>> SecretsManager::get_key(const std::string& path) const {
    std::lock_guard<std::mutex> lock(keys_mutex_);

    auto it = keys_.find(path);
    if (it == keys_.end()) {
        increment_stat("failed_requests");
        return std::nullopt;
    }

    if (!it->second.active) {
        std::cerr << "[SecretsManager] ⚠️  Key is inactive: " << path << std::endl;
        increment_stat("failed_requests");
        return std::nullopt;
    }

    increment_stat("get_requests");
    return it->second.key_data;
}

std::optional<SecretsManager::KeyMetadata> SecretsManager::get_key_metadata(const std::string& path) const {
    std::lock_guard<std::mutex> lock(keys_mutex_);

    auto it = keys_.find(path);
    if (it == keys_.end()) {
        return std::nullopt;
    }

    return it->second;
}

bool SecretsManager::has_key(const std::string& path) const {
    std::lock_guard<std::mutex> lock(keys_mutex_);
    return keys_.find(path) != keys_.end();
}

std::vector<std::string> SecretsManager::list_keys() const {
    std::lock_guard<std::mutex> lock(keys_mutex_);

    std::vector<std::string> paths;
    paths.reserve(keys_.size());

    for (const auto& [path, metadata] : keys_) {
        paths.push_back(path);
    }

    return paths;
}

bool SecretsManager::rotate_key(const std::string& path) {
    std::lock_guard<std::mutex> lock(keys_mutex_);

    auto it = keys_.find(path);
    if (it == keys_.end()) {
        std::cerr << "[SecretsManager] ❌ Key not found for rotation: " << path << std::endl;
        increment_stat("failed_requests");
        return false;
    }

    // Generate new key with same length
    size_t key_length = it->second.key_length;
    std::vector<uint8_t> new_key_data = generate_random_bytes(key_length);
    if (new_key_data.empty()) {
        std::cerr << "[SecretsManager] ❌ Failed to generate new key during rotation" << std::endl;
        increment_stat("failed_requests");
        return false;
    }

    // Update metadata
    it->second.key_data = std::move(new_key_data);
    it->second.last_rotated_at = std::chrono::system_clock::now();
    it->second.rotation_count++;

    increment_stat("keys_rotated");

    std::cout << "[SecretsManager] ✅ Rotated key: " << path
              << " (rotation #" << it->second.rotation_count << ")" << std::endl;

    return true;
}

bool SecretsManager::delete_key(const std::string& path) {
    std::lock_guard<std::mutex> lock(keys_mutex_);

    auto it = keys_.find(path);
    if (it == keys_.end()) {
        return false;
    }

    // Zero out key data before deletion (security best practice)
    sodium_memzero(it->second.key_data.data(), it->second.key_data.size());

    keys_.erase(it);
    increment_stat("keys_deleted");

    std::cout << "[SecretsManager] ✅ Deleted key: " << path << std::endl;

    return true;
}

std::map<std::string, uint64_t> SecretsManager::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    return {
        {"keys_generated", stats_.keys_generated},
        {"keys_rotated", stats_.keys_rotated},
        {"keys_deleted", stats_.keys_deleted},
        {"get_requests", stats_.get_requests},
        {"failed_requests", stats_.failed_requests},
        {"total_keys", keys_.size()}
    };
}

std::string SecretsManager::key_to_hex(const std::vector<uint8_t>& key_data) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');

    for (uint8_t byte : key_data) {
        oss << std::setw(2) << static_cast<int>(byte);
    }

    return oss.str();
}

std::vector<uint8_t> SecretsManager::hex_to_key(const std::string& hex_string) {
    if (hex_string.length() % 2 != 0) {
        throw std::invalid_argument("Hex string must have even length");
    }

    std::vector<uint8_t> key_data;
    key_data.reserve(hex_string.length() / 2);

    for (size_t i = 0; i < hex_string.length(); i += 2) {
        std::string byte_str = hex_string.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoi(byte_str, nullptr, 16));
        key_data.push_back(byte);
    }

    return key_data;
}

std::vector<uint8_t> SecretsManager::generate_random_bytes(size_t length) {
    std::vector<uint8_t> buffer(length);
    randombytes_buf(buffer.data(), length);
    return buffer;
}

bool SecretsManager::validate_key_path(const std::string& path) const {
    // Key path must start with /secrets/
    if (path.find("/secrets/") != 0) {
        return false;
    }

    // Must not be just "/secrets/"
    if (path.length() <= 9) {
        return false;
    }

    // No double slashes
    if (path.find("//") != std::string::npos) {
        return false;
    }

    // No trailing slash
    if (path.back() == '/') {
        return false;
    }

    return true;
}

void SecretsManager::increment_stat(const std::string& stat_name) const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    if (stat_name == "keys_generated") {
        stats_.keys_generated++;
    } else if (stat_name == "keys_rotated") {
        stats_.keys_rotated++;
    } else if (stat_name == "keys_deleted") {
        stats_.keys_deleted++;
    } else if (stat_name == "get_requests") {
        stats_.get_requests++;
    } else if (stat_name == "failed_requests") {
        stats_.failed_requests++;
    }
}

} // namespace etcd