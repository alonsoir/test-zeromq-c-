// etcd-server/include/etcd_server/secrets_manager.hpp
#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <chrono>
#include <optional>

namespace etcd {

/**
 * @brief Manages cryptographic secrets (HMAC keys, encryption keys, etc.)
 *
 * SecretsManager handles generation, storage, rotation, and retrieval of
 * cryptographic keys used throughout the ML Defender system.
 *
 * Features:
 * - HMAC-SHA256 key generation
 * - Thread-safe key storage
 * - Key rotation support (future)
 * - Audit logging (future)
 *
 * Via Appia Quality: Designed for security and simplicity
 */
class SecretsManager {
public:
    /**
     * @brief Key metadata for tracking and rotation
     */
    struct KeyMetadata {
        std::string path;                           // Key path (e.g., "/secrets/rag/log_hmac_key")
        std::vector<uint8_t> key_data;              // Raw key bytes
        size_t key_length;                          // Key length in bytes
        std::string algorithm;                      // Algorithm (e.g., "hmac-sha256")
        std::chrono::system_clock::time_point created_at;
        std::chrono::system_clock::time_point last_rotated_at;
        uint32_t rotation_count;                    // Number of times rotated
        bool active;                                // Is this key currently active?
    };

    /**
     * @brief Configuration for secrets management
     */
    struct Config {
        bool enabled = true;                        // Enable secrets management
        size_t default_key_length = 32;             // Default key length (256 bits)
        uint32_t rotation_interval_hours = 168;     // Weekly rotation (7 * 24 hours)
        bool auto_generate_on_startup = true;       // Generate missing keys on startup
        std::string storage_path = "/var/lib/etcd-server/secrets";  // Persistent storage (future)
    };

    /**
     * @brief Construct SecretsManager with configuration
     * @param config Configuration for secrets management
     */
    explicit SecretsManager(const Config& config);

    /**
     * @brief Construct SecretsManager with default configuration
     */
    SecretsManager();

    ~SecretsManager() = default;

    // Prevent copying (singleton-like behavior)
    SecretsManager(const SecretsManager&) = delete;
    SecretsManager& operator=(const SecretsManager&) = delete;

    // Allow moving
    SecretsManager(SecretsManager&&) = default;
    SecretsManager& operator=(SecretsManager&&) = default;

    /**
     * @brief Initialize the secrets manager
     * @return true if initialization successful
     */
    bool initialize();

    /**
     * @brief Generate a new HMAC key
     * @param path Key path (e.g., "/secrets/rag/log_hmac_key")
     * @param key_length Length in bytes (default: 32 for HMAC-SHA256)
     * @return true if key generated successfully
     */
    bool generate_hmac_key(const std::string& path, size_t key_length = 32);

    /**
     * @brief Get a key by path
     * @param path Key path
     * @return Key data if exists, nullopt otherwise
     */
    std::optional<std::vector<uint8_t>> get_key(const std::string& path) const;

    /**
     * @brief Get key metadata
     * @param path Key path
     * @return Key metadata if exists, nullopt otherwise
     */
    std::optional<KeyMetadata> get_key_metadata(const std::string& path) const;

    /**
     * @brief Check if a key exists
     * @param path Key path
     * @return true if key exists
     */
    bool has_key(const std::string& path) const;

    /**
     * @brief List all key paths
     * @return Vector of key paths
     */
    std::vector<std::string> list_keys() const;

    /**
     * @brief Rotate a key (generate new, mark old as inactive)
     * @param path Key path
     * @return true if rotation successful
     */
    bool rotate_key(const std::string& path);

    /**
     * @brief Delete a key
     * @param path Key path
     * @return true if deletion successful
     */
    bool delete_key(const std::string& path);

    /**
     * @brief Get statistics about secrets management
     * @return Map of statistics
     */
    std::map<std::string, uint64_t> get_statistics() const;

    /**
     * @brief Convert key to hex string (for storage/transmission)
     * @param key_data Raw key bytes
     * @return Hex-encoded string
     */
    static std::string key_to_hex(const std::vector<uint8_t>& key_data);

    /**
     * @brief Convert hex string to key bytes
     * @param hex_string Hex-encoded key
     * @return Key bytes
     */
    static std::vector<uint8_t> hex_to_key(const std::string& hex_string);

private:
    Config config_;

    // Thread-safe key storage
    mutable std::mutex keys_mutex_;
    std::map<std::string, KeyMetadata> keys_;

    // Statistics
    mutable std::mutex stats_mutex_;
    struct {
        uint64_t keys_generated = 0;
        uint64_t keys_rotated = 0;
        uint64_t keys_deleted = 0;
        uint64_t get_requests = 0;
        uint64_t failed_requests = 0;
    } mutable stats_;

    /**
     * @brief Generate random bytes using libsodium
     * @param length Number of bytes to generate
     * @return Random bytes
     */
    std::vector<uint8_t> generate_random_bytes(size_t length);

    /**
     * @brief Validate key path format
     * @param path Key path
     * @return true if valid
     */
    bool validate_key_path(const std::string& path) const;

    /**
     * @brief Update statistics (thread-safe)
     * @param stat_name Name of statistic to increment
     */
    void increment_stat(const std::string& stat_name) const;
};

} // namespace etcd