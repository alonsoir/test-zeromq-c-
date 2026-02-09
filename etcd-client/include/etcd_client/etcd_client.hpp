#pragma once

#include <string>
#include <memory>
#include <vector>
#include <optional>
#include <functional>
#include <cstdint>
#include <nlohmann/json.hpp>  // ← Include completo, no forward declaration

namespace etcd_client {

// Component information structure
struct ComponentInfo {
    std::string name;
    std::string status;           // "active", "inactive", "unhealthy"
    std::string config_path;
    uint64_t last_heartbeat_ms;
    std::string metadata_json;    // Additional metadata as JSON string
};

// Configuration structure (loaded from JSON)
struct Config {
    // Server settings
    std::string host = "127.0.0.1";
    int port = 2379;
    int timeout_seconds = 5;

    // Component identity
    std::string component_name;
    std::string component_config_path;

    // Encryption settings
    bool encryption_enabled = true;
    std::string encryption_algorithm = "chacha20-poly1305";

    // Compression settings
    bool compression_enabled = true;
    std::string compression_algorithm = "lz4";
    int compression_min_size = 256;

    // Heartbeat settings
    bool heartbeat_enabled = true;
    int heartbeat_interval_seconds = 30;

    // Retry settings
    int max_retry_attempts = 3;
    int retry_backoff_seconds = 2;

    // Load from JSON file
    static Config from_json_file(const std::string& json_path);

    // Load from JSON object
    static Config from_json(const nlohmann::json& j);
};

// Main EtcdClient class
class EtcdClient {
public:
    // Constructor with config object
    explicit EtcdClient(const Config& config);

    // Constructor with JSON file path
    explicit EtcdClient(const std::string& config_json_path);

    // Destructor
    ~EtcdClient();

    // Disable copy (avoid accidental copies)
    EtcdClient(const EtcdClient&) = delete;
    EtcdClient& operator=(const EtcdClient&) = delete;

    // Enable move (for efficiency)
    EtcdClient(EtcdClient&&) noexcept;
    EtcdClient& operator=(EtcdClient&&) noexcept;

    // ========================================================================
    // Connection Management
    // ========================================================================

    bool connect();
    bool is_connected() const;
    void disconnect();

    // ========================================================================
    // Key-Value Operations
    // (encryption/compression applied automatically based on config)
    // ========================================================================

    bool set(const std::string& key, const std::string& value);
    std::string get(const std::string& key);
    bool del(const std::string& key);
    std::vector<std::string> list_keys(const std::string& prefix = "");

    // ========================================================================
    // Component Discovery & Registration
    // ========================================================================

    // Register this component with etcd-server
    bool register_component();

    // Unregister this component from etcd-server
    bool unregister_component();

    // Send heartbeat (called automatically if heartbeat_enabled)
    bool heartbeat();

    // Get info about a component
    ComponentInfo get_component_info(const std::string& name);

    // List all registered components
    std::vector<ComponentInfo> list_components();

    // ========================================================================
    // Config Management (Master + Active copies for rollback)
    // ========================================================================

    // Save master config (immutable, original)
    bool save_config_master(const std::string& config_json);

    // Save active config (mutable, working copy)
    bool save_config_active(const std::string& config_json);

    // Get master config
    std::string get_config_master();

    // Get active config
    std::string get_config_active();

    std::string get_component_config(const std::string& component_name);

    /**
     * Upload configuration to etcd-server
     * Compresses and encrypts the JSON before sending
     * @param json_config JSON configuration string
     * @return true if successful, false otherwise
     */
    bool put_config(const std::string& json_config);

    // Rollback active to master
    bool rollback_config();

    // ========================================================================
    // Encryption Key Management (handled automatically by library)
    // ========================================================================

    // Set encryption key manually (normally received from etcd-server)
    void set_encryption_key(const std::string& key);

    // Check if encryption key is available
    bool has_encryption_key() const;

    // Get current encryption key (hex format)
    std::string get_encryption_key() const;

    // ========================================================================
    // HMAC Utilities (Day 53 - Log Integrity)
    // ========================================================================

    /**
     * @brief Get HMAC key from etcd-server
     *
     * Retrieves an HMAC key from etcd-server's SecretsManager.
     * The key is returned as raw bytes for immediate use.
     *
     * @param key_path Path to key in etcd (e.g., "/secrets/rag/log_hmac_key")
     * @return Key bytes if successful, nullopt if key not found or error
     *
     * Example:
     *   auto key = client.get_hmac_key("/secrets/rag/log_hmac_key");
     *   if (key.has_value()) {
     *       // Use key for HMAC operations
     *   }
     */
    std::optional<std::vector<uint8_t>> get_hmac_key(const std::string& key_path);

    /**
     * @brief Compute HMAC-SHA256 over data
     *
     * Computes HMAC-SHA256 signature over the provided data using the given key.
     * The result is returned as a hex-encoded string for easy storage/transmission.
     *
     * @param data Data to sign (log line, message, etc.)
     * @param key HMAC key (32 bytes recommended for HMAC-SHA256)
     * @return HMAC as hex string (64 characters for SHA256)
     *
     * Example:
     *   std::string log_line = "2026-02-09 event data";
     *   std::string hmac = client.compute_hmac_sha256(log_line, key);
     *   // hmac = "a3f5c2d8e9b14f7c..." (64 hex chars)
     */
    std::string compute_hmac_sha256(const std::string& data,
                                    const std::vector<uint8_t>& key);

    /**
     * @brief Validate HMAC-SHA256 signature
     *
     * Validates that the provided HMAC matches the data using constant-time
     * comparison to prevent timing attacks.
     *
     * @param data Original data
     * @param hmac_hex Expected HMAC (hex-encoded string)
     * @param key HMAC key used to generate the signature
     * @return true if HMAC is valid, false if invalid or mismatch
     *
     * Example:
     *   if (client.validate_hmac_sha256(data, hmac, key)) {
     *       // HMAC valid - data is authentic
     *   } else {
     *       // HMAC invalid - data may be tampered
     *   }
     */
    bool validate_hmac_sha256(const std::string& data,
                             const std::string& hmac_hex,
                             const std::vector<uint8_t>& key);

    /**
     * @brief Convert bytes to hex string (utility)
     *
     * Converts raw bytes to hex-encoded string for storage/transmission.
     *
     * @param data Raw bytes
     * @return Hex-encoded string (lowercase)
     */
    static std::string bytes_to_hex(const std::vector<uint8_t>& data);

    /**
     * @brief Convert hex string to bytes (utility)
     *
     * Converts hex-encoded string back to raw bytes.
     *
     * @param hex_string Hex-encoded string
     * @return Raw bytes
     * @throws std::invalid_argument if hex string is invalid
     */
    static std::vector<uint8_t> hex_to_bytes(const std::string& hex_string);

private:
    // PIMPL idiom - implementation hidden
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace etcd_client