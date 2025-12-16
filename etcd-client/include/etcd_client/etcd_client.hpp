#pragma once

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <cstdint>

// Forward declaration for JSON to avoid including heavy header
namespace nlohmann {
    class json;
}

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
    
    // Rollback active to master
    bool rollback_config();
    
    // ========================================================================
    // Encryption Key Management (handled automatically by library)
    // ========================================================================
    
    // Set encryption key manually (normally received from etcd-server)
    void set_encryption_key(const std::string& key);
    
    // Check if encryption key is available
    bool has_encryption_key() const;

private:
    // PIMPL idiom - implementation hidden
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace etcd_client
