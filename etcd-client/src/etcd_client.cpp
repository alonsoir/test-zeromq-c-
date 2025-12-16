// etcd-client/src/etcd_client.cpp
#include "etcd_client/etcd_client.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <atomic>

using json = nlohmann::json;

// Forward declarations of helper functions
namespace etcd_client {
namespace http {
    struct Response {
        int status_code = 0;
        std::string body;
        bool success = false;
    };
    
    Response post(const std::string& host, int port, const std::string& path,
                  const std::string& body, int timeout_seconds, int max_retries, int backoff_seconds);
    Response get(const std::string& host, int port, const std::string& path,
                 int timeout_seconds, int max_retries, int backoff_seconds);
    Response del(const std::string& host, int port, const std::string& path,
                 int timeout_seconds, int max_retries, int backoff_seconds);
}

namespace compression {
    std::string compress_lz4(const std::string& data);
    std::string decompress_lz4(const std::string& compressed_data, size_t original_size);
    bool should_compress(size_t data_size, size_t min_size_threshold);
}

namespace crypto {
    std::string encrypt_chacha20(const std::string& plaintext, const std::string& key);
    std::string decrypt_chacha20(const std::string& encrypted_data, const std::string& key);
    std::string generate_key();
    size_t get_key_size();
}

namespace component {
    std::string build_registration_payload(const Config& config);
    std::string build_heartbeat_payload(const Config& config);
    std::string build_unregister_payload(const Config& config);
    ComponentInfo parse_component_info(const std::string& json_str);
    std::vector<ComponentInfo> parse_component_list(const std::string& json_str);
}
}

namespace etcd_client {

// ============================================================================
// EtcdClient::Impl - Private Implementation
// ============================================================================

struct EtcdClient::Impl {
    Config config_;
    std::string encryption_key_;
    bool connected_ = false;
    mutable std::mutex mutex_;
    
    // Heartbeat thread
    std::thread heartbeat_thread_;
    std::atomic<bool> heartbeat_running_{false};
    
    // Constructor
    explicit Impl(const Config& config) 
        : config_(config) {
        std::cout << "🔧 EtcdClient initialized for component: " 
                  << config_.component_name << std::endl;
    }
    
    // Destructor
    ~Impl() {
        stop_heartbeat_thread();
    }
    
    // Start heartbeat thread
    void start_heartbeat_thread() {
        if (!config_.heartbeat_enabled) {
            return;
        }
        
        if (heartbeat_running_.exchange(true)) {
            return; // Already running
        }
        
        heartbeat_thread_ = std::thread([this]() {
            std::cout << "💓 Heartbeat thread started (interval: " 
                      << config_.heartbeat_interval_seconds << "s)" << std::endl;
            
            while (heartbeat_running_) {
                std::this_thread::sleep_for(
                    std::chrono::seconds(config_.heartbeat_interval_seconds)
                );
                
                if (!heartbeat_running_) break;
                
                try {
                    send_heartbeat();
                } catch (const std::exception& e) {
                    std::cerr << "⚠️  Heartbeat failed: " << e.what() << std::endl;
                }
            }
            
            std::cout << "💓 Heartbeat thread stopped" << std::endl;
        });
    }
    
    // Stop heartbeat thread
    void stop_heartbeat_thread() {
        if (heartbeat_running_.exchange(false)) {
            if (heartbeat_thread_.joinable()) {
                heartbeat_thread_.join();
            }
        }
    }
    
    // Send heartbeat
    bool send_heartbeat() {
        std::string payload = component::build_heartbeat_payload(config_);
        
        auto response = http::post(
            config_.host,
            config_.port,
            "/heartbeat",
            payload,
            config_.timeout_seconds,
            1, // Single attempt for heartbeat
            0
        );
        
        return response.success;
    }
    
    // Apply encryption and compression to data
    std::string process_outgoing_data(const std::string& data) {
        std::string processed = data;
        size_t original_size = data.size();
        
        // Step 1: Compress (if enabled and data size > threshold)
        if (config_.compression_enabled && 
            compression::should_compress(original_size, config_.compression_min_size)) {
            try {
                processed = compression::compress_lz4(processed);
                std::cout << "📦 Compressed: " << original_size << " → " 
                          << processed.size() << " bytes" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "⚠️  Compression failed: " << e.what() << std::endl;
                // Continue without compression
                processed = data;
            }
        }
        
        // Step 2: Encrypt (if enabled and key available)
        if (config_.encryption_enabled && !encryption_key_.empty()) {
            try {
                processed = crypto::encrypt_chacha20(processed, encryption_key_);
                std::cout << "🔒 Encrypted: " << processed.size() << " bytes" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "⚠️  Encryption failed: " << e.what() << std::endl;
                throw; // Encryption failure is critical
            }
        }
        
        return processed;
    }
    
    // Remove encryption and decompression from data
    std::string process_incoming_data(const std::string& data, size_t original_size = 0) {
        std::string processed = data;
        
        // Step 1: Decrypt (if enabled and key available)
        if (config_.encryption_enabled && !encryption_key_.empty()) {
            try {
                processed = crypto::decrypt_chacha20(processed, encryption_key_);
                std::cout << "🔓 Decrypted: " << processed.size() << " bytes" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "⚠️  Decryption failed: " << e.what() << std::endl;
                throw; // Decryption failure is critical
            }
        }
        
        // Step 2: Decompress (if compression was used)
        if (config_.compression_enabled && original_size > 0) {
            try {
                processed = compression::decompress_lz4(processed, original_size);
                std::cout << "📦 Decompressed: " << data.size() << " → " 
                          << processed.size() << " bytes" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "⚠️  Decompression failed: " << e.what() << std::endl;
                throw;
            }
        }
        
        return processed;
    }
};

// ============================================================================
// EtcdClient - Public Interface Implementation
// ============================================================================

// Constructors
EtcdClient::EtcdClient(const Config& config)
    : pImpl(std::make_unique<Impl>(config)) {}

EtcdClient::EtcdClient(const std::string& config_json_path)
    : pImpl(std::make_unique<Impl>(Config::from_json_file(config_json_path))) {}

// Destructor
EtcdClient::~EtcdClient() = default;

// Move semantics
EtcdClient::EtcdClient(EtcdClient&&) noexcept = default;
EtcdClient& EtcdClient::operator=(EtcdClient&&) noexcept = default;

// Connection management
bool EtcdClient::connect() {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    
    if (pImpl->connected_) {
        std::cout << "✅ Already connected" << std::endl;
        return true;
    }
    
    std::cout << "🔗 Connecting to etcd-server: " 
              << pImpl->config_.host << ":" << pImpl->config_.port << std::endl;
    
    // Test connection with a simple GET request
    auto response = http::get(
        pImpl->config_.host,
        pImpl->config_.port,
        "/health",
        pImpl->config_.timeout_seconds,
        pImpl->config_.max_retry_attempts,
        pImpl->config_.retry_backoff_seconds
    );
    
    pImpl->connected_ = response.success;
    
    if (pImpl->connected_) {
        std::cout << "✅ Connected to etcd-server" << std::endl;
    } else {
        std::cerr << "❌ Failed to connect to etcd-server" << std::endl;
    }
    
    return pImpl->connected_;
}

bool EtcdClient::is_connected() const {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    return pImpl->connected_;
}

void EtcdClient::disconnect() {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    
    if (!pImpl->connected_) {
        return;
    }
    
    pImpl->stop_heartbeat_thread();
    pImpl->connected_ = false;
    
    std::cout << "🔌 Disconnected from etcd-server" << std::endl;
}

// Key-value operations
bool EtcdClient::set(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    
    if (!pImpl->connected_) {
        std::cerr << "❌ Not connected to etcd-server" << std::endl;
        return false;
    }
    
    try {
        // Process data (compress + encrypt)
        std::string processed_value = pImpl->process_outgoing_data(value);
        
        // Build JSON payload
        json payload = {
            {"key", key},
            {"value", processed_value},
            {"original_size", value.size()}
        };
        
        // Send to etcd-server
        auto response = http::post(
            pImpl->config_.host,
            pImpl->config_.port,
            "/kv/set",
            payload.dump(),
            pImpl->config_.timeout_seconds,
            pImpl->config_.max_retry_attempts,
            pImpl->config_.retry_backoff_seconds
        );
        
        if (response.success) {
            std::cout << "✅ Key set: " << key << std::endl;
            return true;
        }
        
        std::cerr << "❌ Failed to set key: " << key << std::endl;
        return false;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in set(): " << e.what() << std::endl;
        return false;
    }
}

std::string EtcdClient::get(const std::string& key) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    
    if (!pImpl->connected_) {
        std::cerr << "❌ Not connected to etcd-server" << std::endl;
        return "";
    }
    
    try {
        // Request from etcd-server
        auto response = http::get(
            pImpl->config_.host,
            pImpl->config_.port,
            "/kv/get?key=" + key,
            pImpl->config_.timeout_seconds,
            pImpl->config_.max_retry_attempts,
            pImpl->config_.retry_backoff_seconds
        );
        
        if (!response.success) {
            std::cerr << "❌ Failed to get key: " << key << std::endl;
            return "";
        }
        
        // Parse response
        json j = json::parse(response.body);
        std::string value = j.value("value", "");
        size_t original_size = j.value("original_size", 0);
        
        // Process data (decrypt + decompress)
        std::string processed_value = pImpl->process_incoming_data(value, original_size);
        
        std::cout << "✅ Key retrieved: " << key << std::endl;
        return processed_value;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in get(): " << e.what() << std::endl;
        return "";
    }
}

bool EtcdClient::del(const std::string& key) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    
    if (!pImpl->connected_) {
        std::cerr << "❌ Not connected to etcd-server" << std::endl;
        return false;
    }
    
    auto response = http::del(
        pImpl->config_.host,
        pImpl->config_.port,
        "/kv/delete?key=" + key,
        pImpl->config_.timeout_seconds,
        pImpl->config_.max_retry_attempts,
        pImpl->config_.retry_backoff_seconds
    );
    
    if (response.success) {
        std::cout << "✅ Key deleted: " << key << std::endl;
        return true;
    }
    
    std::cerr << "❌ Failed to delete key: " << key << std::endl;
    return false;
}

std::vector<std::string> EtcdClient::list_keys(const std::string& prefix) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    
    std::vector<std::string> keys;
    
    if (!pImpl->connected_) {
        std::cerr << "❌ Not connected to etcd-server" << std::endl;
        return keys;
    }
    
    try {
        std::string path = "/kv/list";
        if (!prefix.empty()) {
            path += "?prefix=" + prefix;
        }
        
        auto response = http::get(
            pImpl->config_.host,
            pImpl->config_.port,
            path,
            pImpl->config_.timeout_seconds,
            pImpl->config_.max_retry_attempts,
            pImpl->config_.retry_backoff_seconds
        );
        
        if (response.success) {
            json j = json::parse(response.body);
            if (j.contains("keys") && j["keys"].is_array()) {
                for (const auto& key : j["keys"]) {
                    keys.push_back(key.get<std::string>());
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in list_keys(): " << e.what() << std::endl;
    }
    
    return keys;
}

// Component discovery
bool EtcdClient::register_component() {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    
    if (!pImpl->connected_) {
        std::cerr << "❌ Not connected, attempting to connect..." << std::endl;
        pImpl->mutex_.unlock(); // Unlock before calling connect()
        bool conn = connect();
        pImpl->mutex_.lock();   // Re-lock after connect()
        
        if (!conn) {
            std::cerr << "❌ Failed to connect for registration" << std::endl;
            return false;
        }
    }
    
    std::string payload = component::build_registration_payload(pImpl->config_);
    
    auto response = http::post(
        pImpl->config_.host,
        pImpl->config_.port,
        "/register",
        payload,
        pImpl->config_.timeout_seconds,
        pImpl->config_.max_retry_attempts,
        pImpl->config_.retry_backoff_seconds
    );
    
    if (response.success) {
        std::cout << "✅ Component registered: " << pImpl->config_.component_name << std::endl;
        
        // Start heartbeat thread
        pImpl->mutex_.unlock();
        pImpl->start_heartbeat_thread();
        pImpl->mutex_.lock();
        
        return true;
    }
    
    std::cerr << "❌ Component registration failed" << std::endl;
    return false;
}

bool EtcdClient::unregister_component() {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    
    // Stop heartbeat first
    pImpl->mutex_.unlock();
    pImpl->stop_heartbeat_thread();
    pImpl->mutex_.lock();
    
    if (!pImpl->connected_) {
        std::cerr << "⚠️  Not connected, skipping unregister" << std::endl;
        return false;
    }
    
    std::string payload = component::build_unregister_payload(pImpl->config_);
    
    auto response = http::post(
        pImpl->config_.host,
        pImpl->config_.port,
        "/unregister",
        payload,
        pImpl->config_.timeout_seconds,
        1, // Single attempt
        0
    );
    
    if (response.success) {
        std::cout << "✅ Component unregistered: " << pImpl->config_.component_name << std::endl;
        return true;
    }
    
    std::cerr << "⚠️  Component unregister failed (non-critical)" << std::endl;
    return false;
}

bool EtcdClient::heartbeat() {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    return pImpl->send_heartbeat();
}

ComponentInfo EtcdClient::get_component_info(const std::string& name) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    
    ComponentInfo info;
    
    if (!pImpl->connected_) {
        std::cerr << "❌ Not connected to etcd-server" << std::endl;
        return info;
    }
    
    auto response = http::get(
        pImpl->config_.host,
        pImpl->config_.port,
        "/component?name=" + name,
        pImpl->config_.timeout_seconds,
        pImpl->config_.max_retry_attempts,
        pImpl->config_.retry_backoff_seconds
    );
    
    if (response.success) {
        return component::parse_component_info(response.body);
    }
    
    return info;
}

std::vector<ComponentInfo> EtcdClient::list_components() {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    
    std::vector<ComponentInfo> components;
    
    if (!pImpl->connected_) {
        std::cerr << "❌ Not connected to etcd-server" << std::endl;
        return components;
    }
    
    auto response = http::get(
        pImpl->config_.host,
        pImpl->config_.port,
        "/components",
        pImpl->config_.timeout_seconds,
        pImpl->config_.max_retry_attempts,
        pImpl->config_.retry_backoff_seconds
    );
    
    if (response.success) {
        return component::parse_component_list(response.body);
    }
    
    return components;
}

// Config management
bool EtcdClient::save_config_master(const std::string& config_json) {
    return set("/config/" + pImpl->config_.component_name + "/master", config_json);
}

bool EtcdClient::save_config_active(const std::string& config_json) {
    return set("/config/" + pImpl->config_.component_name + "/active", config_json);
}

std::string EtcdClient::get_config_master() {
    return get("/config/" + pImpl->config_.component_name + "/master");
}

std::string EtcdClient::get_config_active() {
    return get("/config/" + pImpl->config_.component_name + "/active");
}

bool EtcdClient::rollback_config() {
    std::string master = get_config_master();
    if (master.empty()) {
        std::cerr << "❌ No master config found for rollback" << std::endl;
        return false;
    }
    
    bool result = save_config_active(master);
    if (result) {
        std::cout << "✅ Config rolled back to master" << std::endl;
    }
    
    return result;
}

// Encryption key management
void EtcdClient::set_encryption_key(const std::string& key) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    
    if (key.size() != crypto::get_key_size()) {
        throw std::runtime_error("Invalid key size (expected " + 
                                 std::to_string(crypto::get_key_size()) + " bytes)");
    }
    
    pImpl->encryption_key_ = key;
    std::cout << "🔑 Encryption key set" << std::endl;
}

bool EtcdClient::has_encryption_key() const {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    return !pImpl->encryption_key_.empty();
}

} // namespace etcd_client
