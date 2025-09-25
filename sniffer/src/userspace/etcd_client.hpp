#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>

namespace sniffer {

/**
 * @brief etcd client for configuration and token management
 *
 * Handles communication with etcd for:
 * - Configuration retrieval and updates
 * - Encryption token management
 * - Service discovery and registration
 * - Real-time configuration watching
 */
class EtcdClient {
public:
    struct EtcdResponse {
        bool success = false;
        std::string value;
        std::string error;
        int64_t revision = 0;
        int64_t create_revision = 0;
        int64_t mod_revision = 0;
        int64_t version = 0;
    };

    struct WatchEvent {
        enum Type { PUT, DELETE };
        Type type;
        std::string key;
        std::string value;
        int64_t revision;
    };

    using WatchCallback = std::function<void(const WatchEvent&)>;

public:
    /**
     * @brief Construct etcd client
     * @param endpoint etcd endpoint (e.g., "http://192.168.56.20:2379")
     * @param timeout_ms Request timeout in milliseconds
     */
    explicit EtcdClient(const std::string& endpoint, int timeout_ms = 5000);

    /**
     * @brief Destructor - cleanup resources
     */
    ~EtcdClient();

    // Non-copyable
    EtcdClient(const EtcdClient&) = delete;
    EtcdClient& operator=(const EtcdClient&) = delete;

    /**
     * @brief Test connection to etcd
     * @return true if etcd is reachable
     */
    bool test_connection();

    /**
     * @brief Get value from etcd
     * @param key Key to retrieve
     * @return Response with value or error
     */
    EtcdResponse get(const std::string& key);

    /**
     * @brief Put value to etcd
     * @param key Key to store
     * @param value Value to store
     * @return Response indicating success/failure
     */
    EtcdResponse put(const std::string& key, const std::string& value);

    /**
     * @brief Delete key from etcd
     * @param key Key to delete
     * @return Response indicating success/failure
     */
    EtcdResponse del(const std::string& key);

    /**
     * @brief Get all keys with prefix
     * @param prefix Key prefix to match
     * @return Map of key-value pairs
     */
    std::map<std::string, std::string> get_prefix(const std::string& prefix);

    /**
     * @brief Watch key for changes
     * @param key Key to watch
     * @param callback Function to call on changes
     * @return Watch ID for stopping watch
     */
    int watch_key(const std::string& key, WatchCallback callback);

    /**
     * @brief Watch prefix for changes
     * @param prefix Prefix to watch
     * @param callback Function to call on changes
     * @return Watch ID for stopping watch
     */
    int watch_prefix(const std::string& prefix, WatchCallback callback);

    /**
     * @brief Stop watching key/prefix
     * @param watch_id Watch ID returned from watch_key/watch_prefix
     */
    void stop_watch(int watch_id);

    /**
     * @brief Get encryption token from etcd
     * @param service_name Service requesting token
     * @return Encryption token or empty string if not available
     */
    std::string get_encryption_token(const std::string& service_name);

    /**
     * @brief Register service heartbeat
     * @param service_name Service name
     * @param endpoint Service endpoint
     * @param ttl_seconds TTL for heartbeat
     * @return true if registered successfully
     */
    bool register_service(const std::string& service_name,
                         const std::string& endpoint,
                         int ttl_seconds = 30);

    /**
     * @brief Keep service registration alive
     * @param service_name Service name to keep alive
     * @return true if renewed successfully
     */
    bool renew_service(const std::string& service_name);

    /**
     * @brief Get sniffer configuration from etcd
     * @param config_key Configuration key (default: "/sniffer/config")
     * @return JSON configuration string
     */
    std::string get_sniffer_config(const std::string& config_key = "/sniffer/config");

    /**
     * @brief Update sniffer configuration
     * @param config_json JSON configuration string
     * @param config_key Configuration key (default: "/sniffer/config")
     * @return true if updated successfully
     */
    bool update_sniffer_config(const std::string& config_json,
                              const std::string& config_key = "/sniffer/config");

    /**
     * @brief Get compression settings from etcd
     * @return Compression configuration
     */
    std::map<std::string, std::string> get_compression_config();

    /**
     * @brief Check if client is connected
     * @return true if connected to etcd
     */
    bool is_connected() const;

    /**
     * @brief Get client statistics
     * @return Statistics map
     */
    std::map<std::string, int64_t> get_stats() const;

private:
    // HTTP helpers
    struct HttpResponse {
        long status_code = 0;
        std::string body;
        std::string error;
    };

    HttpResponse http_get(const std::string& path);
    HttpResponse http_post(const std::string& path, const std::string& data);
    HttpResponse http_delete(const std::string& path);

    // JSON helpers
    std::string create_put_json(const std::string& key, const std::string& value);
    std::string create_get_json(const std::string& key);
    std::string create_delete_json(const std::string& key);
    std::string create_range_json(const std::string& prefix);

    EtcdResponse parse_get_response(const std::string& json);
    EtcdResponse parse_put_response(const std::string& json);
    std::map<std::string, std::string> parse_range_response(const std::string& json);

    // Base64 helpers
    std::string base64_encode(const std::string& data);
    std::string base64_decode(const std::string& data);

    // Watch thread management
    void watch_thread_func();
    void process_watch_events();

    // Service registration helpers
    std::string generate_lease_key(const std::string& service_name);
    int64_t create_lease(int ttl_seconds);
    bool put_with_lease(const std::string& key, const std::string& value, int64_t lease_id);
    bool renew_lease(int64_t lease_id);

private:
    std::string endpoint_;              ///< etcd endpoint URL
    int timeout_ms_;                   ///< Request timeout
    void* curl_handle_;                ///< libcurl handle

    // Watch management
    std::atomic<bool> watch_running_{false};
    std::thread watch_thread_;
    std::mutex watches_mutex_;
    std::map<int, std::pair<std::string, WatchCallback>> watches_;
    std::atomic<int> next_watch_id_{1};

    // Service registration
    std::mutex leases_mutex_;
    std::map<std::string, int64_t> service_leases_;

    // Statistics
    mutable std::mutex stats_mutex_;
    std::atomic<int64_t> requests_sent_{0};
    std::atomic<int64_t> requests_failed_{0};
    std::atomic<int64_t> watches_active_{0};
    std::atomic<bool> connected_{false};
};

} // namespace sniffer