#pragma once

#include <vector>
#include <string>
#include <mutex>
#include <atomic>

namespace sniffer {

/**
 * @brief ZMQ Pool Manager for high-performance message sending
 *
 * Manages a pool of ZeroMQ sockets for load balancing and better throughput
 * when sending large volumes of network security events.
 */
class ZMQPoolManager {
public:
    /**
     * @brief Construct ZMQ pool manager
     * @param pool_size Number of sockets in the pool
     */
    explicit ZMQPoolManager(int pool_size = 4);

    /**
     * @brief Destructor - cleanup resources
     */
    ~ZMQPoolManager();

    // Non-copyable
    ZMQPoolManager(const ZMQPoolManager&) = delete;
    ZMQPoolManager& operator=(const ZMQPoolManager&) = delete;

    /**
     * @brief Connect all sockets to endpoint
     * @param endpoint ZMQ endpoint (e.g., "tcp://192.168.56.20:5571")
     * @return true if all sockets connected successfully
     */
    bool connect(const std::string& endpoint);

    /**
     * @brief Get next available socket from pool (round-robin)
     * @return Socket pointer or nullptr if not connected
     */
    void* get_socket();

    /**
     * @brief Send message using pool (non-blocking)
     * @param data Message data
     * @param size Message size
     * @return true if message sent successfully
     */
    bool send_message(const void* data, size_t size);

    /**
     * @brief Shutdown pool and cleanup
     */
    void shutdown();

    /**
     * @brief Get pool size
     * @return Number of sockets in pool
     */
    size_t get_pool_size() const;

    /**
     * @brief Check if pool is connected
     * @return true if connected
     */
    bool is_connected() const;

private:
    void* context_;                    ///< ZMQ context
    std::vector<void*> sockets_;       ///< Socket pool
    std::string endpoint_;             ///< Connection endpoint
    int pool_size_;                    ///< Pool size
    std::atomic<size_t> current_socket_; ///< Current socket index (round-robin)
    std::mutex socket_mutex_;          ///< Mutex for socket access
    std::atomic<bool> connected_{false}; ///< Connection status
};

} // namespace sniffer