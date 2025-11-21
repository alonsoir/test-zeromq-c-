//===----------------------------------------------------------------------===//
// ML Defender - Firewall ACL Agent
// zmq_subscriber.hpp - ZMQ Subscriber for ML Detection Messages
//
// Purpose:
//   - Receives DetectionBatch messages from ml-detector via ZMQ PUB/SUB
//   - Parses protobuf binary messages
//   - Forwards detections to BatchProcessor for IP blocking
//   - Handles reconnection, error recovery, and metrics
//
// Architecture:
//   ml-detector (ZMQ PUB) → zmq_subscriber (ZMQ SUB) → batch_processor
//
// Design Philosophy:
//   - Simple blocking recv() in dedicated thread (Phase 1)
//   - Automatic reconnection with exponential backoff
//   - Parse errors are logged and skipped (don't crash)
//   - Metrics via atomic counters (zero lock contention)
//
// Thread Safety:
//   - run() intended to be called from single dedicated thread
//   - stop() can be called from any thread (atomic flag)
//   - BatchProcessor handles its own thread safety
//
// Performance Model:
//   Detection Rate     Throughput      Latency
//   100/sec            10K IPs/sec     <100ms
//   1K/sec             100K IPs/sec    <50ms
//   10K/sec            1M IPs/sec      <10ms
//
// Via Appia Quality: Simple subscriber, let batch_processor do the work
//===----------------------------------------------------------------------===//

#ifndef FIREWALL_ZMQ_SUBSCRIBER_HPP
#define FIREWALL_ZMQ_SUBSCRIBER_HPP

#include "firewall/batch_processor.hpp"
#include "network_security.pb.h"
#include <zmq.hpp>
#include <string>
#include <memory>
#include <atomic>
#include <chrono>

namespace mldefender {
namespace firewall {

/**
 * @brief ZMQ Subscriber for receiving ML detection messages
 *
 * This class subscribes to a ZMQ PUB socket (typically from ml-detector),
 * receives DetectionBatch protobuf messages, and forwards them to a
 * BatchProcessor for IP blocking.
 *
 * Thread Model:
 *   - Call run() from a dedicated thread (blocking event loop)
 *   - Call stop() from any thread to trigger graceful shutdown
 *   - get_stats() and is_running() are thread-safe
 *
 * Example Usage:
 * @code
 *   ZMQSubscriber::Config config;
 *   config.endpoint = "tcp://localhost:5555";
 *   config.topic = "";  // Subscribe to all messages
 *
 *   ZMQSubscriber subscriber(batch_processor, config);
 *
 *   std::thread zmq_thread([&] {
 *       subscriber.run();  // Blocking
 *   });
 *
 *   // ... do work ...
 *
 *   subscriber.stop();
 *   zmq_thread.join();
 * @endcode
 */
class ZMQSubscriber {
public:
    //==========================================================================
    // CONFIGURATION
    //==========================================================================

    /**
     * @brief Configuration for ZMQ subscriber
     */
    struct Config {
        std::string endpoint;           ///< ZMQ endpoint (e.g., "tcp://localhost:5555")
        std::string topic;              ///< Subscription topic ("" = all messages)
        int recv_timeout_ms;            ///< ZMQ_RCVTIMEO (0 = infinite)
        int linger_ms;                  ///< ZMQ_LINGER on close
        int reconnect_interval_ms;      ///< Base reconnection interval
        int max_reconnect_interval_ms;  ///< Max exponential backoff interval
        bool enable_reconnect;          ///< Auto-reconnect on connection loss

        /**
         * @brief Default configuration
         */
        Config()
            : endpoint("tcp://localhost:5555")
            , topic("")  // Subscribe to all messages
            , recv_timeout_ms(1000)  // 1 second timeout for graceful shutdown
            , linger_ms(1000)        // Wait 1s for pending sends on close
            , reconnect_interval_ms(1000)      // Start with 1s backoff
            , max_reconnect_interval_ms(30000) // Max 30s backoff
            , enable_reconnect(true)
        {}
    };

    //==========================================================================
    // METRICS
    //==========================================================================

    /**
     * @brief Runtime statistics (thread-safe via atomics)
     */
    struct Stats {
        std::atomic<uint64_t> messages_received{0};
        std::atomic<uint64_t> detections_processed{0};
        std::atomic<uint64_t> parse_errors{0};
        std::atomic<uint64_t> reconnects{0};
        std::atomic<uint64_t> empty_batches{0};
        std::atomic<uint64_t> invalid_detections{0};

        // Timing
        std::atomic<uint64_t> total_processing_time_us{0};
        std::atomic<uint64_t> last_message_timestamp{0};

        // Connection state
        std::atomic<bool> currently_connected{false};
        std::atomic<uint64_t> connection_established_at{0};
        std::atomic<uint64_t> last_reconnect_attempt{0};
    };

    //==========================================================================
    // LIFECYCLE
    //==========================================================================

    /**
     * @brief Constructor
     *
     * @param processor BatchProcessor to send detections to
     * @param config ZMQ configuration
     *
     * @throws zmq::error_t if ZMQ context creation fails
     */
    ZMQSubscriber(BatchProcessor& processor, const Config& config);

    /**
     * @brief Destructor - ensures graceful shutdown
     */
    ~ZMQSubscriber();

    // Non-copyable, non-movable (manages ZMQ resources)
    ZMQSubscriber(const ZMQSubscriber&) = delete;
    ZMQSubscriber& operator=(const ZMQSubscriber&) = delete;
    ZMQSubscriber(ZMQSubscriber&&) = delete;
    ZMQSubscriber& operator=(ZMQSubscriber&&) = delete;

    //==========================================================================
    // CONTROL
    //==========================================================================

    /**
     * @brief Start receiving messages (blocking)
     *
     * This is the main event loop. It will:
     *   1. Connect to the ZMQ endpoint
     *   2. Subscribe to the configured topic
     *   3. Receive messages in a loop until stop() is called
     *   4. Parse DetectionBatch protobuf messages
     *   5. Forward detections to BatchProcessor
     *   6. Handle reconnection on errors
     *
     * @note This function blocks until stop() is called
     * @note Should be called from a dedicated thread
     *
     * @throws std::runtime_error if called while already running
     */
    void run();

    /**
     * @brief Stop receiving messages (thread-safe)
     *
     * Sets the stop flag, causing run() to exit gracefully after
     * the current recv() completes (or times out).
     *
     * @note Can be called from any thread
     * @note Safe to call multiple times
     */
    void stop();

    /**
     * @brief Check if subscriber is running
     *
     * @return true if run() is executing, false otherwise
     */
    bool is_running() const {
        return running_.load();
    }

    //==========================================================================
    // METRICS
    //==========================================================================

    /**
     * @brief Get current statistics
     *
     * @return Reference to stats (thread-safe via atomics)
     */
    const Stats& get_stats() const {
        return stats_;
    }

    /**
     * @brief Get configuration
     */
    const Config& get_config() const {
        return config_;
    }

private:
    //==========================================================================
    // INTERNAL METHODS
    //==========================================================================

    /**
     * @brief Connect/reconnect to ZMQ endpoint
     *
     * @throws zmq::error_t on connection failure
     */
    void connect();

    /**
     * @brief Main receive loop
     *
     * Called by run() after successful connection.
     * Continues until running_ becomes false.
     */
    void receive_loop();

    /**
     * @brief Handle a single received message
     *
     * @param msg_data Raw message bytes
     * @param msg_size Size of message in bytes
     *
     * Parses DetectionBatch and forwards to processor.
     * Catches and logs parse errors without crashing.
     */
    void handle_message(const void* msg_data, size_t msg_size);

    /**
     * @brief Handle reconnection with exponential backoff
     *
     * Increments reconnect counter and sleeps for backoff interval.
     * Backoff doubles each time up to max_reconnect_interval_ms.
     */
    void handle_reconnect();

    /**
     * @brief Reset reconnection backoff to initial interval
     *
     * Called after successful connection.
     */
    void reset_reconnect_backoff();

    //==========================================================================
    // MEMBER VARIABLES
    //==========================================================================

    BatchProcessor& processor_;          ///< Target for detections
    Config config_;                      ///< Configuration
    mutable Stats stats_;                ///< Runtime statistics

    std::unique_ptr<zmq::context_t> context_;  ///< ZMQ context
    std::unique_ptr<zmq::socket_t> socket_;    ///< ZMQ SUB socket

    std::atomic<bool> running_;          ///< Event loop control flag
    int current_reconnect_interval_ms_;  ///< Current backoff interval
};

} // namespace firewall
} // namespace mldefender

#endif // FIREWALL_ZMQ_SUBSCRIBER_HPP