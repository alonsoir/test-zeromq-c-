// sniffer/include/ring_consumer.hpp
#pragma once

#include "main.h"
#include "network_security.pb.h"
#include "thread_manager.hpp"
#include <bpf/libbpf.h>
#include <zmq.hpp>
#include <memory>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <array>
#include <functional>

namespace sniffer {

struct RingConsumerStats {
    std::atomic<uint64_t> events_processed{0};
    std::atomic<uint64_t> events_sent{0};
    std::atomic<uint64_t> events_dropped{0};
    std::atomic<uint64_t> protobuf_serialization_failures{0};
    std::atomic<uint64_t> zmq_send_failures{0};
    std::atomic<uint64_t> ring_buffer_polls{0};
    std::atomic<uint64_t> ring_buffer_timeouts{0};
    std::atomic<uint64_t> total_processing_time_us{0};
    std::atomic<double> average_processing_time_us{0.0};
    std::chrono::steady_clock::time_point start_time;
};

// Snapshot for reading stats (non-atomic copy)
struct RingConsumerStatsSnapshot {
    uint64_t events_processed;
    uint64_t events_sent;
    uint64_t events_dropped;
    uint64_t protobuf_serialization_failures;
    uint64_t zmq_send_failures;
    uint64_t ring_buffer_polls;
    uint64_t ring_buffer_timeouts;
    uint64_t total_processing_time_us;
    double average_processing_time_us;
    std::chrono::steady_clock::time_point start_time;
};

struct EventBatch {
    std::vector<SimpleEvent> events;
    size_t max_size;

    explicit EventBatch(size_t size) : max_size(size) {
        events.reserve(size);
    }

    bool is_ready() const { return events.size() >= max_size; }
};

class RingBufferConsumer {
public:
    using EventCallback = std::function<void(const SimpleEvent&)>;

    explicit RingBufferConsumer(const SnifferConfig& config);
    ~RingBufferConsumer();

    bool initialize(int ring_fd, std::shared_ptr<ThreadManager> thread_manager);
    bool start();
    void stop();

    // Statistics
    RingConsumerStatsSnapshot get_stats() const;
    void print_stats() const;
    void reset_stats();

    // Configuration
    void set_event_callback(EventCallback callback);
    void set_stats_interval(int seconds) { stats_interval_seconds_ = seconds; }

private:
    // Core functionality
    bool initialize_zmq();
    bool initialize_buffers();
    bool validate_configuration() const;

    // Thread loops
    void ring_consumer_loop(int consumer_id);
    void feature_processor_loop();
    void zmq_sender_loop();
    void stats_display_loop();  // NEW: Thread for periodic stats

    // Event processing
    static int handle_event(void* ctx, void* data, size_t data_sz);
    void process_raw_event(const SimpleEvent& event, int consumer_id);
    void process_event_features(const SimpleEvent& event);

    // Batching
    void add_to_batch(const SimpleEvent& event);
    void flush_current_batch();
    void send_event_batch(const std::vector<SimpleEvent>& events);

    // Protobuf
    void populate_protobuf_event(const SimpleEvent& event,
                                protobuf::NetworkSecurityEvent& proto_event,
                                int buffer_index) const;
    std::string protocol_to_string(uint8_t protocol) const;

    // ZMQ
    bool send_protobuf_message(const std::vector<uint8_t>& data);
    zmq::socket_t* get_next_socket();

    // Configuration helpers
    size_t get_optimal_batch_size() const;
    int get_optimal_consumer_count() const;
    std::chrono::milliseconds get_processing_timeout() const;

    // Error handling
    void handle_ring_buffer_error(int error_code);
    void handle_zmq_error(const zmq::error_t& error);
    void handle_protobuf_error(const std::exception& error);
    void update_processing_time(std::chrono::microseconds duration);

    // Shutdown
    void shutdown_consumers();
    void shutdown_zmq();
    void cleanup_resources();

    // Configuration
    SnifferConfig config_;
    int stats_interval_seconds_{30};  // Default 30 seconds

    // Ring buffer
    ring_buffer* ring_buf_;
    int ring_fd_;

    // Threading
    std::shared_ptr<ThreadManager> thread_manager_;
    std::vector<std::thread> consumer_threads_;
    std::atomic<int> active_consumers_{0};
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> initialized_{false};

    // Queues
    std::queue<SimpleEvent> processing_queue_;
    std::mutex processing_queue_mutex_;
    std::condition_variable processing_queue_cv_;

    std::queue<std::vector<uint8_t>> send_queue_;
    std::mutex send_queue_mutex_;
    std::condition_variable send_queue_cv_;

    // Batching
    std::unique_ptr<EventBatch> current_batch_;
    std::mutex batch_mutex_;

    // ZeroMQ
    std::unique_ptr<zmq::context_t> zmq_context_;
    std::vector<std::unique_ptr<zmq::socket_t>> zmq_sockets_;
    std::atomic<size_t> socket_round_robin_{0};

    // Pre-allocated buffers for IP conversion
    mutable std::vector<std::array<char, 16>> ip_buffers_src_;
    mutable std::vector<std::array<char, 16>> ip_buffers_dst_;

    // Statistics
    RingConsumerStats stats_;
    EventCallback external_callback_;
};

} // namespace sniffer