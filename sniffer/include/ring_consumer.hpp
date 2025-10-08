#pragma once
// sniffer/src/userspace/ring_consumer.hpp
#include "config_manager.hpp"
#include "thread_manager.hpp"
#include "../../protobuf/network_security.pb.h"
#include "main.h"
#include <functional>
#include <atomic>
#include <thread>
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <bpf/libbpf.h>
#include <zmq.hpp>
#include <string>
#include <arpa/inet.h>
#include <array>


namespace sniffer {

// Forward declarations
class ThreadManager;

// Callback para procesar eventos
using EventCallback = std::function<void(const SimpleEvent& event)>;

struct RingConsumerStats {
    std::atomic<uint64_t> events_processed{0};
    std::atomic<uint64_t> events_dropped{0};
    std::atomic<uint64_t> events_sent{0};
    std::atomic<uint64_t> protobuf_serialization_failures{0};
    std::atomic<uint64_t> zmq_send_failures{0};
    std::atomic<uint64_t> ring_buffer_polls{0};
    std::atomic<uint64_t> ring_buffer_timeouts{0};
    std::atomic<double> average_processing_time_us{0.0};
    std::chrono::steady_clock::time_point start_time;
    std::atomic<uint64_t> total_processing_time_us{0};

    // Constructor por defecto
    RingConsumerStats() : start_time(std::chrono::steady_clock::now()) {}

    // Constructor de copia
    RingConsumerStats(const RingConsumerStats& other)
        : events_processed(other.events_processed.load())
        , events_dropped(other.events_dropped.load())
        , events_sent(other.events_sent.load())
        , protobuf_serialization_failures(other.protobuf_serialization_failures.load())
        , zmq_send_failures(other.zmq_send_failures.load())
        , ring_buffer_polls(other.ring_buffer_polls.load())
        , ring_buffer_timeouts(other.ring_buffer_timeouts.load())
        , average_processing_time_us(other.average_processing_time_us.load())
        , start_time(other.start_time)
        , total_processing_time_us(other.total_processing_time_us.load()) {}

    // Operador de asignación
    RingConsumerStats& operator=(const RingConsumerStats& other) {
        if (this != &other) {
            events_processed.store(other.events_processed.load());
            events_dropped.store(other.events_dropped.load());
            events_sent.store(other.events_sent.load());
            protobuf_serialization_failures.store(other.protobuf_serialization_failures.load());
            zmq_send_failures.store(other.zmq_send_failures.load());
            ring_buffer_polls.store(other.ring_buffer_polls.load());
            ring_buffer_timeouts.store(other.ring_buffer_timeouts.load());
            average_processing_time_us.store(other.average_processing_time_us.load());
            start_time = other.start_time;
            total_processing_time_us.store(other.total_processing_time_us.load());
        }
        return *this;
    }
};

// Enhanced ring buffer consumer with multi-threading
class RingBufferConsumer {
public:
    // Constructor using full enhanced configuration
    explicit RingBufferConsumer(const SnifferConfig& config);
    ~RingBufferConsumer();

    // Initialize with enhanced configuration
    bool initialize(int ring_fd, std::shared_ptr<ThreadManager> thread_manager);

    // Start multi-threaded consumption
    bool start();

    // Stop all threads and cleanup
    void stop();

    // Status checking
    bool is_running() const { return running_; }
    bool is_initialized() const { return initialized_; }

    // Statistics and monitoring
    RingConsumerStats get_stats() const;
    void print_stats() const;
    void reset_stats();

    // Configuration access
    const SnifferConfig& get_config() const { return config_; }

    // Event callback for external processing (optional)
    void set_event_callback(EventCallback callback);

private:
    // Configuration and thread management
    SnifferConfig config_;
    std::shared_ptr<ThreadManager> thread_manager_;

    // Ring buffer components
    struct ring_buffer* ring_buf_;
    int ring_fd_;

    // State management
    std::atomic<bool> initialized_{false};
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};

    // Statistics
    RingConsumerStats stats_;

    // ZeroMQ components - enhanced with pools
    std::unique_ptr<zmq::context_t> zmq_context_;
    std::vector<std::unique_ptr<zmq::socket_t>> zmq_sockets_;
    std::atomic<size_t> socket_round_robin_{0};

    // Threading components
    std::vector<std::thread> consumer_threads_;
    std::atomic<int> active_consumers_{0};

    // Work queues for different processing stages
    std::queue<SimpleEvent> processing_queue_;
    std::mutex processing_queue_mutex_;
    std::condition_variable processing_queue_cv_;

    std::queue<std::vector<uint8_t>> send_queue_;
    std::mutex send_queue_mutex_;
    std::condition_variable send_queue_cv_;

    // Batching support
    struct EventBatch {
        std::vector<SimpleEvent> events;
        std::chrono::steady_clock::time_point created_time;
        size_t target_size;

        EventBatch(size_t target) : target_size(target) {
            events.reserve(target);
            created_time = std::chrono::steady_clock::now();
        }

        bool is_ready() const {
            return events.size() >= target_size ||
                   std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - created_time).count() > 10;
        }
    };

    std::unique_ptr<EventBatch> current_batch_;
    std::mutex batch_mutex_;

    mutable std::vector<std::array<char, 16>> ip_buffers_src_;  // One per consumer thread
    mutable std::vector<std::array<char, 16>> ip_buffers_dst_;  // One per consumer thread

    // Optional external callback
    EventCallback external_callback_;

    // Private methods
    bool initialize_zmq();
    bool initialize_buffers();
    bool validate_configuration() const;

    // Thread functions
    void ring_consumer_loop(int consumer_id);
    void feature_processor_loop();
    void zmq_sender_loop();

    // Event processing pipeline
    void process_raw_event(const SimpleEvent& event, int consumer_id);
    void process_event_features(const SimpleEvent& event);
    void send_event_batch(const std::vector<SimpleEvent>& events);

    // eBPF callback (static)
    static int handle_event(void* ctx, void* data, size_t data_sz);

    // Utility methods
    std::string protocol_to_string(uint8_t protocol) const;
    void populate_protobuf_event(const SimpleEvent& event,
                                protobuf::NetworkSecurityEvent& proto_event,
                                int buffer_index) const;

    bool send_protobuf_message(const std::vector<uint8_t>& serialized_data);
    zmq::socket_t* get_next_socket();

    // Statistics helpers
    void update_processing_time(std::chrono::microseconds duration);
    void log_performance_metrics() const;

    // Batch processing
    void add_to_batch(const SimpleEvent& event);
    void flush_current_batch();
    void process_batch_if_ready();

    // Configuration helpers
    size_t get_optimal_batch_size() const;
    int get_optimal_consumer_count() const;
    std::chrono::milliseconds get_processing_timeout() const;

    // Error handling and recovery
    void handle_ring_buffer_error(int error_code);
    void handle_zmq_error(const zmq::error_t& error);
    void handle_protobuf_error(const std::exception& error);

    // Shutdown coordination
    void shutdown_consumers();
    void shutdown_zmq();
    void cleanup_resources();
};

} // namespace sniffer