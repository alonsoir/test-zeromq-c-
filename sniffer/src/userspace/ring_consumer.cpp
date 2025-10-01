#include "compression_handler.hpp"
#include "ring_consumer.hpp"
#include <iostream>
#include <cstring>
#include <arpa/inet.h>
#include <google/protobuf/timestamp.pb.h>
#include <chrono>
#include <iomanip>

namespace sniffer {

RingBufferConsumer::RingBufferConsumer(const SnifferConfig& config)
    : config_(config), ring_buf_(nullptr), ring_fd_(-1) {

    stats_.start_time = std::chrono::steady_clock::now();

    std::cout << "[INFO] Enhanced RingBufferConsumer created with config:" << std::endl;
    std::cout << "  - Ring consumer threads: " << config_.threading.ring_consumer_threads << std::endl;
    std::cout << "  - Feature processor threads: " << config_.threading.feature_processor_threads << std::endl;
    std::cout << "  - ZMQ sender threads: " << config_.threading.zmq_sender_threads << std::endl;
    std::cout << "  - Batch size: " << config_.buffers.batch_processing_size << std::endl;
}

RingBufferConsumer::~RingBufferConsumer() {
    if (running_) {
        stop();
    }
    cleanup_resources();
}

bool RingBufferConsumer::initialize(int ring_fd, std::shared_ptr<ThreadManager> thread_manager) {
    if (initialized_) {
        std::cout << "[WARNING] RingBufferConsumer already initialized" << std::endl;
        return true;
    }

    if (ring_fd < 0) {
        std::cerr << "[ERROR] Invalid ring buffer file descriptor: " << ring_fd << std::endl;
        return false;
    }

    if (!thread_manager) {
        std::cerr << "[ERROR] ThreadManager is required for enhanced RingBufferConsumer" << std::endl;
        return false;
    }

    if (!validate_configuration()) {
        std::cerr << "[ERROR] Invalid configuration" << std::endl;
        return false;
    }

    ring_fd_ = ring_fd;
    thread_manager_ = thread_manager;

    // Initialize ring buffer
    ring_buf_ = ring_buffer__new(ring_fd_, handle_event, this, nullptr);
    if (!ring_buf_) {
        std::cerr << "[ERROR] Failed to create ring buffer consumer" << std::endl;
        return false;
    }

    // Initialize ZeroMQ
    if (!initialize_zmq()) {
        std::cerr << "[ERROR] Failed to initialize ZeroMQ" << std::endl;
        return false;
    }

    // Initialize buffers
    if (!initialize_buffers()) {
        std::cerr << "[ERROR] Failed to initialize buffers" << std::endl;
        return false;
    }

    // Initialize batching
    current_batch_ = std::make_unique<EventBatch>(get_optimal_batch_size());

    initialized_ = true;
    std::cout << "[INFO] Enhanced RingBufferConsumer initialized successfully" << std::endl;
    std::cout << "  - Ring buffer FD: " << ring_fd_ << std::endl;
    std::cout << "  - ZMQ sockets: " << zmq_sockets_.size() << std::endl;
    std::cout << "  - Optimal batch size: " << get_optimal_batch_size() << std::endl;

    return true;
}

bool RingBufferConsumer::start() {
    if (!initialized_) {
        std::cerr << "[ERROR] Cannot start - not initialized. Call initialize() first" << std::endl;
        return false;
    }

    if (running_) {
        std::cout << "[INFO] Enhanced RingBufferConsumer already running" << std::endl;
        return true;
    }

    should_stop_ = false;
    running_ = true;
    active_consumers_ = 0;

    // Start ring buffer consumer threads
    int consumer_count = config_.threading.ring_consumer_threads;
    consumer_threads_.reserve(consumer_count);

    for (int i = 0; i < consumer_count; ++i) {
        consumer_threads_.emplace_back(&RingBufferConsumer::ring_consumer_loop, this, i);
        active_consumers_++;
    }

	// Start feature processor threads
    for (int i = 0; i < config_.threading.feature_processor_threads; ++i) {
        consumer_threads_.emplace_back(&RingBufferConsumer::feature_processor_loop, this);
    }

	// Start ZMQ sender threads
    for (int i = 0; i < config_.threading.zmq_sender_threads; ++i) {
        consumer_threads_.emplace_back(&RingBufferConsumer::zmq_sender_loop, this);
    }

    stats_.start_time = std::chrono::steady_clock::now();

    std::cout << "[INFO] Enhanced RingBufferConsumer started with " << consumer_count
              << " ring consumer threads" << std::endl;
    std::cout << "[INFO] + " << config_.threading.feature_processor_threads
              << " feature processor threads" << std::endl;
    std::cout << "[INFO] + " << config_.threading.zmq_sender_threads
              << " ZMQ sender threads" << std::endl;
    std::cout << "[INFO] Multi-threaded protobuf pipeline active" << std::endl;

    return true;
}

void RingBufferConsumer::stop() {
    if (!running_) {
        return;
    }

    std::cout << "[INFO] Stopping Enhanced RingBufferConsumer..." << std::endl;
    should_stop_ = true;

    // Wake up all waiting threads
    processing_queue_cv_.notify_all();
    send_queue_cv_.notify_all();

    // Flush any remaining batch
    flush_current_batch();

    // Stop consumer threads
    shutdown_consumers();

    // Stop ZMQ
    shutdown_zmq();

    running_ = false;
    active_consumers_ = 0;

    // Print final statistics
    print_stats();

    std::cout << "[INFO] Enhanced RingBufferConsumer stopped" << std::endl;
}

bool RingBufferConsumer::initialize_zmq() {
    try {
        // Create ZeroMQ context with optimal IO threads
        int io_threads = config_.zmq.io_thread_pools;
        zmq_context_ = std::make_unique<zmq::context_t>(io_threads);

        // Create socket pool
        int socket_count = config_.zmq.socket_pools.push_sockets;
        zmq_sockets_.reserve(socket_count);

        std::string endpoint = "tcp://" + config_.network.output_socket.address + ":" +
                              std::to_string(config_.network.output_socket.port);
		std::cout << "[DEBUG] config_.network.output_socket.address: " << config_.network.output_socket.address << std::endl;
		std::cout << "[DEBUG] endpoint: " << endpoint << std::endl;
		std::cout << "[DEBUG] config_.network.output_socket.port: " << config_.network.output_socket.port << std::endl;
		std::cout << "[DEBUG] to_string(config_.network.output_socket.port): " << std::to_string(config_.network.output_socket.port) << endpoint << std::endl;
		std::cout << "[DEBUG] endpoint: " << endpoint << std::endl;

		for (int i = 0; i < socket_count; ++i) {
            auto socket = std::make_unique<zmq::socket_t>(*zmq_context_, ZMQ_PUSH);

            // Configure socket with enhanced settings
			socket->set(zmq::sockopt::sndhwm, static_cast<int>(config_.zmq.connection_settings.sndhwm));
			socket->set(zmq::sockopt::linger, static_cast<int>(config_.zmq.connection_settings.linger_ms));
			socket->set(zmq::sockopt::sndbuf, static_cast<int>(config_.zmq.connection_settings.sndbuf));
			socket->set(zmq::sockopt::tcp_keepalive, static_cast<int>(config_.zmq.connection_settings.tcp_keepalive));

            // Bind socket
            if (config_.network.output_socket.mode == "bind") {
                socket->bind(endpoint);
            } else {
                socket->connect(endpoint);
            }

            zmq_sockets_.push_back(std::move(socket));
        }

        std::cout << "[INFO] ZeroMQ initialized with " << socket_count
                  << " sockets to " << endpoint << std::endl;
        return true;

    } catch (const zmq::error_t& e) {
        std::cerr << "[ERROR] ZeroMQ initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool RingBufferConsumer::initialize_buffers() {
    int thread_count = config_.threading.ring_consumer_threads;

    // Pre-allocate IP buffers for each consumer thread
    ip_buffers_src_.resize(thread_count);
    ip_buffers_dst_.resize(thread_count);

    std::cout << "[INFO] Initialized " << thread_count << " buffer sets for consumer threads" << std::endl;
    return true;
}

bool RingBufferConsumer::validate_configuration() const {
    if (config_.threading.ring_consumer_threads <= 0) {
        std::cerr << "[ERROR] Ring consumer threads must be > 0" << std::endl;
        return false;
    }

    if (config_.buffers.batch_processing_size == 0) {
        std::cerr << "[ERROR] Batch processing size must be > 0" << std::endl;
        return false;
    }

    if (config_.zmq.socket_pools.push_sockets <= 0) {
        std::cerr << "[ERROR] ZMQ socket count must be > 0" << std::endl;
        return false;
    }

    return true;
}

void RingBufferConsumer::ring_consumer_loop(int consumer_id) {
    std::cout << "[INFO] Ring consumer " << consumer_id << " started" << std::endl;

    while (!should_stop_) {
        // Poll ring buffer with configurable timeout
        auto timeout_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            get_processing_timeout()).count();

        int err = ring_buffer__poll(ring_buf_, static_cast<int>(timeout_ms));
        stats_.ring_buffer_polls++;

        if (err < 0) {
            if (err != -EINTR) {
                handle_ring_buffer_error(err);
                break;
            }
        } else if (err == 0) {
            stats_.ring_buffer_timeouts++;
        }
    }

    active_consumers_--;
    std::cout << "[INFO] Ring consumer " << consumer_id << " stopped" << std::endl;
}

void RingBufferConsumer::feature_processor_loop() {
    while (!should_stop_) {
        SimpleEvent event;

        {
            std::unique_lock<std::mutex> lock(processing_queue_mutex_);
            processing_queue_cv_.wait_for(lock, get_processing_timeout(), [this] {
                return !processing_queue_.empty() || should_stop_;
            });

            if (processing_queue_.empty()) {
                continue;
            }

            event = processing_queue_.front();
            processing_queue_.pop();
        }

        process_event_features(event);
    }
}

void RingBufferConsumer::zmq_sender_loop() {
    while (!should_stop_) {
        std::vector<uint8_t> data;

        {
            std::unique_lock<std::mutex> lock(send_queue_mutex_);
            send_queue_cv_.wait_for(lock, get_processing_timeout(), [this] {
                return !send_queue_.empty() || should_stop_;
            });

            if (send_queue_.empty()) {
                continue;
            }

            data = std::move(send_queue_.front());
            send_queue_.pop();
        }

        send_protobuf_message(data);
    }
}

int RingBufferConsumer::handle_event(void* ctx, void* data, size_t data_sz) {
	std::cerr << "[DEBUG] CALLBACK CALLED! data_sz=" << data_sz << std::endl;
    RingBufferConsumer* consumer = static_cast<RingBufferConsumer*>(ctx);

    if (data_sz != sizeof(SimpleEvent)) {
    	std::cerr << "[ERROR] SIZE MISMATCH! Received: " << data_sz
              << " bytes, Expected: " << sizeof(SimpleEvent)
              << " bytes - DROPPING EVENT" << std::endl;
    	consumer->stats_.events_dropped++;
    	return 0;
}

    const SimpleEvent* event = static_cast<const SimpleEvent*>(data);
    consumer->stats_.events_processed++;

    if (consumer->stats_.events_processed % 10 == 0) {
        std::cout << "[DEBUG] Eventos procesados: " << consumer->stats_.events_processed << std::endl;
    }

    // Process the event in the current consumer thread context
    consumer->process_raw_event(*event, consumer->active_consumers_.load());

    return 0;
}

void RingBufferConsumer::process_raw_event(const SimpleEvent& event, int consumer_id) {
    auto start_time = std::chrono::steady_clock::now();

    // Call external callback if set
    if (external_callback_) {
        try {
            external_callback_(event);
        } catch (const std::exception& e) {
            std::cerr << "[WARNING] External callback failed: " << e.what() << std::endl;
        }
    }

    // Add to batch for efficient processing
    add_to_batch(event);

    // Update processing time statistics
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    update_processing_time(duration);
}

void RingBufferConsumer::process_event_features(const SimpleEvent& event) {
    try {
        // Create protobuf message
        protobuf::NetworkSecurityEvent proto_event;
        populate_protobuf_event(event, proto_event, 0);  // Use buffer index 0 for feature processing

        // Serialize to binary
        std::vector<uint8_t> serialized_data;
        std::string serialized_string;

        if (!proto_event.SerializeToString(&serialized_string)) {
            stats_.protobuf_serialization_failures++;
            return;
        }

        // Convert to vector for queue
        serialized_data.assign(serialized_string.begin(), serialized_string.end());

        // Add to send queue
        {
            std::lock_guard<std::mutex> lock(send_queue_mutex_);
            send_queue_.push(std::move(serialized_data));
        }
        send_queue_cv_.notify_one();

    } catch (const std::exception& e) {
        handle_protobuf_error(e);
    }
}

void RingBufferConsumer::send_event_batch(const std::vector<SimpleEvent>& events) {
    for (const auto& event : events) {
        // Submit to feature processing queue
        {
            std::lock_guard<std::mutex> lock(processing_queue_mutex_);
            processing_queue_.push(event);
        }
        processing_queue_cv_.notify_one();
    }
}

bool RingBufferConsumer::send_protobuf_message(const std::vector<uint8_t>& serialized_data) {
    try {
        std::vector<uint8_t> data_to_send;

        // Comprimir si está habilitado y el tamaño es mayor al mínimo
        if (config_.transport.compression.enabled &&
            serialized_data.size() >= config_.transport.compression.min_compress_size) {

            try {
                // Crear instancia de CompressionHandler y comprimir
                CompressionHandler compressor;
                auto compressed = compressor.compress_lz4(
                serialized_data.data(),
                serialized_data.size()
            );

                data_to_send = std::move(compressed);

            } catch (const std::exception& e) {
                // Si falla la compresión, envía sin comprimir
                std::cerr << "[WARNING] LZ4 compression failed: " << e.what()
                 << ", sending uncompressed" << std::endl;
                data_to_send = serialized_data;
            }
        } else {
            // No comprimir
            data_to_send = serialized_data;
        }

        zmq::message_t message(data_to_send.size());
        memcpy(message.data(), data_to_send.data(), data_to_send.size());

        // Get next socket using round-robin
        zmq::socket_t* socket = get_next_socket();
        auto result = socket->send(message, zmq::send_flags::dontwait);

        if (result.has_value()) {
            stats_.events_sent++;
            return true;
        } else {
            stats_.zmq_send_failures++;
            return false;
        }

    } catch (const zmq::error_t& e) {
        handle_zmq_error(e);
        return false;
    }
}

zmq::socket_t* RingBufferConsumer::get_next_socket() {
    size_t current = socket_round_robin_.fetch_add(1) % zmq_sockets_.size();
    return zmq_sockets_[current].get();
}

void RingBufferConsumer::add_to_batch(const SimpleEvent& event) {
    std::lock_guard<std::mutex> lock(batch_mutex_);

    if (!current_batch_) {
        current_batch_ = std::make_unique<EventBatch>(get_optimal_batch_size());
    }

    current_batch_->events.push_back(event);

    if (current_batch_->is_ready()) {
        send_event_batch(current_batch_->events);
        current_batch_ = std::make_unique<EventBatch>(get_optimal_batch_size());
    }
}

void RingBufferConsumer::flush_current_batch() {
    std::lock_guard<std::mutex> lock(batch_mutex_);

    if (current_batch_ && !current_batch_->events.empty()) {
        send_event_batch(current_batch_->events);
        current_batch_.reset();
    }
}

void RingBufferConsumer::populate_protobuf_event(const SimpleEvent& event,
                                                 protobuf::NetworkSecurityEvent& proto_event,
                                                 int buffer_index) const {

    // Use pre-allocated buffers for IP conversion
    struct in_addr src_addr = {.s_addr = event.src_ip};
    struct in_addr dst_addr = {.s_addr = event.dst_ip};

    char* src_buffer = ip_buffers_src_[buffer_index % ip_buffers_src_.size()].data();
    char* dst_buffer = ip_buffers_dst_[buffer_index % ip_buffers_dst_.size()].data();

    inet_ntop(AF_INET, &src_addr, src_buffer, 16);
    inet_ntop(AF_INET, &dst_addr, dst_buffer, 16);

    // Generate event ID
    std::string event_id = std::to_string(event.timestamp) + "_" +
                          std::to_string(event.src_ip ^ event.dst_ip);

    // Populate basic fields
    proto_event.set_event_id(event_id);
    proto_event.set_originating_node_id(config_.node_id);
    proto_event.set_correlation_id(event_id);
    proto_event.set_schema_version(31);
    proto_event.set_protobuf_version("3.1.0");

    // Timestamp
    google::protobuf::Timestamp* timestamp = proto_event.mutable_event_timestamp();
    timestamp->set_seconds(event.timestamp / 1000000000ULL);
    timestamp->set_nanos(event.timestamp % 1000000000ULL);

    // Network features
    protobuf::NetworkFeatures* features = proto_event.mutable_network_features();
    features->set_source_ip(src_buffer);
    features->set_destination_ip(dst_buffer);
    features->set_source_port(event.src_port);
    features->set_destination_port(event.dst_port);
    features->set_protocol_number(event.protocol);
    features->set_protocol_name(protocol_to_string(event.protocol));

    // Basic packet features
    features->set_total_forward_packets(1);
    features->set_total_backward_packets(0);
    features->set_total_forward_bytes(event.packet_len);
    features->set_total_backward_bytes(0);
    features->set_minimum_packet_length(event.packet_len);
    features->set_maximum_packet_length(event.packet_len);
    features->set_packet_length_mean(static_cast<double>(event.packet_len));
    features->set_average_packet_size(static_cast<double>(event.packet_len));

    // Flow start time
    google::protobuf::Timestamp* flow_start = features->mutable_flow_start_time();
    flow_start->set_seconds(event.timestamp / 1000000000ULL);
    flow_start->set_nanos(event.timestamp % 1000000000ULL);

    // Time window
    protobuf::TimeWindow* time_window = proto_event.mutable_time_window();
    google::protobuf::Timestamp* window_start = time_window->mutable_window_start();
    window_start->set_seconds(event.timestamp / 1000000000ULL);
    window_start->set_nanos(event.timestamp % 1000000000ULL);
    time_window->set_window_type(protobuf::TimeWindow::SLIDING);
    time_window->set_sequence_number(event.timestamp);

    // Distributed node info
    protobuf::DistributedNode* node = proto_event.mutable_capturing_node();
    node->set_node_id(config_.node_id);
    node->set_node_hostname("ebpf_sniffer_enhanced");
    node->set_node_role(protobuf::DistributedNode::PACKET_SNIFFER);
    node->set_node_status(protobuf::DistributedNode::ACTIVE);
    node->set_agent_version("3.1.0");

    // Heartbeat timestamp
    google::protobuf::Timestamp* heartbeat = node->mutable_last_heartbeat();
    heartbeat->set_seconds(event.timestamp / 1000000000ULL);
    heartbeat->set_nanos(event.timestamp % 1000000000ULL);

    // Classification
    proto_event.set_overall_threat_score(0.0);
    proto_event.set_final_classification("UNCATEGORIZED");
    proto_event.set_threat_category("RAW_CAPTURE");

    // Tags
    proto_event.add_event_tags("raw_ebpf_capture");
    proto_event.add_event_tags("enhanced_multithreaded");
    proto_event.add_event_tags("requires_processing");
}

std::string RingBufferConsumer::protocol_to_string(uint8_t protocol) const {
    switch (protocol) {
        case 1: return "ICMP";
        case 6: return "TCP";
        case 17: return "UDP";
        case 47: return "GRE";
        case 50: return "ESP";
        case 51: return "AH";
        case 58: return "ICMPv6";
        default: return "UNKNOWN";
    }
}

// Statistics and utility methods
RingConsumerStats RingBufferConsumer::get_stats() const {
    return stats_;
}

void RingBufferConsumer::print_stats() const {
    auto now = std::chrono::steady_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::seconds>(now - stats_.start_time).count();

    std::cout << "\n=== Enhanced RingBufferConsumer Statistics ===" << std::endl;
    std::cout << "Runtime: " << runtime << " seconds" << std::endl;
    std::cout << "Events processed: " << stats_.events_processed << std::endl;
    std::cout << "Events sent: " << stats_.events_sent << std::endl;
    std::cout << "Events dropped: " << stats_.events_dropped << std::endl;
    std::cout << "Protobuf failures: " << stats_.protobuf_serialization_failures << std::endl;
    std::cout << "ZMQ send failures: " << stats_.zmq_send_failures << std::endl;
    std::cout << "Ring buffer polls: " << stats_.ring_buffer_polls << std::endl;
    std::cout << "Ring buffer timeouts: " << stats_.ring_buffer_timeouts << std::endl;

    if (runtime > 0) {
        std::cout << "Events per second: " << (stats_.events_processed / runtime) << std::endl;
        std::cout << "Send rate: " << (stats_.events_sent / runtime) << std::endl;
    }

    std::cout << "Active consumers: " << active_consumers_ << std::endl;
    std::cout << "Processing queue size: " << processing_queue_.size() << std::endl;
    std::cout << "Send queue size: " << send_queue_.size() << std::endl;
    std::cout << "=============================================" << std::endl;
}

void RingBufferConsumer::reset_stats() {
    stats_ = RingConsumerStats{};
    stats_.start_time = std::chrono::steady_clock::now();
}

void RingBufferConsumer::set_event_callback(EventCallback callback) {
    external_callback_ = std::move(callback);
}

// Configuration helpers
size_t RingBufferConsumer::get_optimal_batch_size() const {
    return config_.buffers.batch_processing_size;
}

int RingBufferConsumer::get_optimal_consumer_count() const {
    return config_.threading.ring_consumer_threads;
}

std::chrono::milliseconds RingBufferConsumer::get_processing_timeout() const {
    return std::chrono::milliseconds(config_.zmq.queue_management.queue_timeout_ms);
}

// Error handling
void RingBufferConsumer::handle_ring_buffer_error(int error_code) {
    std::cerr << "[ERROR] Ring buffer error: " << strerror(-error_code) << std::endl;
    stats_.events_dropped++;
}

void RingBufferConsumer::handle_zmq_error(const zmq::error_t& error) {
    if (error.num() != EAGAIN) {
        std::cerr << "[ERROR] ZeroMQ error: " << error.what() << std::endl;
    }
    stats_.zmq_send_failures++;
}

void RingBufferConsumer::handle_protobuf_error(const std::exception& error) {
    std::cerr << "[ERROR] Protobuf error: " << error.what() << std::endl;
    stats_.protobuf_serialization_failures++;
}

// Shutdown methods
void RingBufferConsumer::shutdown_consumers() {
    for (auto& thread : consumer_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    consumer_threads_.clear();
}

void RingBufferConsumer::shutdown_zmq() {
    for (auto& socket : zmq_sockets_) {
        if (socket) {
            socket->close();
        }
    }
    zmq_sockets_.clear();
    zmq_context_.reset();
}

void RingBufferConsumer::cleanup_resources() {
    if (ring_buf_) {
        ring_buffer__free(ring_buf_);
        ring_buf_ = nullptr;
    }
}

void RingBufferConsumer::update_processing_time(std::chrono::microseconds duration) {
    stats_.total_processing_time_us += duration.count();

    // Update rolling average (simple exponential moving average)
    double current_avg = stats_.average_processing_time_us.load();
    double new_avg = 0.9 * current_avg + 0.1 * duration.count();
    stats_.average_processing_time_us = new_avg;
}

} // namespace sniffer