//===----------------------------------------------------------------------===//
// ML Defender - Firewall ACL Agent
// zmq_subscriber.cpp - ZMQ Subscriber Implementation (Day 26 - crypto-transport)
// firewall-acl-agent/src/api/zmq_subscriber.cpp
//===----------------------------------------------------------------------===//

#include "firewall/zmq_subscriber.hpp"
#include <crypto_transport/crypto.hpp>
#include <crypto_transport/compression.hpp>
#include <crypto_transport/utils.hpp>
#include <iostream>
#include <thread>
#include <stdexcept>
#include <cstring>

namespace mldefender {
namespace firewall {

// Helper functions for string <-> vector<uint8_t> conversion
namespace {
    inline std::vector<uint8_t> string_to_bytes(const std::string& str) {
        return std::vector<uint8_t>(str.begin(), str.end());
    }

    inline std::string bytes_to_string(const std::vector<uint8_t>& bytes) {
        return std::string(bytes.begin(), bytes.end());
    }
}

//==============================================================================
// LIFECYCLE
//==============================================================================

ZMQSubscriber::ZMQSubscriber(BatchProcessor& processor, const Config& config)
    : processor_(processor)
    , config_(config)
    , running_(false)
    , current_reconnect_interval_ms_(config.reconnect_interval_ms)
{
    // Create ZMQ context (1 IO thread is sufficient)
    try {
        context_ = std::make_unique<zmq::context_t>(1);
    } catch (const zmq::error_t& e) {
        throw std::runtime_error(
            std::string("Failed to create ZMQ context: ") + e.what()
        );
    }

    // Initialize logger
    try {
        logger_ = std::make_unique<FirewallLogger>(
            config_.log_directory,
            config_.log_queue_size
        );
        logger_->start();

        std::cerr << "[ZMQSubscriber] Logger initialized: "
                  << config_.log_directory << std::endl;
        std::cerr << "[ZMQSubscriber] Log queue size: "
                  << config_.log_queue_size << std::endl;

    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("Failed to initialize logger: ") + e.what()
        );
    }

    std::cerr << "[ZMQSubscriber] Initialized with endpoint: "
              << config_.endpoint << std::endl;
    std::cerr << "[ZMQSubscriber] Crypto: "
              << (config_.encryption_enabled ? "ENABLED" : "disabled")
              << " | Compression: "
              << (config_.compression_enabled ? "ENABLED" : "disabled")
              << std::endl;
}

ZMQSubscriber::~ZMQSubscriber() {
    // Ensure graceful shutdown
    if (running_.load()) {
        std::cerr << "[ZMQSubscriber] Destructor called while running, stopping..."
                  << std::endl;
        stop();

        // Give it a moment to stop gracefully
        for (int i = 0; i < 10 && running_.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    // Stop logger and print statistics
    if (logger_) {
        std::cerr << "[ZMQSubscriber] Stopping logger..." << std::endl;
        logger_->stop(5000);  // 5 second timeout

        std::cerr << "[ZMQSubscriber] Logger statistics:" << std::endl;
        std::cerr << "  Events logged:  " << logger_->total_logged() << std::endl;
        std::cerr << "  Events dropped: " << logger_->total_dropped() << std::endl;
        std::cerr << "  Queue size:     " << logger_->queue_size() << std::endl;
    }

    // Close socket before destroying context
    if (socket_) {
        try {
            socket_->close();
        } catch (const zmq::error_t& e) {
            std::cerr << "[ZMQSubscriber] Error closing socket: "
                      << e.what() << std::endl;
        }
    }

    // Context will be destroyed automatically
    std::cerr << "[ZMQSubscriber] Destroyed" << std::endl;
}

//==============================================================================
// CONTROL
//==============================================================================

void ZMQSubscriber::run() {
    // Check if already running
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        throw std::runtime_error("ZMQSubscriber::run() called while already running");
    }

    std::cerr << "[ZMQSubscriber] Starting event loop..." << std::endl;

    try {
        while (running_.load()) {
            try {
                // Connect/reconnect
                connect();
                reset_reconnect_backoff();

                std::cerr << "[ZMQSubscriber] Connected successfully" << std::endl;

                // Main receive loop
                receive_loop();

            } catch (const zmq::error_t& e) {
                // ZMQ error - likely connection issue
                std::cerr << "[ZMQSubscriber] ZMQ error: " << e.what() << std::endl;

                if (config_.enable_reconnect && running_.load()) {
                    handle_reconnect();
                } else {
                    break;
                }

            } catch (const std::exception& e) {
                // Other error
                std::cerr << "[ZMQSubscriber] Error: " << e.what() << std::endl;

                if (config_.enable_reconnect && running_.load()) {
                    handle_reconnect();
                } else {
                    break;
                }
            }
        }
    } catch (...) {
        std::cerr << "[ZMQSubscriber] Unexpected exception in event loop" << std::endl;
        running_.store(false);
        throw;
    }

    running_.store(false);
    std::cerr << "[ZMQSubscriber] Event loop stopped" << std::endl;
}

void ZMQSubscriber::stop() {
    std::cerr << "[ZMQSubscriber] Stop requested" << std::endl;
    running_.store(false);
}

//==============================================================================
// INTERNAL METHODS
//==============================================================================

void ZMQSubscriber::connect() {
    // Close existing socket if any
    if (socket_) {
        try {
            socket_->close();
        } catch (const zmq::error_t& e) {
            std::cerr << "[ZMQSubscriber] Error closing old socket: "
                      << e.what() << std::endl;
        }
        socket_.reset();
    }

    // Create new SUB socket
    socket_ = std::make_unique<zmq::socket_t>(*context_, zmq::socket_type::sub);

    // Set socket options
    try {
        // Receive timeout for graceful shutdown
        socket_->set(zmq::sockopt::rcvtimeo, config_.recv_timeout_ms);

        // Linger time on close
        socket_->set(zmq::sockopt::linger, config_.linger_ms);

        // High water mark (buffer size)
        socket_->set(zmq::sockopt::rcvhwm, 10000);

    } catch (const zmq::error_t& e) {
        throw std::runtime_error(
            std::string("Failed to set socket options: ") + e.what()
        );
    }

    // Connect to endpoint
    try {
        socket_->connect(config_.endpoint);
    } catch (const zmq::error_t& e) {
        throw std::runtime_error(
            std::string("Failed to connect to ") + config_.endpoint + ": " + e.what()
        );
    }

    // Subscribe to topic
    try {
        socket_->set(zmq::sockopt::subscribe, config_.topic);
    } catch (const zmq::error_t& e) {
        throw std::runtime_error(
            std::string("Failed to subscribe to topic '") +
            config_.topic + "': " + e.what()
        );
    }

    std::cerr << "[ZMQSubscriber] Connected to " << config_.endpoint
              << ", subscribed to topic: '" << config_.topic << "'" << std::endl;
}

void ZMQSubscriber::receive_loop() {
    while (running_.load()) {
        try {
            // Receive message
            zmq::message_t message;
            auto result = socket_->recv(message, zmq::recv_flags::none);

            if (!result) {
                // Timeout or EAGAIN - check if we should stop
                if (!running_.load()) {
                    break;
                }
                continue;
            }

            // Update metrics
            size_t msg_size = message.size();

            // Handle message
            if (msg_size > 0) {
                handle_message(message.data(), msg_size);
            }

        } catch (const zmq::error_t& e) {
            // Check for specific error codes
            if (e.num() == ETERM) {
                // Context terminated
                std::cerr << "[ZMQSubscriber] Context terminated" << std::endl;
                break;
            } else if (e.num() == EAGAIN) {
                // Timeout - check stop flag
                if (!running_.load()) {
                    break;
                }
                continue;
            } else {
                // Other ZMQ error - rethrow to trigger reconnect
                throw;
            }
        }
    }
}

void ZMQSubscriber::handle_message(const void* msg_data, size_t msg_size) {
    const uint8_t* data_ptr = static_cast<const uint8_t*>(msg_data);
    size_t data_size = msg_size;

    // If we have a topic filter, skip the topic prefix
    // ZMQ PUB/SUB sends: [TOPIC][DATA]
    if (!config_.topic.empty()) {
        size_t topic_len = config_.topic.length();

        if (msg_size > topic_len &&
            std::memcmp(data_ptr, config_.topic.c_str(), topic_len) == 0) {
            data_ptr += topic_len;
            data_size -= topic_len;

            std::cerr << "[ZMQSubscriber] DEBUG: Skipped topic prefix ("
                      << topic_len << " bytes), data: "
                      << data_size << " bytes" << std::endl;
        }
    }

    // Copy to vector for decrypt/decompress
    std::vector<uint8_t> data(data_ptr, data_ptr + data_size);
    size_t original_size = data_size;  // Store for decompress

    try {
        // STEP 1 - Decrypt if enabled (using crypto-transport)
        if (config_.encryption_enabled) {
            auto start = std::chrono::high_resolution_clock::now();

            // Convert hex key to bytes
            auto key = crypto_transport::hex_to_bytes(config_.crypto_token);

            // Decrypt
            data = crypto_transport::decrypt(data, key);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            std::cerr << "[ZMQSubscriber] 🔓 Decrypted: " << duration.count() << " µs "
                      << "(" << original_size << " → " << data.size() << " bytes)" << std::endl;

            original_size = data.size();  // Update for decompress
        }

        // STEP 2 - Decompress if enabled (using crypto-transport)
        if (config_.compression_enabled) {
            auto start = std::chrono::high_resolution_clock::now();

            // Extract original size from first 4 bytes (little-endian)
            if (data.size() < 4) {
                throw std::runtime_error("Compressed data too small");
            }

            uint32_t decompressed_size;
            std::memcpy(&decompressed_size, data.data(), sizeof(uint32_t));

            // Remove size header and decompress
            std::vector<uint8_t> compressed_only(data.begin() + 4, data.end());
            data = crypto_transport::decompress(compressed_only, decompressed_size);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            std::cerr << "[ZMQSubscriber] 📦 Decompressed: " << duration.count() << " µs "
                      << "(" << compressed_only.size() << " → " << data.size() << " bytes)" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "[ZMQSubscriber] ❌ Decrypt/decompress failed: " << e.what() << std::endl;
        stats_.parse_errors++;
        return;
    }

    // Parse NetworkSecurityEvent protobuf (now data is plaintext)
    protobuf::NetworkSecurityEvent event;

    try {
        if (!event.ParseFromArray(data.data(), static_cast<int>(data.size()))) {
            std::cerr << "[ZMQSubscriber] Failed to parse NetworkSecurityEvent protobuf ("
                      << data.size() << " bytes)" << std::endl;
            stats_.parse_errors++;
            return;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ZMQSubscriber] Exception parsing protobuf: "
                  << e.what() << std::endl;
        stats_.parse_errors++;
        return;
    }

    std::cerr << "[DEBUG] Event parsed successfully" << std::endl;
    std::cerr << "[DEBUG]   has_ml_analysis: " << event.has_ml_analysis() << std::endl;

    if (event.has_ml_analysis()) {
        const auto& ml = event.ml_analysis();
        std::cerr << "[DEBUG]   attack_detected_level1: "
                  << ml.attack_detected_level1() << std::endl;
        std::cerr << "[DEBUG]   level1_confidence: "
                  << ml.level1_confidence() << std::endl;
    }

    std::cerr << "[DEBUG]   has_network_features: "
              << event.has_network_features() << std::endl;
    std::cerr << "[DEBUG]   threat_category: "
              << event.threat_category() << std::endl;

    stats_.messages_received++;

    // Check if event has ML analysis
    if (!event.has_ml_analysis()) {
        return;
    }

    const auto& ml = event.ml_analysis();

    // Only process if attack was detected at Level 1
    if (!ml.attack_detected_level1()) {
        return;
    }

    // Extract source IP from network_features
    if (!event.has_network_features()) {
        std::cerr << "[ZMQSubscriber] Event missing network_features, skipping" << std::endl;
        return;
    }

    const auto& nf = event.network_features();
    if (nf.source_ip().empty()) {
        std::cerr << "[ZMQSubscriber] Event missing source_ip, skipping" << std::endl;
        return;
    }

    // Determine ipset and timeout based on threat category
    std::string ipset_name = "ml_defender_blacklist_test";
    int timeout_sec = 300;  // 5 minutes default

    std::string threat_cat = event.threat_category();
    if (threat_cat == "DDOS") {
        timeout_sec = 600;  // DDoS: 10 minutes
    } else if (threat_cat == "RANSOMWARE") {
        timeout_sec = 3600; // Ransomware: 1 hour
    }

    // Create detection for BatchProcessor
    std::vector<protobuf::Detection> detections;
    protobuf::Detection detection;

    detection.set_src_ip(nf.source_ip());

    // Set detection type based on threat category
    if (threat_cat == "DDOS") {
        detection.set_type(protobuf::DetectionType::DETECTION_DDOS);
    } else if (threat_cat == "RANSOMWARE") {
        detection.set_type(protobuf::DetectionType::DETECTION_RANSOMWARE);
    } else if (threat_cat == "SUSPICIOUS_INTERNAL") {
        detection.set_type(protobuf::DetectionType::DETECTION_INTERNAL_THREAT);
    } else {
        detection.set_type(protobuf::DetectionType::DETECTION_SUSPICIOUS_TRAFFIC);
    }

    detection.set_confidence(ml.level1_confidence());
    detection.set_timestamp(event.event_timestamp().seconds());
    detection.set_description("Attack detected: " + threat_cat);

    detections.push_back(detection);

    // Forward to BatchProcessor
    try {
        processor_.add_detections(detections);
        stats_.detections_processed += detections.size();

        std::cerr << "[ZMQSubscriber] Processed attack from " << nf.source_ip()
                  << " (type: " << threat_cat << ", conf: "
                  << ml.level1_confidence() << ")" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[ZMQSubscriber] Error forwarding detections to processor: "
                  << e.what() << std::endl;
        return;  // Don't log if we didn't actually block
    }

    // Log the blocked event
    try {
        BlockedEvent log_event = create_blocked_event_from_proto(
            event,
            "BLOCKED",
            ipset_name,
            timeout_sec
        );

        bool logged = logger_->log_blocked_event(log_event);

        if (logged) {
            stats_.events_logged++;

            std::cerr << "[ZMQSubscriber] ✅ Logged event: "
                      << nf.source_ip() << " → "
                      << config_.log_directory << "/"
                      << log_event.timestamp_ms << ".{json,proto}"
                      << std::endl;
        } else {
            stats_.log_errors++;
            std::cerr << "[ZMQSubscriber] ⚠️  Log queue full, event dropped (IP: "
                      << nf.source_ip() << ")" << std::endl;
        }

    } catch (const std::exception& e) {
        stats_.log_errors++;
        std::cerr << "[ZMQSubscriber] ❌ Error logging event: "
                  << e.what() << std::endl;
    }
}

void ZMQSubscriber::handle_reconnect() {
    stats_.reconnects++;

    std::cerr << "[ZMQSubscriber] Connection lost, reconnecting in "
              << current_reconnect_interval_ms_ << "ms (attempt #"
              << stats_.reconnects.load() << ")..." << std::endl;

    // Sleep for backoff interval
    std::this_thread::sleep_for(
        std::chrono::milliseconds(current_reconnect_interval_ms_)
    );

    // Exponential backoff (double interval, up to max)
    current_reconnect_interval_ms_ = std::min(
        current_reconnect_interval_ms_ * 2,
        config_.max_reconnect_interval_ms
    );
}

void ZMQSubscriber::reset_reconnect_backoff() {
    current_reconnect_interval_ms_ = config_.reconnect_interval_ms;
}

} // namespace firewall
} // namespace mldefender