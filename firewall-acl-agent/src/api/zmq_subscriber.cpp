//===----------------------------------------------------------------------===//
// ML Defender - Firewall ACL Agent
// zmq_subscriber.cpp - ZMQ Subscriber Implementation
//===----------------------------------------------------------------------===//

#include "firewall/zmq_subscriber.hpp"
#include <iostream>
#include <thread>
#include <stdexcept>
#include <cstring>

namespace mldefender {
namespace firewall {

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

    std::cerr << "[ZMQSubscriber] Initialized with endpoint: "
              << config_.endpoint << std::endl;
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
                      << topic_len << " bytes), protobuf data: " 
                      << data_size << " bytes" << std::endl;
        }
    }
    
    // Parse NetworkSecurityEvent protobuf
    protobuf::NetworkSecurityEvent event;

    try {
        if (!event.ParseFromArray(data_ptr, static_cast<int>(data_size))) {
            std::cerr << "[ZMQSubscriber] Failed to parse NetworkSecurityEvent protobuf ("
                      << data_size << " bytes)" << std::endl;
            stats_.parse_errors++;
            return;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ZMQSubscriber] Exception parsing protobuf: "
                  << e.what() << std::endl;
        stats_.parse_errors++;
        return;
    }

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

    // Create detection
    std::vector<protobuf::Detection> detections;
    protobuf::Detection detection;
    
    detection.set_src_ip(nf.source_ip());
    
    // Set detection type based on threat category
    std::string threat_cat = event.threat_category();
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

    try {
        processor_.add_detections(detections);
        stats_.detections_processed += detections.size();
        
        std::cerr << "[ZMQSubscriber] Processed attack from " << nf.source_ip() 
                  << " (type: " << threat_cat << ", conf: " 
                  << ml.level1_confidence() << ")" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[ZMQSubscriber] Error forwarding detections to processor: "
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