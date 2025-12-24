//===----------------------------------------------------------------------===//
// ML Defender - Firewall ACL Agent
// zmq_subscriber.cpp - ZMQ Subscriber Implementation
// firewall-acl-agent/src/api/zmq_subscriber.cpp
//===----------------------------------------------------------------------===//

#include "firewall/zmq_subscriber.hpp"
#include <iostream>
#include <thread>
#include <stdexcept>
#include <cstring>
#include <lz4.h>           // Day 23
#include <openssl/evp.h>   // Day 23
#include <vector>          // Day 23

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

    // ✅ AÑADIDO: Initialize logger
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

    // ✅ AÑADIDO: Stop logger and print statistics
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

    // ✅ Day 23: Copy to vector for decrypt/decompress
    std::vector<uint8_t> data(data_ptr, data_ptr + data_size);

    try {
        // ✅ Day 23: STEP 1 - Decrypt if enabled
        if (config_.encryption_enabled) {
            auto start = std::chrono::high_resolution_clock::now();

            data = decrypt_chacha20_poly1305(data, config_.crypto_token);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            std::cerr << "[ZMQSubscriber] 🔓 Decrypted: " << duration.count() << " µs" << std::endl;
        }

        // ✅ Day 23: STEP 2 - Decompress if enabled
        if (config_.compression_enabled) {
            auto start = std::chrono::high_resolution_clock::now();

            data = decompress_lz4(data);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            std::cerr << "[ZMQSubscriber] 📦 Decompressed: " << duration.count() << " µs" << std::endl;
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

    // ✅ AÑADIDO: Log the blocked event
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

//==============================================================================
// Day 23: CRYPTO HELPER METHODS
//==============================================================================

std::vector<uint8_t> ZMQSubscriber::decrypt_chacha20_poly1305(
    const std::vector<uint8_t>& encrypted_data,
    const std::string& key_hex
) {
    // Validate input
    if (encrypted_data.size() < 12 + 16) {  // nonce(12) + tag(16)
        throw std::runtime_error("Encrypted data too small");
    }

    // Extract nonce (first 12 bytes)
    std::vector<uint8_t> nonce(encrypted_data.begin(), encrypted_data.begin() + 12);

    // Extract ciphertext + tag
    std::vector<uint8_t> ciphertext_and_tag(encrypted_data.begin() + 12, encrypted_data.end());

    // Convert hex key to bytes
    std::vector<uint8_t> key;
    key.reserve(key_hex.length() / 2);
    for (size_t i = 0; i < key_hex.length(); i += 2) {
        std::string byte_str = key_hex.substr(i, 2);
        key.push_back(static_cast<uint8_t>(std::stoi(byte_str, nullptr, 16)));
    }

    // Prepare output
    std::vector<uint8_t> plaintext(ciphertext_and_tag.size() - 16);

    // Create EVP context
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        throw std::runtime_error("Failed to create EVP context");
    }

    try {
        // Initialize decryption
        if (EVP_DecryptInit_ex(ctx, EVP_chacha20_poly1305(), nullptr,
                              key.data(), nonce.data()) != 1) {
            throw std::runtime_error("EVP_DecryptInit_ex failed");
        }

        // Decrypt
        int len = 0;
        if (EVP_DecryptUpdate(ctx, plaintext.data(), &len,
                             ciphertext_and_tag.data(),
                             ciphertext_and_tag.size() - 16) != 1) {
            throw std::runtime_error("EVP_DecryptUpdate failed");
        }

        // Set expected tag
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_TAG, 16,
                               const_cast<uint8_t*>(ciphertext_and_tag.data() +
                               ciphertext_and_tag.size() - 16)) != 1) {
            throw std::runtime_error("Failed to set auth tag");
        }

        // Finalize (verifies tag)
        int final_len = 0;
        if (EVP_DecryptFinal_ex(ctx, plaintext.data() + len, &final_len) != 1) {
            throw std::runtime_error("Decryption failed - auth tag mismatch");
        }

        plaintext.resize(len + final_len);
        EVP_CIPHER_CTX_free(ctx);

        return plaintext;

    } catch (...) {
        EVP_CIPHER_CTX_free(ctx);
        throw;
    }
}

std::vector<uint8_t> ZMQSubscriber::decompress_lz4(
    const std::vector<uint8_t>& compressed_data
) {
    // Validate input
    if (compressed_data.size() < 4) {
        throw std::runtime_error("Compressed data too small");
    }

    // Extract decompressed size (first 4 bytes, little-endian)
    uint32_t decompressed_size;
    std::memcpy(&decompressed_size, compressed_data.data(), sizeof(uint32_t));

    // Validate size (max 10MB)
    if (decompressed_size == 0 || decompressed_size > 10 * 1024 * 1024) {
        throw std::runtime_error("Invalid decompressed size");
    }

    // Prepare output
    std::vector<uint8_t> decompressed(decompressed_size);

    // Decompress (skip first 4 bytes)
    int result = LZ4_decompress_safe(
        reinterpret_cast<const char*>(compressed_data.data() + 4),
        reinterpret_cast<char*>(decompressed.data()),
        compressed_data.size() - 4,
        decompressed_size
    );

    if (result < 0) {
        throw std::runtime_error("LZ4 decompression failed");
    }

    return decompressed;
}

} // namespace firewall
} // namespace mldefender