#pragma once

#include <zmq.hpp>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <spdlog/spdlog.h>

#include "feature_extractor.hpp"

// ✅ INCLUDES CORRECTOS (ml_defender/ no ml/)
#include "ml_defender/ransomware_detector.hpp"
#include "config_loader.hpp"
#include "onnx_model.hpp"
#include "ml_defender/ddos_detector.hpp"
#include "ml_defender/traffic_detector.hpp"
#include "ml_defender/internal_detector.hpp"
#include "config_loader.hpp"

#include "../protobuf/network_security.pb.h"

namespace ml_detector {

class ZMQHandler {
public:
    struct Stats {
        uint64_t events_received = 0;
        uint64_t events_processed = 0;
        uint64_t events_sent = 0;
        uint64_t attacks_detected = 0;
        uint64_t deserialization_errors = 0;
        uint64_t feature_extraction_errors = 0;
        uint64_t inference_errors = 0;
        double avg_processing_time_ms = 0.0;
    };

    ZMQHandler(
        const DetectorConfig& config,
        std::shared_ptr<ONNXModel> level1_model,
        std::shared_ptr<FeatureExtractor> extractor,
        std::shared_ptr<ml_defender::DDoSDetector> ddos_detector,           // ✨ NUEVO
        std::shared_ptr<ml_defender::RansomwareDetector> ransomware_detector,
        std::shared_ptr<ml_defender::TrafficDetector> traffic_detector,     // ✨ NUEVO
        std::shared_ptr<ml_defender::InternalDetector> internal_detector    // ✨ NUEVO
    );

    ~ZMQHandler();

    void start();
    void stop();

    Stats get_stats() const;
    void reset_stats();

private:
    void run();
    void process_event(const std::string& message);
    void send_enriched_event(const protobuf::NetworkSecurityEvent& event);

    // Config & Models
    DetectorConfig config_;
    std::shared_ptr<ONNXModel> level1_model_;
    std::shared_ptr<ml_defender::DDoSDetector> ddos_detector_;              // ✨ NUEVO
    std::shared_ptr<ml_defender::RansomwareDetector> ransomware_detector_;
    std::shared_ptr<ml_defender::TrafficDetector> traffic_detector_;        // ✨ NUEVO
    std::shared_ptr<ml_defender::InternalDetector> internal_detector_;      // ✨ NUEVO
    std::shared_ptr<FeatureExtractor> extractor_;

    // ZMQ
    zmq::context_t context_;
    std::unique_ptr<zmq::socket_t> input_socket_;
    std::unique_ptr<zmq::socket_t> output_socket_;

    // Threading
    std::atomic<bool> running_;
    std::unique_ptr<std::thread> worker_thread_;

    // Stats
    mutable std::mutex stats_mutex_;
    Stats stats_;
    std::chrono::steady_clock::time_point last_stats_report_;

    // Logging
    std::shared_ptr<spdlog::logger> logger_;
};

} // namespace ml_detector