// sniffer/include/ransomware_feature_processor.hpp
#pragma once

#include "ransomware_feature_extractor.hpp"
#include "flow_tracker.hpp"
#include "dns_analyzer.hpp"
#include "ip_whitelist.hpp"
#include "main.h"  // ⭐ AÑADIR ESTO
#include "network_security.pb.h"
#include <thread>
#include <atomic>
#include <memory>

namespace sniffer {

    class RansomwareFeatureProcessor {
    public:
        explicit RansomwareFeatureProcessor();
        ~RansomwareFeatureProcessor();

        // Lifecycle
        bool initialize();
        bool start();
        void stop();

        // ⭐ CAMBIADO: packet_event → SimpleEvent
        void process_packet(const SimpleEvent& event);

        // Get extracted features (for ZMQ sender)
        bool get_features_if_ready(protobuf::RansomwareFeatures& features);

    private:
        // Components
        std::unique_ptr<FlowTracker> flow_tracker_;
        std::unique_ptr<DNSAnalyzer> dns_analyzer_;
        std::unique_ptr<IPWhitelist> ip_whitelist_;
        std::unique_ptr<RansomwareFeatureExtractor> extractor_;

        // Timer thread for periodic extraction
        std::thread extraction_thread_;
        std::atomic<bool> running_{false};

        // Config
        uint32_t extraction_interval_seconds_ = 30;

        // Internal state
        std::mutex features_mutex_;
        protobuf::RansomwareFeatures latest_features_;
        bool features_ready_ = false;

        // Thread function
        void extraction_loop();

        // Helpers
        void parse_and_feed_packet(const SimpleEvent& event);  // ⭐ CAMBIADO
        bool is_dns_packet(const SimpleEvent& event);          // ⭐ CAMBIADO
        void extract_and_store_features();
    };

} // namespace sniffer