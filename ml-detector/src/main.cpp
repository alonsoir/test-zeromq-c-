#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include "config_loader.hpp"
#include "logger.hpp"
#include "onnx_model.hpp"
#include "feature_extractor.hpp"
#include "zmq_handler.hpp"
#include "etcd_client.hpp"
#include "ml_defender/ransomware_detector.hpp"
#include "ml_defender/ddos_detector.hpp"
#include "ml_defender/traffic_detector.hpp"
#include "ml_defender/internal_detector.hpp"

// 🎯 DAY 27: Crypto-Transport Integration
#include <crypto_transport/crypto_manager.hpp>
#include <crypto_transport/utils.hpp>  // 🎯 DAY 29: For hex_to_bytes

//ml-detector/src/main.cpp
using namespace ml_detector;

// Signal handling para shutdown graceful
static std::atomic<bool> shutdown_requested(false);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\n🛑 Shutdown signal received..." << std::endl;
        shutdown_requested.store(true);
    }
}

void print_banner() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  ML Detector Tricapa - Network Security Analysis           ║\n";
    std::cout << "║  Version 1.0.0                                             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "OPTIONS:\n";
    std::cout << "  --config <path>    Path to configuration file (default: ../config/ml_detector_config.json)\n";
    std::cout << "  --verbose          Enable verbose logging (DEBUG level)\n";
    std::cout << "  --help             Show this help message\n";
    std::cout << "  --version          Show version information\n";
    std::cout << "\n";
}

// Helper: Convertir string a spdlog level
spdlog::level::level_enum string_to_log_level(const std::string& level_str) {
    if (level_str == "trace") return spdlog::level::trace;
    if (level_str == "debug") return spdlog::level::debug;
    if (level_str == "info") return spdlog::level::info;
    if (level_str == "warn" || level_str == "warning") return spdlog::level::warn;
    if (level_str == "error" || level_str == "err") return spdlog::level::err;
    if (level_str == "critical") return spdlog::level::critical;
    if (level_str == "off") return spdlog::level::off;
    return spdlog::level::info; // default
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string config_path = "../config/ml_detector_config.json";
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_banner();
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--version" || arg == "-v") {
            print_banner();
            std::cout << "ML Detector Tricapa v1.0.0\n";
            std::cout << "Built: " << __DATE__ << " " << __TIME__ << "\n";
            return 0;
        }
        else if (arg == "--config" || arg == "-c") {
            if (i + 1 < argc) {
                config_path = argv[++i];
            } else {
                std::cerr << "Error: --config requires a path argument\n";
                return 1;
            }
        }
        else if (arg == "--verbose") {
            verbose = true;
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    print_banner();

    // Setup signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        // 1. Load configuration
        std::cout << "📋 Loading configuration from: " << config_path << std::endl;

        ConfigLoader loader(config_path);
        DetectorConfig config = loader.load();

        std::cout << "✅ Configuration loaded successfully\n\n" << std::endl;

        // ═══════════════════════════════════════════════════════════════════════
        // ETCD INTEGRATION - Register component and upload config
        // ═══════════════════════════════════════════════════════════════════════
        std::unique_ptr<ml_detector::EtcdClient> etcd_client;

        if (config.etcd.enabled) {
            std::string etcd_endpoint = config.etcd.endpoints[0];

            std::cout << "🔗 [etcd] Initializing connection to " << etcd_endpoint << std::endl;

            etcd_client = std::make_unique<ml_detector::EtcdClient>(etcd_endpoint, "ml-detector");

            if (!etcd_client->initialize()) {
                std::cerr << "❌ [etcd] Failed to initialize - REQUIRED for ml-detector" << std::endl;
                return 1;
            }

            if (!etcd_client->registerService()) {
                std::cerr << "❌ [etcd] Failed to register service - REQUIRED for ml-detector" << std::endl;
                return 1;
            }

            std::cout << "✅ [etcd] ml-detector registered and config uploaded" << std::endl;

        } else {
            std::cerr << "❌ [etcd] etcd integration is REQUIRED for ml-detector" << std::endl;
            std::cerr << "   Enable etcd in config: ml_detector_config.json" << std::endl;
            return 1;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 🎯 DAY 29: CRYPTO-TRANSPORT INITIALIZATION WITH HEX→BINARY CONVERSION
        // ═══════════════════════════════════════════════════════════════════════
        std::cout << "\n🔐 [crypto] Initializing Crypto-Transport..." << std::endl;

        // Get encryption seed from etcd (HEX format)
        std::string encryption_seed_hex = etcd_client->get_encryption_seed();

        if (encryption_seed_hex.empty()) {
            std::cerr << "❌ [crypto] Failed to get encryption seed from etcd" << std::endl;
            return 1;
        }

        std::cout << "🔑 [ml-detector] Retrieved encryption seed (" << encryption_seed_hex.size() << " hex chars)" << std::endl;

        // Convert HEX to binary (64 hex chars → 32 bytes)
        std::string encryption_seed;
        try {
            auto key_bytes = crypto_transport::hex_to_bytes(encryption_seed_hex);
            encryption_seed = std::string(key_bytes.begin(), key_bytes.end());
        } catch (const std::exception& e) {
            std::cerr << "❌ [crypto] Failed to convert hex seed: " << e.what() << std::endl;
            return 1;
        }

        if (encryption_seed.size() != 32) {
            std::cerr << "❌ [crypto] Invalid key size: " << encryption_seed.size() << " bytes (expected 32)" << std::endl;
            return 1;
        }

        std::cout << "✅ [crypto] Encryption key converted: 32 bytes" << std::endl;

        // Create CryptoManager
        auto crypto_manager = std::make_shared<crypto::CryptoManager>(encryption_seed);

        std::cout << "✅ [crypto] CryptoManager initialized (ChaCha20-Poly1305 + LZ4)" << std::endl;
        std::cout << std::endl;

        // Print configuration summary
        std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  ML DETECTOR TRICAPA - CONFIGURATION                          ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";

        std::cout << "📦 Component: " << config.component.name << " v" << config.component.version << "\n";
        std::cout << "   Mode: " << config.component.mode << "\n";
        std::cout << "   Node: " << config.node_id << " @ " << config.cluster_name << "\n";
        std::cout << "   Profile: " << config.active_profile << "\n\n";

        std::cout << "🔌 Network:\n";
        std::cout << "   Input:  " << config.network.input_socket.socket_type
                  << " " << config.network.input_socket.mode
                  << " " << config.network.input_socket.endpoint << "\n";
        std::cout << "   Output: " << config.network.output_socket.socket_type
                  << " " << config.network.output_socket.mode
                  << " " << config.network.output_socket.endpoint << "\n\n";

        std::cout << "🧵 Threading:\n";
        std::cout << "   Workers: " << config.threading.worker_threads << "\n";
        std::cout << "   ML Inference: " << config.threading.ml_inference_threads << " threads\n";
        std::cout << "   Feature Extraction: " << config.threading.feature_extractor_threads << " threads\n";
        std::cout << "   CPU Affinity: " << (config.threading.cpu_affinity.enabled ? "✅ enabled" : "❌ disabled") << "\n\n";

        std::cout << "🤖 ML Models:\n";
        std::cout << "   Base Dir: " << config.ml.models_base_dir << "\n";
        std::cout << "   Level 1: " << (config.ml.level1.enabled ? "✅" : "❌")
                  << " " << config.ml.level1.name
                  << " (" << config.ml.level1.features_count << " features)\n";
        std::cout << "   Level 2 DDoS: " << (config.ml.level2.ddos.enabled ? "✅" : "❌")
                  << " " << config.ml.level2.ddos.name
                  << " (" << config.ml.level2.ddos.features_count << " features)\n";
        std::cout << "   Level 2 Ransomware: " << (config.ml.level2.ransomware.enabled ? "✅" : "❌")
                  << " " << config.ml.level2.ransomware.name
                  << " (" << config.ml.level2.ransomware.features_count << " features)\n";
        std::cout << "   Level 3 Internal: " << (config.ml.level3.internal.enabled ? "✅" : "❌")
                  << " " << config.ml.level3.internal.name
                  << " (" << config.ml.level3.internal.features_count << " features)\n";
        std::cout << "   Level 3 Web: " << (config.ml.level3.web.enabled ? "✅" : "❌")
                  << " " << config.ml.level3.web.name
                  << " (" << config.ml.level3.web.features_count << " features)\n\n";

        std::cout << "📝 Logging: " << config.logging.level
                  << " → " << config.logging.file << "\n";
        std::cout << "📊 Monitoring: Stats every " << config.monitoring.stats_interval_seconds << "s\n";
        std::cout << "🔒 Transport: Compression=" << (config.transport.compression.enabled ? "✅" : "❌")
                  << ", Encryption=" << (config.transport.encryption.enabled ? "✅" : "❌") << "\n\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n\n";

        // 2. Initialize logger
        auto log_level = verbose ? spdlog::level::debug : string_to_log_level(config.logging.level);
        auto log = Logger::create("ml-detector", config.logging.file, log_level);

        if (!log) {
            throw std::runtime_error("Failed to create logger");
        }

        log->info("🚀 ML Detector Tricapa starting...");
        log->info("   Component: {} v{}", config.component.name, config.component.version);
        log->info("   Node: {} @ {}", config.node_id, config.cluster_name);
        log->info("   Profile: {}", config.active_profile);
        log->info("   Input: {} {}", config.network.input_socket.socket_type,
                    config.network.input_socket.endpoint);
        log->info("   Output: {} {}", config.network.output_socket.socket_type,
                    config.network.output_socket.endpoint);
        log->info("   Threads: {} workers, {} ML inference",
                    config.threading.worker_threads,
                    config.threading.ml_inference_threads);
        log->info("   Crypto: ChaCha20-Poly1305 + LZ4 enabled");

        // 3. Load ML Model (Level 1)
        std::shared_ptr<ONNXModel> model;

        if (config.ml.level1.enabled) {
            log->info("📦 Loading Level 1 model: {}", config.ml.level1.model_file);
            log->info("   Name: {}", config.ml.level1.name);
            log->info("   Type: {}", config.ml.level1.model_type);
            log->info("   Features: {}", config.ml.level1.features_count);
            log->info("   Threshold: {}", config.ml.thresholds.level1_attack);

            // Construir path completo
            std::string model_path = config.ml.models_base_dir + "/" + config.ml.level1.model_file;

            model = std::make_shared<ONNXModel>(model_path, config.ml.inference.intra_op_threads);

            // Warmup (simple test inference)
            if (config.ml.inference.enable_model_warmup && config.ml.inference.warmup_iterations > 0) {
                log->info("🧪 Warming up Level 1 model ({} iterations)...",
                           config.ml.inference.warmup_iterations);

                std::vector<float> dummy_features(static_cast<size_t>(config.ml.level1.features_count), 0.0f);
                for (int i = 0; i < config.ml.inference.warmup_iterations; ++i) {
                    model->predict(dummy_features);
                }

                log->info("✅ Model warmup complete");
            }
        } else {
            log->error("❌ Level 1 model is not enabled in configuration");
            return 1;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 4. Load Level 2 Models
        // ═══════════════════════════════════════════════════════════════════════

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // ✨ LEVEL 2: DDoS Detector (C++20 Embedded)
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        std::shared_ptr<ml_defender::DDoSDetector> ddos_detector;

        if (config.ml.level2.ddos.enabled) {
            log->info("📦 Loading Level 2 DDoS Detector (Embedded C++20)");
            log->info("   Name: {}", config.ml.level2.ddos.name);
            log->info("   Type: RandomForest-Embedded (100 trees, 612 nodes)");
            log->info("   Features: {} (normalized 0.0-1.0)", config.ml.level2.ddos.features_count);
            log->info("   Threshold: {}", config.ml.thresholds.level2_ddos);
            log->info("   Implementation: Native C++20 (no ONNX)");

            ddos_detector = std::make_shared<ml_defender::DDoSDetector>();

            // Test de inicialización
            ml_defender::DDoSDetector::Features test_features{
                .syn_ack_ratio = 0.5f,
                .packet_symmetry = 0.5f,
                .source_ip_dispersion = 0.5f,
                .protocol_anomaly_score = 0.5f,
                .packet_size_entropy = 0.5f,
                .traffic_amplification_factor = 0.5f,
                .flow_completion_rate = 0.5f,
                .geographical_concentration = 0.5f,
                .traffic_escalation_rate = 0.5f,
                .resource_saturation_score = 0.5f
            };

            auto test_result = ddos_detector->predict(test_features);
            log->info("✅ DDoS detector initialized (test: class={}, prob={:.4f})",
                       test_result.class_id, test_result.probability);

        } else {
            log->warn("⚠️  Level 2 DDoS detector is disabled in configuration");
        }

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // Level 2 Ransomware - Embedded C++20 Detector
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        std::shared_ptr<ml_defender::RansomwareDetector> ransomware_detector;

        if (config.ml.level2.ransomware.enabled) {
            log->info("📦 Loading Level 2 Ransomware Detector (Embedded C++20)");
            log->info("   Name: {}", config.ml.level2.ransomware.name);
            log->info("   Type: RandomForest-Embedded (100 trees, 3764 nodes)");
            log->info("   Features: {}", config.ml.level2.ransomware.features_count);
            log->info("   Threshold: {}", config.ml.thresholds.level2_ransomware);
            log->info("   Implementation: Native C++20 (no ONNX)");

            ransomware_detector = std::make_shared<ml_defender::RansomwareDetector>();

            // Test de inicialización
            ml_defender::RansomwareDetector::Features test_features{
                .io_intensity = 0.5f,
                .entropy = 0.5f,
                .resource_usage = 0.5f,
                .network_activity = 0.5f,
                .file_operations = 0.5f,
                .process_anomaly = 0.5f,
                .temporal_pattern = 0.5f,
                .access_frequency = 0.5f,
                .data_volume = 0.5f,
                .behavior_consistency = 0.5f
            };

            auto test_result = ransomware_detector->predict(test_features);
            log->info("✅ Ransomware detector initialized (test: class={}, prob={:.4f})",
                       test_result.class_id, test_result.probability);

        } else {
            log->warn("⚠️  Level 2 Ransomware detector is disabled in configuration");
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 5. Load Level 3 Models
        // ═══════════════════════════════════════════════════════════════════════

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // ✨ LEVEL 3: Traffic Detector (C++20 Embedded)
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        std::shared_ptr<ml_defender::TrafficDetector> traffic_detector;

        if (config.ml.level3.web.enabled) {  // Nota: config usa "web" para traffic
            log->info("📦 Loading Level 3 Traffic Detector (Embedded C++20)");
            log->info("   Name: {} (Internet vs Internal)", config.ml.level3.web.name);
            log->info("   Type: RandomForest-Embedded (100 trees, 1,014 nodes)");
            log->info("   Features: {} (normalized 0.0-1.0)", config.ml.level3.web.features_count);
            log->info("   Threshold: {}", config.ml.thresholds.level2_ddos);
            log->info("   Implementation: Native C++20 (no ONNX)");

            traffic_detector = std::make_shared<ml_defender::TrafficDetector>();

            // Test de inicialización
            ml_defender::TrafficDetector::Features test_features{
                .packet_rate = 0.5f,
                .connection_rate = 0.5f,
                .tcp_udp_ratio = 0.5f,
                .avg_packet_size = 0.5f,
                .port_entropy = 0.5f,
                .flow_duration_std = 0.5f,
                .src_ip_entropy = 0.5f,
                .dst_ip_concentration = 0.5f,
                .protocol_variety = 0.5f,
                .temporal_consistency = 0.5f
            };

            auto test_result = traffic_detector->predict(test_features);
            log->info("✅ Traffic detector initialized (test: class={}, prob={:.4f})",
                       test_result.class_id, test_result.probability);

        } else {
            log->warn("⚠️  Level 3 Traffic detector is disabled in configuration");
        }

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // ✨ LEVEL 3: Internal Detector (C++20 Embedded)
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        std::shared_ptr<ml_defender::InternalDetector> internal_detector;

        if (config.ml.level3.internal.enabled) {
            log->info("📦 Loading Level 3 Internal Detector (Embedded C++20)");
            log->info("   Name: {} (Benign vs Suspicious)", config.ml.level3.internal.name);
            log->info("   Type: RandomForest-Embedded (100 trees, 940 nodes)");
            log->info("   Features: {} (normalized 0.0-1.0)", config.ml.level3.internal.features_count);
            log->info("   Threshold: {}", config.ml.thresholds.level3_web);
            log->info("   Implementation: Native C++20 (no ONNX)");
            log->info("   Detection: Lateral Movement, Data Exfiltration");

            internal_detector = std::make_shared<ml_defender::InternalDetector>();

            // Test de inicialización
            ml_defender::InternalDetector::Features test_features{
                .internal_connection_rate = 0.5f,
                .service_port_consistency = 0.5f,
                .protocol_regularity = 0.5f,
                .packet_size_consistency = 0.5f,
                .connection_duration_std = 0.5f,
                .lateral_movement_score = 0.5f,
                .service_discovery_patterns = 0.5f,
                .data_exfiltration_indicators = 0.5f,
                .temporal_anomaly_score = 0.5f,
                .access_pattern_entropy = 0.5f
            };

            auto test_result = internal_detector->predict(test_features);
            log->info("✅ Internal detector initialized (test: class={}, prob={:.4f})",
                       test_result.class_id, test_result.probability);

        } else {
            log->warn("⚠️  Level 3 Internal detector is disabled in configuration");
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 6. Create Feature Extractor
        // ═══════════════════════════════════════════════════════════════════════
        auto feature_extractor = std::make_shared<FeatureExtractor>();
        log->info("✅ Feature Extractor initialized");

        // ═══════════════════════════════════════════════════════════════════════
        // 7. Create and start ZMQ Handler (with crypto_manager)
        // ═══════════════════════════════════════════════════════════════════════
        log->info("🔌 Initializing ZMQ Handler...");
        log->info("   Detectors loaded:");
        log->info("     Level 1: General Attack (ONNX)");
        if (ddos_detector) {
            log->info("     Level 2: DDoS (C++20 - {} trees)", ddos_detector->num_trees());
        }
        if (ransomware_detector) {
            log->info("     Level 2: Ransomware (C++20 - {} trees)", ransomware_detector->num_trees());
        }
        if (traffic_detector) {
            log->info("     Level 3: Traffic (C++20 - {} trees)", traffic_detector->num_trees());
        }
        if (internal_detector) {
            log->info("     Level 3: Internal (C++20 - {} trees)", internal_detector->num_trees());
        }

        // ✨ DAY 27: PASAR crypto_manager AL ZMQHANDLER
        ZMQHandler zmq_handler(
            config,
            model,
            feature_extractor,
            ddos_detector,
            ransomware_detector,
            traffic_detector,
            internal_detector,
            crypto_manager  // 🎯 DAY 27: NEW PARAMETER
        );

        zmq_handler.start();

        log->info("✅ ML Detector initialization complete");
        log->info("   Press Ctrl+C to stop");

        // ═══════════════════════════════════════════════════════════════════════
        // 8. Main loop - wait for shutdown signal
        // ═══════════════════════════════════════════════════════════════════════
        while (!shutdown_requested.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 9. Graceful shutdown
        // ═══════════════════════════════════════════════════════════════════════
        log->info("🛑 Shutting down ML Detector...");

        zmq_handler.stop();

        // Print final stats
        auto stats = zmq_handler.get_stats();
        log->info("📊 Final Statistics:");
        log->info("   Events Received: {}", stats.events_received);
        log->info("   Events Processed: {}", stats.events_processed);
        log->info("   Events Sent: {}", stats.events_sent);
        log->info("   Attacks Detected: {}", stats.attacks_detected);
        log->info("   Errors: deser={}, feat={}, inf={}",
                    stats.deserialization_errors,
                    stats.feature_extraction_errors,
                    stats.inference_errors);
        log->info("   Avg Processing Time: {:.2f}ms", stats.avg_processing_time_ms);

        log->info("✅ ML Detector stopped gracefully");

        Logger::shutdown();

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "❌ Fatal error: " << e.what() << std::endl;
        return 1;
    }
}