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
#include "ml_defender/ransomware_detector.hpp"

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

            model = std::make_shared<ONNXModel>(model_path,config.ml.inference.intra_op_threads);
            
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
        // TODO: IMPLEMENTAR EN SNIFFER
        // ═══════════════════════════════════════════════════════════════════════
        // Level 2 DDoS model espera 8 features, pero sniffer solo captura 6:
        //   ✅ [0-5]: backward_packet_length_max, flow_bytes_per_second, etc.
        //   ❌ [6]: active_mean  - Requiere tracking temporal en user-space
        //   ❌ [7]: idle_mean    - Requiere tracking temporal en user-space
        //
        // ESTADO ACTUAL: Model funciona con 6/8 features (active_mean=0, idle_mean=0)
        // ACCIÓN PENDIENTE: Analizar accuracy del model antes de implementar
        // ═══════════════════════════════════════════════════════════════════════

        std::shared_ptr<ONNXModel> level2_ddos_model;

        if (config.ml.level2.ddos.enabled) {
            log->info("📦 Loading Level 2 DDoS model: {}", config.ml.level2.ddos.model_file);
            log->info("   Name: {}", config.ml.level2.ddos.name);
            log->info("   Type: {}", config.ml.level2.ddos.model_type);
            log->info("   Features: {}", config.ml.level2.ddos.features_count);
            log->info("   Threshold: {}", config.ml.thresholds.level2_ddos);

            std::string model_path = config.ml.models_base_dir + "/" + config.ml.level2.ddos.model_file;
            level2_ddos_model = std::make_shared<ONNXModel>(model_path, config.ml.inference.intra_op_threads);

            // Warmup Level 2 DDoS
            if (config.ml.inference.enable_model_warmup && config.ml.inference.warmup_iterations > 0) {
                log->info("🧪 Warming up Level 2 DDoS model ({} iterations)...",
                           config.ml.inference.warmup_iterations);

                std::vector<float> dummy_features(static_cast<size_t>(config.ml.level2.ddos.features_count), 0.0f);
                for (int i = 0; i < config.ml.inference.warmup_iterations; ++i) {
                    level2_ddos_model->predict(dummy_features);
                }

                log->info("✅ Level 2 DDoS warmup complete");
            }
        } else {
            log->warn("⚠️  Level 2 DDoS model is disabled in configuration");
        }

        // Level 2 Ransomware - Embedded C++20 Detector
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
    log->info("✅ Ransomware detector initialized (test prediction: class={}, prob={:.4f})",
               test_result.class_id, test_result.probability);

} else {
    log->warn("⚠️  Level 2 Ransomware detector is disabled in configuration");
}

if (config.ml.level3.internal.enabled) {
            log->info("📦 Level 3 Internal model: {} (TODO: implement)",
                       config.ml.level3.internal.name);
        }
        
        // 5. Create Feature Extractor
        auto feature_extractor = std::make_shared<FeatureExtractor>();
        log->info("✅ Feature Extractor initialized");
        
        // 6. Create and start ZMQ Handler
        log->info("🔌 Initializing ZMQ Handler...");
        ZMQHandler zmq_handler(config, model, feature_extractor, level2_ddos_model, ransomware_detector);
        
        zmq_handler.start();
        
        log->info("✅ ML Detector initialization complete");
        log->info("   Press Ctrl+C to stop");
        
        // 7. Main loop - wait for shutdown signal
        while (!shutdown_requested.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        // 8. Graceful shutdown
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
