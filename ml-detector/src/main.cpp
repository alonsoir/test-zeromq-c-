#include "logger.hpp"
#include "config_loader.hpp"
#include "onnx_model.hpp"
#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <filesystem>

namespace {
    std::atomic<bool> running{true};
    
    void signal_handler(int signal) {
        if (signal == SIGINT || signal == SIGTERM) {
            running.store(false);
        }
    }
}

int main(int argc, char** argv) {
    try {
        // Parse arguments
        std::string config_path = "/vagrant/ml-detector/config/ml_detector_config.json";
        bool verbose = false;
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "\nOptions:\n"
                          << "  -h, --help            Show this help\n"
                          << "  -v, --version         Show version\n"
                          << "  -c, --config <path>   Config file path\n"
                          << "  --verbose             Show complete configuration\n"
                          << std::endl;
                return 0;
            }
            
            if (arg == "--version" || arg == "-v") {
                std::cout << "ML Detector Tricapa v1.0.0" << std::endl;
                return 0;
            }
            
            if (arg == "--config" || arg == "-c") {
                if (i + 1 < argc) {
                    config_path = argv[++i];
                } else {
                    std::cerr << "❌ --config requires a path" << std::endl;
                    return 1;
                }
            }
            
            if (arg == "--verbose") {
                verbose = true;
            }
        }

        // Banner
        std::cout << R"(
╔════════════════════════════════════════════════════════════╗
║  ML Detector Tricapa - Network Security Analysis           ║
║  Version 1.0.0                                             ║
╚════════════════════════════════════════════════════════════╝
)" << std::endl;

        // Load configuration
        std::cout << "📋 Loading configuration from: " << config_path << std::endl;
        
        ml_detector::ConfigLoader config_loader(config_path);
        auto config = config_loader.load();
        
        std::cout << "✅ Configuration loaded successfully\n" << std::endl;
        
        // Print config (verbose or summary)
        ml_detector::ConfigLoader::print_config(config, verbose);
        
        // Initialize logger based on config
        spdlog::level::level_enum log_level = spdlog::level::info;
        std::string level_str = config.logging.level;
        if (level_str == "TRACE" || level_str == "trace") log_level = spdlog::level::trace;
        else if (level_str == "DEBUG" || level_str == "debug") log_level = spdlog::level::debug;
        else if (level_str == "INFO" || level_str == "info") log_level = spdlog::level::info;
        else if (level_str == "WARN" || level_str == "warn") log_level = spdlog::level::warn;
        else if (level_str == "ERROR" || level_str == "error") log_level = spdlog::level::err;
        
        auto logger = ml_detector::Logger::create(
            "ml-detector",
            config.logging.file,
            log_level
        );
        
        logger->info("🚀 ML Detector Tricapa starting...");
        logger->info("   Component: {} v{}", config.component.name, config.component.version);
        logger->info("   Node: {} @ {}", config.node_id, config.cluster_name);
        logger->info("   Profile: {}", config.active_profile);
        logger->info("   Input: {} {}", config.network.input_socket.socket_type, config.network.input_socket.endpoint);
        logger->info("   Output: {} {}", config.network.output_socket.socket_type, config.network.output_socket.endpoint);
        logger->info("   Threads: {} workers, {} ML inference", 
                     config.threading.worker_threads, config.threading.ml_inference_threads);
        
        // Load ONNX models based on config
        std::unique_ptr<ml_detector::ONNXModel> level1_model;
        
        if (config.ml.level1.enabled) {
            // Construir path completo al modelo
            std::filesystem::path model_path = std::filesystem::path(config.ml.models_base_dir) / config.ml.level1.model_file;
            
            logger->info("📦 Loading Level 1 model: {}", model_path.string());
            logger->info("   Name: {}", config.ml.level1.name);
            logger->info("   Type: {}", config.ml.level1.model_type);
            logger->info("   Features: {}", config.ml.level1.features_count);
            logger->info("   Threshold: {:.2f}", config.ml.thresholds.level1_attack);
            
            try {
                level1_model = std::make_unique<ml_detector::ONNXModel>(
                    model_path.string(),
                    config.ml.inference.intra_op_threads
                );
                
                logger->info("✅ Level 1 model loaded successfully");
                logger->info("   Input: {} ({} features)", 
                            level1_model->get_input_name(), level1_model->get_num_features());
                
                // Verificar que el número de features coincide
                if (level1_model->get_num_features() != static_cast<size_t>(config.ml.level1.features_count)) {
                    logger->warn("⚠️  Feature count mismatch!");
                    logger->warn("   Config expects: {}", config.ml.level1.features_count);
                    logger->warn("   Model provides: {}", level1_model->get_num_features());
                }
                
                // Test inference
                if (config.ml.inference.enable_model_warmup) {
                    logger->info("🧪 Warming up Level 1 model ({} iterations)...", 
                                config.ml.inference.warmup_iterations);
                    
                    std::vector<float> test_features(level1_model->get_num_features(), 0.0f);
                    
                    for (int i = 0; i < config.ml.inference.warmup_iterations; ++i) {
                        auto [label, confidence] = level1_model->predict(test_features);
                        if (i == 0) {
                            logger->debug("   First inference: label={}, confidence={:.4f}", label, confidence);
                        }
                    }
                    
                    logger->info("✅ Model warmup complete");
                }
                
            } catch (const std::exception& e) {
                logger->error("❌ Failed to load Level 1 model: {}", e.what());
                logger->error("   Path: {}", model_path.string());
                logger->error("   Check that the model file exists and is valid ONNX format");
                return 1;
            }
        } else {
            logger->warn("⚠️  Level 1 model is DISABLED in config");
            logger->warn("   ml.level1.enabled = false");
        }
        
        // TODO: Load Level 2 models (DDoS, Ransomware)
        if (config.ml.level2.ddos.enabled) {
            logger->info("📦 Level 2 DDoS model: {} (TODO: implement)", config.ml.level2.ddos.name);
        }
        
        if (config.ml.level2.ransomware.enabled) {
            logger->info("📦 Level 2 Ransomware model: {} (TODO: implement)", config.ml.level2.ransomware.name);
        }
        
        // TODO: Load Level 3 models (Internal, Web)
        if (config.ml.level3.internal.enabled) {
            logger->info("📦 Level 3 Internal model: {} (TODO: implement)", config.ml.level3.internal.name);
        }
        
        // Setup signal handlers
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        
        logger->info("✅ ML Detector initialization complete");
        logger->info("   Press Ctrl+C to stop");
        
        // Main loop (placeholder - será reemplazado por el loop real)
        while (running.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        logger->info("🛑 Shutting down...");
        ml_detector::Logger::shutdown();
        
        std::cout << "✅ ML Detector stopped gracefully" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ FATAL ERROR: " << e.what() << std::endl;
        std::cerr << "\nℹ️  Configuration must be complete and valid (JSON is LA LEY)" << std::endl;
        std::cerr << "   Fix the issue in your config file and restart\n" << std::endl;
        return 1;
    }
}