#include "logger.hpp"
#include "config_loader.hpp"
#include "onnx_model.hpp"
#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>

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
        // Banner
        std::cout << R"(
╔════════════════════════════════════════════════════════════╗
║  ML Detector Tricapa - Network Security Analysis           ║
║  Version 1.0.0                                             ║
╚════════════════════════════════════════════════════════════╝
)" << std::endl;

        // Parse arguments
        std::string config_path = "config/ml_detector_config.yaml";
        
        if (argc > 1) {
            std::string arg = argv[1];
            if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "\nOptions:\n"
                          << "  -h, --help     Show this help\n"
                          << "  -v, --version  Show version\n"
                          << "  -c, --config   Config file path\n"
                          << std::endl;
                return 0;
            }
            if (arg == "--version" || arg == "-v") {
                std::cout << "ML Detector Tricapa v1.0.0" << std::endl;
                return 0;
            }
        }

        // Initialize logger
        auto logger = ml_detector::Logger::create(
            "ml-detector",
            "/tmp/ml-detector.log",
            spdlog::level::info
        );
        
        logger->info("🚀 ML Detector Tricapa starting...");
        
        // Load config
        ml_detector::ConfigLoader config_loader(config_path);
        auto config = config_loader.load();
        
        logger->info("✅ Configuration loaded");
        logger->info("   ZMQ Endpoint: {}", config.zmq_endpoint);
        logger->info("   Level 1 Model: {}", config.level1_model_path);
        logger->info("   Threads: {}", config.num_threads);
        
        // Load ONNX model
        logger->info("📦 Loading ONNX model...");
        ml_detector::ONNXModel level1_model(config.level1_model_path);
        logger->info("✅ ONNX model loaded: {} features", level1_model.get_num_features());
        
        // Test inference
        logger->info("🧪 Testing model inference...");
        std::vector<float> test_features(level1_model.get_num_features(), 0.0f);
        auto [label, confidence] = level1_model.predict(test_features);
        logger->info("✅ Test inference OK: label={}, confidence={:.4f}", label, confidence);
        
        // Setup signal handlers
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        
        logger->info("✅ ML Detector ready");
        logger->info("   Press Ctrl+C to stop");
        
        // Main loop (placeholder)
        while (running.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        logger->info("🛑 Shutting down...");
        ml_detector::Logger::shutdown();
        
        std::cout << "✅ ML Detector stopped gracefully" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "❌ Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
