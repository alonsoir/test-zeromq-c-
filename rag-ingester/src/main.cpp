#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include "common/config_parser.hpp"
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>

namespace {
std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        spdlog::warn("Received signal {}, shutting down gracefully...", signal);
        g_running = false;
    }
}
} // anonymous namespace

int main(int argc, char** argv) {
    // Setup logging
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        "/tmp/rag-ingester.log", 1024 * 1024 * 10, 3  // ← Cambiado a /tmp
    );
    
    std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
    auto logger = std::make_shared<spdlog::logger>("rag-ingester", sinks.begin(), sinks.end());
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");
    
    spdlog::info("====================================");
    spdlog::info("  RAG Ingester Starting");
    spdlog::info("  Version: 0.1.0");
    spdlog::info("====================================");
    
    // Parse command line
    std::string config_path = "/vagrant/rag-ingester/config/rag-ingester.json";
    if (argc > 1) {
        config_path = argv[1];
    }
    
    try {
        // Load configuration
        auto config = rag_ingester::ConfigParser::load(config_path);
        
        // Register signal handlers
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        
        // TODO: Initialize etcd client
        spdlog::info("TODO: Connecting to etcd: {}", 
                    config.service.etcd.endpoints[0]);
        
        // TODO: Register service in etcd
        spdlog::info("TODO: Registering service: {}", config.service.id);
        spdlog::info("  Partner detector: {}", config.service.etcd.partner_detector);
        
        // TODO: Initialize embedders
        spdlog::info("TODO: Initializing embedders...");
        spdlog::info("  Chronos: {}", config.ingester.embedders.chronos.onnx_path);
        spdlog::info("  SBERT: {}", config.ingester.embedders.sbert.onnx_path);
        spdlog::info("  Attack: {}", config.ingester.embedders.attack.onnx_path);
        
        // TODO: Initialize FAISS indices
        spdlog::info("TODO: Initializing FAISS indices (4 total)...");
        
        // TODO: Start file watcher
        spdlog::info("TODO: Watching directory: {}", config.ingester.input.directory);
        
        // Main loop (placeholder)
        spdlog::info("✅ RAG Ingester ready and waiting for events");
        
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            // TODO: Process incoming events
            // TODO: Update etcd heartbeat
        }
        
        spdlog::info("Shutting down gracefully...");
        
        // TODO: Persist FAISS indices
        // TODO: Deregister from etcd
        
        spdlog::info("✅ Shutdown complete");
        
    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return 1;
    }
    
    return 0;
}
