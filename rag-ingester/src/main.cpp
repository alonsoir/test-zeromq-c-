// main.cpp
// RAG Ingester - Main Entry Point
// Day 38: Integration with etcd-client for encryption consistency

#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <filesystem>
#include <vector>

#include <spdlog/spdlog.h>

#include "common/config_parser.hpp"
#include "file_watcher.hpp"
#include "event_loader.hpp"

// Day 38: etcd-client integration (PRODUCTION CODE)
#include <etcd_client/etcd_client.hpp>
#include <crypto_transport/utils.hpp>

namespace {
    std::atomic<bool> running{true};

    void signal_handler(int signal) {
        if (signal == SIGINT || signal == SIGTERM) {
            spdlog::info("Received signal {}, shutting down gracefully...", signal);
            running = false;
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        // ====================================================================
        // 1. Parse Configuration
        // ====================================================================
        std::string config_path = argc > 1 ? argv[1] : "config/rag-ingester.json";

        spdlog::info("RAG Ingester starting...");
        spdlog::info("Loading configuration from: {}", config_path);

        auto config = rag_ingester::ConfigParser::load(config_path);

        spdlog::info("Configuration loaded:");
        spdlog::info("  Service ID: {}", config.service.id);
        spdlog::info("  Location: {}", config.service.location);
        spdlog::info("  Threading mode: {}", config.ingester.threading.mode);
        spdlog::info("  Input directory: {}", config.ingester.input.directory);
        spdlog::info("  File pattern: {}", config.ingester.input.pattern);
        spdlog::info("  Encrypted: {}", config.ingester.input.encrypted);
        spdlog::info("  Compressed: {}", config.ingester.input.compressed);

        // ====================================================================
        // 2. Initialize etcd-client and get encryption seed (PRODUCTION CODE)
        // ====================================================================
        std::shared_ptr<crypto::CryptoManager> crypto_manager;

        if (config.ingester.input.encrypted) {
            spdlog::info("Initializing etcd-client for encryption...");

            // Initialize etcd-client (same as ml-detector)
            EtcdClient etcd(config.service.etcd.endpoints);

            // Get encryption seed (64 hex chars)
            std::string seed_hex = etcd.get_encryption_seed();
            spdlog::info("Retrieved encryption seed from etcd ({} chars)", seed_hex.size());

            // Convert hex to bytes
            auto key_bytes = crypto_transport::hex_to_bytes(seed_hex);
            if (key_bytes.size() != 32) {
                throw std::runtime_error("Invalid encryption seed size: " +
                                        std::to_string(key_bytes.size()) + " (expected 32)");
            }

            // Create encryption seed string
            std::string encryption_seed(key_bytes.begin(), key_bytes.end());

            // Create CryptoManager (same as ml-detector)
            crypto_manager = std::make_shared<crypto::CryptoManager>(encryption_seed);
            spdlog::info("✅ CryptoManager initialized (ChaCha20-Poly1305 + LZ4)");

        } else {
            spdlog::warn("⚠️  Encryption DISABLED - data in plaintext");
            crypto_manager = nullptr;
        }

        // ====================================================================
        // 3. Initialize EventLoader (UPDATED: uses CryptoManager)
        // ====================================================================
        spdlog::info("Initializing EventLoader...");
        rag_ingester::EventLoader loader(crypto_manager);

        // ====================================================================
        // 4. Initialize FileWatcher
        // ====================================================================
        spdlog::info("Initializing FileWatcher...");
        spdlog::info("  Watching: {}", config.ingester.input.directory);
        spdlog::info("  Pattern: {}", config.ingester.input.pattern);

        rag_ingester::FileWatcher watcher(
            config.ingester.input.directory,
            config.ingester.input.pattern
        );

        // ====================================================================
        // 5. Set up signal handlers
        // ====================================================================
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        // ====================================================================
        // 6. Event processing callback
        // ====================================================================
        uint64_t events_processed = 0;
        uint64_t events_failed = 0;

        auto callback = [&](const std::string& filepath) {
            spdlog::info("New file detected: {}", filepath);

            try {
                auto event = loader.load(filepath);
                events_processed++;

                spdlog::info("Event loaded: id={}, features={}, class={}, confidence={:.4f}",
                    event.event_id,
                    event.features.size(),
                    event.final_class,
                    event.confidence
                );

                // TODO Day 39+: Send to embedders and indexers

                // Delete file if configured
                if (config.ingester.input.delete_after_process) {
                    std::filesystem::remove(filepath);
                    spdlog::debug("Deleted processed file: {}", filepath);
                }

            } catch (const std::exception& e) {
                events_failed++;
                spdlog::error("Failed to process {}: {}", filepath, e.what());
            }
        };

        // ====================================================================
        // 7. Start watching
        // ====================================================================
        watcher.start(callback);
        spdlog::info("✅ RAG Ingester ready and waiting for events");

        // ====================================================================
        // 8. Main loop with statistics
        // ====================================================================
        auto last_stats = std::chrono::steady_clock::now();

        while (running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));

            // Print statistics every 60 seconds
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_stats);

            if (elapsed.count() >= 60) {
                auto stats = loader.get_stats();

                spdlog::info("=== Statistics (last 60s) ===");
                spdlog::info("  Events processed: {}", events_processed);
                spdlog::info("  Events failed: {}", events_failed);
                spdlog::info("  Total loaded: {}", stats.total_loaded);
                spdlog::info("  Total failed: {}", stats.total_failed);
                spdlog::info("  Bytes processed: {}", stats.bytes_processed);
                spdlog::info("  Partial features: {}", stats.partial_feature_count);

                last_stats = now;
            }
        }

        // ====================================================================
        // 9. Graceful shutdown
        // ====================================================================
        spdlog::info("Shutting down...");
        watcher.stop();

        auto final_stats = loader.get_stats();
        spdlog::info("=== Final Statistics ===");
        spdlog::info("  Total events: {}", events_processed);
        spdlog::info("  Failed events: {}", events_failed);
        spdlog::info("  Bytes processed: {}", final_stats.bytes_processed);

        spdlog::info("RAG Ingester stopped gracefully");
        return 0;

    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return 1;
    }
}