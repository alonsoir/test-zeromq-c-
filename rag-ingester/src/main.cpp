// main.cpp
// RAG Ingester - Main Entry Point
// Day 40: Producer complete - metadata_db + FAISS persistence

#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <filesystem>
#include <vector>

#include <spdlog/spdlog.h>
#include "metadata_db.hpp"
#include <faiss/index_io.h>
#include <faiss/IndexFlat.h>
#include "common/config_parser.hpp"
#include "file_watcher.hpp"
#include "event_loader.hpp"
#include "embedders/simple_embedder.hpp"
#include "indexers/multi_index_manager.hpp"

// Day 38: etcd-client integration (PRODUCTION CODE)
#include <etcd_client/etcd_client.hpp>
#include <crypto_transport/utils.hpp>

namespace {
    std::atomic<bool> running{true};

    // Global for save_indices_to_disk()
    rag_ingester::MultiIndexManager* g_index_manager = nullptr;
    rag::MetadataDB* g_metadata_db = nullptr;

    void signal_handler(int signal) {
        if (signal == SIGINT || signal == SIGTERM) {
            spdlog::info("Received signal {}, shutting down gracefully...", signal);
            running = false;
        }
    }
}

// ============================================================================
// FUNCTION: Save indices to disk
// ============================================================================
void save_indices_to_disk() {
    if (!g_index_manager) {
        spdlog::warn("Cannot save indices: index_manager not initialized");
        return;
    }

    std::string base_path = "/vagrant/shared/indices/";

    spdlog::info("💾 Saving FAISS indices to disk...");

    try {
        // Create directory if it doesn't exist
        std::filesystem::create_directories(base_path);

        // Get raw FAISS indices from MultiIndexManager
        auto& chronos_index = g_index_manager->get_chronos_index();
        auto& sbert_index = g_index_manager->get_sbert_index();
        auto& attack_index = g_index_manager->get_entity_malicious_index();

        // Write indices to disk
        faiss::write_index(&chronos_index, (base_path + "chronos.faiss").c_str());
        faiss::write_index(&sbert_index, (base_path + "sbert.faiss").c_str());
        faiss::write_index(&attack_index, (base_path + "attack.faiss").c_str());

        // Flush metadata
        if (g_metadata_db) {
            g_metadata_db->flush();
        }

        spdlog::info("✅ FAISS indices saved:");
        spdlog::info("   Chronos: {} vectors", chronos_index.ntotal);
        spdlog::info("   SBERT:   {} vectors", sbert_index.ntotal);
        spdlog::info("   Attack:  {} vectors", attack_index.ntotal);

    } catch (const std::exception& e) {
        spdlog::error("❌ Failed to save indices: {}", e.what());
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
        // 1.1 Initialize metadata_db
        // ====================================================================
        std::unique_ptr<rag::MetadataDB> metadata_db;

        // ====================================================================
        // 2. Initialize etcd-client and get encryption seed (PRODUCTION CODE)
        // ====================================================================
        std::shared_ptr<crypto::CryptoManager> crypto_manager;

        if (config.ingester.input.encrypted) {
            spdlog::info("Initializing etcd-client for encryption...");

            // Parse endpoint (host:port)
            std::string endpoint = config.service.etcd.endpoints[0];
            size_t colon_pos = endpoint.find(':');
            std::string host = endpoint.substr(0, colon_pos);
            int port = std::stoi(endpoint.substr(colon_pos + 1));

            spdlog::info("🔗 [etcd] Connecting to: {}:{}", host, port);

            // Build etcd-client Config
            etcd_client::Config etcd_config;
            etcd_config.host = host;
            etcd_config.port = port;
            etcd_config.timeout_seconds = 5;
            etcd_config.component_name = config.service.id;
            etcd_config.encryption_enabled = true;
            etcd_config.heartbeat_enabled = true;

            // Initialize etcd-client
            etcd_client::EtcdClient etcd(etcd_config);

            // Connect and register
            if (!etcd.connect()) {
                throw std::runtime_error("[etcd] Failed to connect to etcd-server");
            }

            if (!etcd.register_component()) {
                throw std::runtime_error("[etcd] Failed to register component");
            }

            // Get encryption key (NOT seed)
            std::string seed_hex = etcd.get_encryption_key();
            spdlog::info("Retrieved encryption key from etcd ({} chars)", seed_hex.size());

            // Convert hex to bytes
            auto key_bytes = crypto_transport::hex_to_bytes(seed_hex);
            if (key_bytes.size() != 32) {
                throw std::runtime_error("Invalid encryption key size: " +
                                        std::to_string(key_bytes.size()) + " (expected 32)");
            }

            // Create encryption seed string
            std::string encryption_seed(key_bytes.begin(), key_bytes.end());

            // Create CryptoManager
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
        // 3.5. Initialize SimpleEmbedder and MultiIndexManager (Day 38.5)
        // ====================================================================
        spdlog::info("Initializing SimpleEmbedder...");
        rag_ingester::SimpleEmbedder embedder;

        spdlog::info("Initializing MultiIndexManager...");
        rag_ingester::MultiIndexManager index_manager;

        // Set global pointer for save_indices_to_disk()
        g_index_manager = &index_manager;

        // ====================================================================
        // 3.6. Initialize MetadataDB (Day 40)
        // ====================================================================
        try {
            // Create shared indices directory
            std::filesystem::create_directories("/vagrant/shared/indices/");

            std::string db_path = "/vagrant/shared/indices/metadata.db";
            metadata_db = std::make_unique<rag::MetadataDB>(db_path);
            g_metadata_db = metadata_db.get();

            spdlog::info("✅ Metadata DB initialized: {}", db_path);
            spdlog::info("   Existing events: {}", metadata_db->count());

        } catch (const std::exception& e) {
            spdlog::error("❌ Failed to init metadata DB: {}", e.what());
            return 1;
        }

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
        // 6. Event processing callback (Day 40: + metadata_db)
        // ====================================================================
        uint64_t events_processed = 0;
        uint64_t events_failed = 0;
        uint64_t vectors_indexed = 0;

        const size_t SAVE_INTERVAL = 100;  // Save every 100 events
        size_t events_since_last_save = 0;

        auto callback = [&](const std::string& filepath) {
            spdlog::info("New file detected: {}", filepath);

            try {
                // Load event
                auto event = loader.load(filepath);
                events_processed++;

                spdlog::info("Event loaded: id={}, features={}, class={}, confidence={:.4f}",
                    event.event_id,
                    event.features.size(),
                    event.final_class,
                    event.confidence
                );

                // Day 38.5: Generate embeddings
                auto chronos_emb = embedder.embed_chronos(event);
                auto sbert_emb = embedder.embed_sbert(event);
                auto attack_emb = embedder.embed_attack(event);

                spdlog::debug("Embeddings generated: chronos={}, sbert={}, attack={}",
                    chronos_emb.size(), sbert_emb.size(), attack_emb.size());

                // Day 38.5: Add to FAISS indices
                index_manager.add_chronos(chronos_emb);
                index_manager.add_sbert(sbert_emb);
                index_manager.add_entity_malicious(attack_emb);

                vectors_indexed++;

                // ============================================================
                // DAY 40: Insert metadata (Producer responsibility)
                // ============================================================

                // FAISS index of this event (0-based, last added)
                size_t faiss_idx = vectors_indexed - 1;

                // Extract metadata from event
                std::string event_id = event.event_id;
                std::string classification = event.final_class;
                float discrepancy_score = event.discrepancy_score;

                // Insert into metadata DB (sin IPs - no están en Event struct)
                try {
                    metadata_db->insert_event(
                        faiss_idx,
                        event_id,
                        classification,
                        discrepancy_score
                    );

                    spdlog::debug("Metadata inserted: faiss_idx={}, event_id={}",
                                 faiss_idx, event_id);

                } catch (const std::exception& e) {
                    spdlog::error("Failed to insert metadata for {}: {}",
                                 event_id, e.what());
                }

                // ============================================================
                // Periodic save to disk
                // ============================================================
                events_since_last_save++;
                if (events_since_last_save >= SAVE_INTERVAL) {
                    save_indices_to_disk();
                    events_since_last_save = 0;
                }

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
        spdlog::info("   Indices will be saved every {} events", SAVE_INTERVAL);

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
                spdlog::info("  Vectors indexed: {}", vectors_indexed);
                spdlog::info("  Metadata entries: {}", metadata_db->count());
                spdlog::info("  Total loaded: {}", stats.total_loaded);
                spdlog::info("  Total failed: {}", stats.total_failed);
                spdlog::info("  Bytes processed: {}", stats.bytes_processed);
                spdlog::info("  Partial features: {}", stats.partial_feature_count);

                last_stats = now;
            }
        }

        // ====================================================================
        // 9. Graceful shutdown (Day 40: SAVE FINAL STATE)
        // ====================================================================
        spdlog::info("Shutting down...");
        watcher.stop();

        // Save final state to disk
        spdlog::info("🔒 Saving final state to disk...");
        save_indices_to_disk();

        auto final_stats = loader.get_stats();
        spdlog::info("=== Final Statistics ===");
        spdlog::info("  Total events: {}", events_processed);
        spdlog::info("  Failed events: {}", events_failed);
        spdlog::info("  Vectors indexed: {}", vectors_indexed);
        spdlog::info("  Metadata entries: {}", metadata_db->count());
        spdlog::info("  Bytes processed: {}", final_stats.bytes_processed);

        spdlog::info("RAG Ingester stopped gracefully");
        return 0;

    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return 1;
    }
}