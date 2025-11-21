//===----------------------------------------------------------------------===//
// ML Defender - Firewall ACL Agent
// main.cpp - Application Entry Point
//
// Purpose:
//   Receive ML detections from ml-detector via ZMQ and block malicious IPs
//   in the kernel using ipset + iptables.
//
// Architecture:
//   ml-detector → zmq_subscriber → batch_processor → ipset_wrapper → kernel
//
// Via Appia Quality: Simple, robust, built to run for years
//===----------------------------------------------------------------------===//

#include "firewall/ipset_wrapper.hpp"
#include "firewall/iptables_wrapper.hpp"
#include "firewall/batch_processor.hpp"
#include "firewall/zmq_subscriber.hpp"

#include <json/json.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <csignal>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>

using namespace mldefender::firewall;

//===----------------------------------------------------------------------===//
// Global State (for signal handlers)
//===----------------------------------------------------------------------===//

std::atomic<bool> g_running{true};
ZMQSubscriber* g_subscriber = nullptr;
BatchProcessor* g_processor = nullptr;

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

struct Config {
    // IPSet configuration
    IPSetConfig ipset;

    // Batch processor configuration
    BatchConfig batch;

    // ZMQ configuration
    ZMQConfig zmq;

    // Metrics
    bool enable_metrics = true;
    int metrics_interval_sec = 60;
    std::string metrics_file;

    // Health checks
    bool enable_health_checks = true;
    int health_check_interval_sec = 30;

    // Logging
    bool verbose = false;
};

//===----------------------------------------------------------------------===//
// Configuration Loading
//===----------------------------------------------------------------------===//

bool load_config(const std::string& config_file, Config& config) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "[CONFIG] ERROR: Cannot open config file: " << config_file << std::endl;
        return false;
    }

    Json::Value root;
    Json::CharReaderBuilder reader;
    std::string errors;

    if (!Json::parseFromStream(reader, file, &root, &errors)) {
        std::cerr << "[CONFIG] ERROR: Failed to parse JSON: " << errors << std::endl;
        return false;
    }

    try {
        // IPSet configuration
        if (root.isMember("ipset")) {
            const auto& ipset = root["ipset"];
            config.ipset.set_name = ipset.get("set_name", "ml_defender_blacklist").asString();
            config.ipset.set_type = ipset.get("set_type", "hash:ip").asString();
            config.ipset.hash_size = ipset.get("hash_size", 4096).asUInt();
            config.ipset.max_elements = ipset.get("max_elements", 1000000).asUInt();
            config.ipset.default_timeout = ipset.get("timeout", 3600).asUInt();
        }

        // Batch processor configuration
        if (root.isMember("batch_processor")) {
            const auto& batch = root["batch_processor"];
            config.batch.batch_size_threshold = batch.get("batch_size_threshold", 1000).asUInt();
            config.batch.batch_time_threshold_ms = batch.get("batch_time_threshold_ms", 100).asUInt();
            config.batch.max_pending_ips = batch.get("max_pending_ips", 10000).asUInt();
            config.batch.min_confidence = batch.get("min_confidence", 0.5).asFloat();
            config.batch.enable_dedup = batch.get("enable_dedup", true).asBool();
        }

        // ZMQ configuration
        if (root.isMember("zmq")) {
            const auto& zmq = root["zmq"];
            config.zmq.endpoint = zmq.get("endpoint", "tcp://localhost:5555").asString();
            config.zmq.topic = zmq.get("topic", "").asString();
            config.zmq.recv_timeout_ms = zmq.get("recv_timeout_ms", 1000).asInt();
            config.zmq.linger_ms = zmq.get("linger_ms", 1000).asInt();
            config.zmq.reconnect_interval_ms = zmq.get("reconnect_interval_ms", 1000).asInt();
            config.zmq.max_reconnect_interval_ms = zmq.get("max_reconnect_interval_ms", 30000).asInt();
            config.zmq.reconnect_backoff_multiplier = zmq.get("reconnect_backoff_multiplier", 2.0).asDouble();
            config.zmq.rcvhwm = zmq.get("rcvhwm", 1000).asInt();
            config.zmq.enable_stats = zmq.get("enable_stats", true).asBool();
            config.zmq.stats_interval_sec = zmq.get("stats_interval_sec", 60).asInt();
        }

        // Metrics configuration
        if (root.isMember("metrics")) {
            const auto& metrics = root["metrics"];
            config.enable_metrics = metrics.get("enable_export", true).asBool();
            config.metrics_interval_sec = metrics.get("export_interval_sec", 60).asInt();
            config.metrics_file = metrics.get("export_file", "").asString();
        }

        // Health check configuration
        if (root.isMember("health_check")) {
            const auto& health = root["health_check"];
            config.enable_health_checks = health.get("enable", true).asBool();
            config.health_check_interval_sec = health.get("check_interval_sec", 30).asInt();
        }

        // Logging
        if (root.isMember("logging")) {
            const auto& logging = root["logging"];
            std::string level = logging.get("level", "info").asString();
            config.verbose = (level == "debug" || level == "trace");
        }

        std::cerr << "[CONFIG] ✓ Configuration loaded from " << config_file << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[CONFIG] ERROR: Exception parsing config: " << e.what() << std::endl;
        return false;
    }
}

//===----------------------------------------------------------------------===//
// Default Configuration
//===----------------------------------------------------------------------===//

Config create_default_config() {
    Config config;

    // IPSet defaults
    config.ipset.set_name = "ml_defender_blacklist";
    config.ipset.set_type = "hash:ip";
    config.ipset.hash_size = 4096;
    config.ipset.max_elements = 1000000;
    config.ipset.default_timeout = 3600;

    // Batch processor defaults
    config.batch.batch_size_threshold = 1000;
    config.batch.batch_time_threshold_ms = 100;
    config.batch.max_pending_ips = 10000;
    config.batch.min_confidence = 0.5f;
    config.batch.enable_dedup = true;

    // ZMQ defaults
    config.zmq.endpoint = "tcp://localhost:5555";
    config.zmq.topic = "";
    config.zmq.recv_timeout_ms = 1000;
    config.zmq.linger_ms = 1000;
    config.zmq.reconnect_interval_ms = 1000;
    config.zmq.max_reconnect_interval_ms = 30000;
    config.zmq.reconnect_backoff_multiplier = 2.0;
    config.zmq.rcvhwm = 1000;
    config.zmq.enable_stats = true;
    config.zmq.stats_interval_sec = 60;

    // Metrics defaults
    config.enable_metrics = true;
    config.metrics_interval_sec = 60;

    // Health checks
    config.enable_health_checks = true;
    config.health_check_interval_sec = 30;

    return config;
}

//===----------------------------------------------------------------------===//
// Signal Handlers
//===----------------------------------------------------------------------===//

void signal_handler(int signal) {
    std::cerr << "\n[SIGNAL] Received signal " << signal << " (";

    switch (signal) {
        case SIGINT:
            std::cerr << "SIGINT";
            break;
        case SIGTERM:
            std::cerr << "SIGTERM";
            break;
        case SIGHUP:
            std::cerr << "SIGHUP";
            break;
        default:
            std::cerr << "UNKNOWN";
    }

    std::cerr << "), initiating graceful shutdown..." << std::endl;

    // Stop the main loop
    g_running.store(false);

    // Stop components (they will finish current operations)
    if (g_subscriber) {
        g_subscriber->stop();
    }
}

void setup_signal_handlers() {
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGHUP, &sa, nullptr);

    // Ignore SIGPIPE (ZMQ handles it)
    signal(SIGPIPE, SIG_IGN);

    std::cerr << "[SIGNAL] Signal handlers installed (SIGINT, SIGTERM, SIGHUP)" << std::endl;
}

//===----------------------------------------------------------------------===//
// Metrics Export
//===----------------------------------------------------------------------===//

void export_metrics(
    const Config& config,
    const ZMQSubscriber& subscriber,
    const BatchProcessor& processor
) {
    if (!config.enable_metrics) {
        return;
    }

    // Get stats
    const auto& zmq_stats = subscriber.get_stats();
    const auto& proc_stats = processor.get_stats();

    // Create JSON
    Json::Value root;
    root["timestamp"] = static_cast<Json::Value::Int64>(
        std::chrono::system_clock::now().time_since_epoch().count()
    );

    // ZMQ stats
    root["zmq"]["messages_received"] = static_cast<Json::Value::UInt64>(
        zmq_stats.messages_received.load()
    );
    root["zmq"]["detections_processed"] = static_cast<Json::Value::UInt64>(
        zmq_stats.detections_processed.load()
    );
    root["zmq"]["parse_errors"] = static_cast<Json::Value::UInt64>(
        zmq_stats.parse_errors.load()
    );
    root["zmq"]["reconnects"] = static_cast<Json::Value::UInt64>(
        zmq_stats.reconnects.load()
    );
    root["zmq"]["connected"] = zmq_stats.currently_connected.load();

    // Processor stats
    root["processor"]["detections_received"] = static_cast<Json::Value::UInt64>(
        proc_stats.detections_received.load()
    );
    root["processor"]["batches_flushed"] = static_cast<Json::Value::UInt64>(
        proc_stats.batches_flushed.load()
    );
    root["processor"]["ips_blocked"] = static_cast<Json::Value::UInt64>(
        proc_stats.ips_blocked.load()
    );
    root["processor"]["duplicates_filtered"] = static_cast<Json::Value::UInt64>(
        proc_stats.duplicates_filtered.load()
    );
    root["processor"]["flush_errors"] = static_cast<Json::Value::UInt64>(
        proc_stats.flush_errors.load()
    );

    // Export to file if configured
    if (!config.metrics_file.empty()) {
        std::ofstream file(config.metrics_file);
        if (file.is_open()) {
            Json::StreamWriterBuilder writer;
            file << Json::writeString(writer, root);
            file.close();
        }
    }

    // Also print to console
    std::cerr << "\n╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cerr << "║  Metrics Summary                                      ║" << std::endl;
    std::cerr << "╚════════════════════════════════════════════════════════╝" << std::endl;
    std::cerr << "ZMQ Messages:       " << zmq_stats.messages_received.load() << std::endl;
    std::cerr << "Detections:         " << zmq_stats.detections_processed.load() << std::endl;
    std::cerr << "IPs Blocked:        " << proc_stats.ips_blocked.load() << std::endl;
    std::cerr << "Duplicates Filtered:" << proc_stats.duplicates_filtered.load() << std::endl;
    std::cerr << "Batches Flushed:    " << proc_stats.batches_flushed.load() << std::endl;
    std::cerr << "Parse Errors:       " << zmq_stats.parse_errors.load() << std::endl;
    std::cerr << "Flush Errors:       " << proc_stats.flush_errors.load() << std::endl;
    std::cerr << "ZMQ Connected:      " << (zmq_stats.currently_connected.load() ? "YES" : "NO") << std::endl;
    std::cerr << "════════════════════════════════════════════════════════" << std::endl;
}

//===----------------------------------------------------------------------===//
// Health Checks
//===----------------------------------------------------------------------===//

bool perform_health_checks(
    const Config& config,
    IPSetWrapper& ipset,
    IPTablesWrapper& iptables,
    const ZMQSubscriber& subscriber
) {
    if (!config.enable_health_checks) {
        return true;
    }

    bool healthy = true;

    // Check IPSet
    if (!ipset.exists(config.ipset.set_name)) {
        std::cerr << "[HEALTH] ✗ IPSet '" << config.ipset.set_name << "' does not exist!" << std::endl;
        healthy = false;
    }

    // Check IPTables rules
    auto rules = iptables.list_rules("INPUT");
    bool found_rule = false;
    for (const auto& rule : rules) {
        if (rule.find(config.ipset.set_name) != std::string::npos) {
            found_rule = true;
            break;
        }
    }

    if (!found_rule) {
        std::cerr << "[HEALTH] ✗ IPTables rule for '" << config.ipset.set_name << "' not found!" << std::endl;
        healthy = false;
    }

    // Check ZMQ connection
    if (!subscriber.is_running()) {
        std::cerr << "[HEALTH] ✗ ZMQ subscriber not running!" << std::endl;
        healthy = false;
    }

    if (healthy) {
        std::cerr << "[HEALTH] ✓ All systems healthy" << std::endl;
    }

    return healthy;
}

//===----------------------------------------------------------------------===//
// Main Function
//===----------------------------------------------------------------------===//

int main(int argc, char* argv[]) {
    // Print banner
    std::cerr << "\n";
    std::cerr << "╔════════════════════════════════════════════════════════╗\n";
    std::cerr << "║  ML Defender - Firewall ACL Agent v1.0.0             ║\n";
    std::cerr << "║  Purpose: Block malicious IPs detected by ML         ║\n";
    std::cerr << "║  Philosophy: Via Appia Quality - Built to Last       ║\n";
    std::cerr << "╚════════════════════════════════════════════════════════╝\n";
    std::cerr << std::endl;

    // Parse command line
    std::string config_file = "/etc/ml-defender/firewall.json";

    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [config_file]\n";
            std::cout << "\n";
            std::cout << "Options:\n";
            std::cout << "  config_file    Path to JSON configuration (default: /etc/ml-defender/firewall.json)\n";
            std::cout << "  -h, --help     Show this help message\n";
            std::cout << "\n";
            std::cout << "Signals:\n";
            std::cout << "  SIGINT/SIGTERM Graceful shutdown (flush pending IPs)\n";
            std::cout << "  SIGHUP         Reload configuration (future)\n";
            std::cout << "\n";
            return 0;
        }
        config_file = arg;
    }

    try {
        // Load configuration
        std::cerr << "[CONFIG] Loading configuration from " << config_file << std::endl;
        Config config;

        if (!load_config(config_file, config)) {
            std::cerr << "[CONFIG] WARNING: Using default configuration" << std::endl;
            config = create_default_config();
        }

        // Setup signal handlers
        setup_signal_handlers();

        // Check root privileges
        if (geteuid() != 0) {
            std::cerr << "[INIT] WARNING: Not running as root, ipset/iptables operations may fail!" << std::endl;
        }

        //==================================================================
        // COMPONENT INITIALIZATION
        //==================================================================

        std::cerr << "\n[INIT] Initializing components..." << std::endl;

        // 1. IPSet Wrapper
        std::cerr << "[INIT] Creating IPSet wrapper..." << std::endl;
        IPSetWrapper ipset(config.ipset);

        // Create the ipset if it doesn't exist
        if (!ipset.exists(config.ipset.set_name)) {
            std::cerr << "[INIT] Creating ipset '" << config.ipset.set_name << "'..." << std::endl;
            if (!ipset.create_set(config.ipset.set_name)) {
                std::cerr << "[INIT] ERROR: Failed to create ipset!" << std::endl;
                return 1;
            }
            std::cerr << "[INIT] ✓ IPSet created" << std::endl;
        } else {
            std::cerr << "[INIT] ✓ IPSet already exists" << std::endl;
        }

        // 2. IPTables Wrapper
        std::cerr << "[INIT] Creating IPTables wrapper..." << std::endl;
        IPTablesWrapper iptables;

        // Setup base rules
        std::cerr << "[INIT] Setting up iptables rules..." << std::endl;
        if (!iptables.setup_base_rules(config.ipset.set_name)) {
            std::cerr << "[INIT] WARNING: Failed to setup iptables rules (may already exist)" << std::endl;
        } else {
            std::cerr << "[INIT] ✓ IPTables rules configured" << std::endl;
        }

        // 3. Batch Processor
        std::cerr << "[INIT] Creating Batch Processor..." << std::endl;
        BatchProcessor processor(ipset, config.batch);
        g_processor = &processor;
        std::cerr << "[INIT] ✓ Batch Processor ready" << std::endl;

        // 4. ZMQ Subscriber
        std::cerr << "[INIT] Creating ZMQ Subscriber..." << std::endl;
        ZMQSubscriber subscriber(processor, config.zmq);
        g_subscriber = &subscriber;
        std::cerr << "[INIT] ✓ ZMQ Subscriber ready" << std::endl;

        //==================================================================
        // START SUBSCRIBER THREAD
        //==================================================================

        std::cerr << "\n[INIT] Starting ZMQ subscriber thread..." << std::endl;
        std::thread zmq_thread([&subscriber]() {
            subscriber.run();  // Blocking
        });

        std::cerr << "[INIT] ✓ All components initialized and running" << std::endl;

        //==================================================================
        // EVENT LOOP
        //==================================================================

        std::cerr << "\n";
        std::cerr << "╔════════════════════════════════════════════════════════╗\n";
        std::cerr << "║  System Ready - Monitoring for ML Detections         ║\n";
        std::cerr << "║  Press Ctrl+C to shutdown gracefully                  ║\n";
        std::cerr << "╚════════════════════════════════════════════════════════╝\n";
        std::cerr << std::endl;

        auto last_metrics_time = std::chrono::steady_clock::now();
        auto last_health_check_time = std::chrono::steady_clock::now();

        while (g_running.load()) {
            // Sleep for a short interval
            std::this_thread::sleep_for(std::chrono::seconds(1));

            auto now = std::chrono::steady_clock::now();

            // Periodic metrics export
            if (config.enable_metrics) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_metrics_time
                ).count();

                if (elapsed >= config.metrics_interval_sec) {
                    export_metrics(config, subscriber, processor);
                    last_metrics_time = now;
                }
            }

            // Periodic health checks
            if (config.enable_health_checks) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_health_check_time
                ).count();

                if (elapsed >= config.health_check_interval_sec) {
                    perform_health_checks(config, ipset, iptables, subscriber);
                    last_health_check_time = now;
                }
            }
        }

        //==================================================================
        // GRACEFUL SHUTDOWN
        //==================================================================

        std::cerr << "\n[SHUTDOWN] Initiating graceful shutdown sequence..." << std::endl;

        // 1. Stop ZMQ subscriber
        std::cerr << "[SHUTDOWN] Stopping ZMQ subscriber..." << std::endl;
        subscriber.stop();

        // 2. Wait for ZMQ thread to finish
        std::cerr << "[SHUTDOWN] Waiting for ZMQ thread to finish..." << std::endl;
        if (zmq_thread.joinable()) {
            zmq_thread.join();
        }
        std::cerr << "[SHUTDOWN] ✓ ZMQ thread stopped" << std::endl;

        // 3. Flush pending IPs in batch processor
        std::cerr << "[SHUTDOWN] Flushing pending IPs..." << std::endl;
        processor.flush();
        std::cerr << "[SHUTDOWN] ✓ Pending IPs flushed" << std::endl;

        // 4. Final metrics
        std::cerr << "\n[SHUTDOWN] Final statistics:" << std::endl;
        export_metrics(config, subscriber, processor);

        // 5. Optional: cleanup iptables rules (commented out - keep rules active)
        // std::cerr << "[SHUTDOWN] Cleaning up iptables rules..." << std::endl;
        // iptables.cleanup_rules(config.ipset.set_name);

        std::cerr << "\n";
        std::cerr << "╔════════════════════════════════════════════════════════╗\n";
        std::cerr << "║  Shutdown Complete - All systems stopped cleanly     ║\n";
        std::cerr << "╚════════════════════════════════════════════════════════╝\n";
        std::cerr << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL] Exception in main: " << e.what() << std::endl;
        return 1;
    }
}