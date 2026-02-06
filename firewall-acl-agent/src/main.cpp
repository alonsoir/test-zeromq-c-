//===----------------------------------------------------------------------===//
// ML Defender - Firewall ACL Agent
// main.cpp - High-Performance Packet DROP Agent with Full Observability
// Day 50: Comprehensive diagnostics and crash analysis
//===----------------------------------------------------------------------===//

#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <ctime>

// Day 50: Observability infrastructure
#include "firewall_observability_logger.hpp"
#include "crash_diagnostics.hpp"

#include "firewall/config_loader.hpp"
#include "firewall/ipset_wrapper.hpp"
#include "firewall/iptables_wrapper.hpp"
#include "firewall/batch_processor.hpp"
#include "firewall/zmq_subscriber.hpp"
#include "firewall/etcd_client.hpp"

#include <json/json.h>
#include <unistd.h>

using namespace mldefender::firewall;

//===----------------------------------------------------------------------===//
// Global Observability Infrastructure (Day 50)
//===----------------------------------------------------------------------===//

namespace mldefender::firewall::observability {
    std::unique_ptr<ObservabilityLogger> g_logger;
}

namespace mldefender::firewall::diagnostics {
    std::unique_ptr<SystemState> g_system_state;
}

//===----------------------------------------------------------------------===//
// Signal Handling (Enhanced for Day 50)
//===----------------------------------------------------------------------===//

std::atomic<bool> g_running{true};

void signal_handler(int signum) {
    FIREWALL_LOG_INFO("Shutdown signal received", "signal", signum);

    // Stop processing loop
    if (mldefender::firewall::diagnostics::g_system_state) {
        mldefender::firewall::diagnostics::g_system_state->is_running.store(false);
    }
    g_running.store(false);

    // Dump final state
    DUMP_STATE_ON_ERROR("graceful_shutdown");
}

void setup_signal_handlers() {
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    FIREWALL_LOG_INFO("Signal handlers installed", "signals", "SIGINT,SIGTERM");
}

//===----------------------------------------------------------------------===//
// Metrics Export (Enhanced for Day 50)
//===----------------------------------------------------------------------===//

void export_metrics(const FirewallAgentConfig& config,
                   const ZMQSubscriber& subscriber,
                   const BatchProcessor& processor) {
    const auto& zmq_stats = subscriber.get_stats();

    FIREWALL_LOG_INFO("Periodic metrics export",
        "messages_received", zmq_stats.messages_received.load(),
        "detections_processed", zmq_stats.detections_processed.load(),
        "parse_errors", zmq_stats.parse_errors.load(),
        "reconnects", zmq_stats.reconnects.load());

    // Also dump system state from diagnostics
    DUMP_STATE_ON_ERROR("periodic_metrics");
}

//===----------------------------------------------------------------------===//
// Health Checks (Enhanced for Day 50)
//===----------------------------------------------------------------------===//

bool perform_health_checks(const FirewallAgentConfig& config,
                          IPSetWrapper& ipset,
                          IPTablesWrapper& iptables,
                          const ZMQSubscriber& subscriber) {
    FIREWALL_LOG_INFO("Starting health checks");
    bool all_healthy = true;

    // Check IPSet
    bool ipset_exists = ipset.set_exists(config.ipset.set_name);
    if (!ipset_exists) {
        FIREWALL_LOG_ERROR("Health check failed: IPSet missing",
            "ipset_name", config.ipset.set_name);
        all_healthy = false;
    } else {
        FIREWALL_LOG_DEBUG("Health check passed: IPSet exists",
            "ipset_name", config.ipset.set_name);
    }

    // Check IPTables rule
    auto rules = iptables.list_rules("INPUT");
    bool rule_found = false;
    for (const auto& rule : rules) {
        if (rule.find(config.ipset.set_name) != std::string::npos) {
            rule_found = true;
            break;
        }
    }

    if (!rule_found) {
        FIREWALL_LOG_ERROR("Health check failed: IPTables rule missing",
            "ipset_name", config.ipset.set_name);
        all_healthy = false;
    } else {
        FIREWALL_LOG_DEBUG("Health check passed: IPTables rule exists");
    }

    // Check ZMQ connection
    const auto& stats = subscriber.get_stats();
    bool zmq_connected = stats.currently_connected.load();
    if (!zmq_connected) {
        FIREWALL_LOG_WARN("Health check warning: ZMQ not connected");
        all_healthy = false;
    } else {
        FIREWALL_LOG_DEBUG("Health check passed: ZMQ connected");
    }

    FIREWALL_LOG_INFO("Health checks completed",
        "status", all_healthy ? "HEALTHY" : "DEGRADED",
        "ipset_ok", ipset_exists,
        "iptables_ok", rule_found,
        "zmq_ok", zmq_connected);

    return all_healthy;
}

//===----------------------------------------------------------------------===//
// Usage
//===----------------------------------------------------------------------===//

void print_usage(const char* program_name) {
    std::cout << "ML Defender - Firewall ACL Agent\n"
              << "High-Performance Packet DROP Agent (1M+ packets/sec)\n"
              << "Day 50: Enhanced with comprehensive observability\n\n"
              << "Usage: " << program_name << " [options]\n\n"
              << "Options:\n"
              << "  -c, --config <file>    Configuration file (required)\n"
              << "  -d, --daemonize        Run as daemon\n"
              << "  -t, --test-config      Test configuration and exit\n"
              << "  --verbose              Enable verbose logging (DEBUG level)\n"
              << "  --quiet                Quiet mode (ERROR level only)\n"
              << "  -v, --version          Show version\n"
              << "  -h, --help             Show this help\n\n"
              << "Day 50 Features:\n"
              << "  • Comprehensive crash diagnostics with backtrace\n"
              << "  • Microsecond-precision event logging\n"
              << "  • Performance counters and metrics\n"
              << "  • Automatic state dumps on errors\n"
              << std::endl;
}

void print_version() {
    std::cout << "ML Defender - Firewall ACL Agent v1.0.0 (Day 50)\n"
              << "Via Appia Quality: Built to last decades\n"
              << "Enhanced: Comprehensive observability and diagnostics\n"
              << std::endl;
}

//===----------------------------------------------------------------------===//
// Main Entry Point (Day 50 Enhanced)
//===----------------------------------------------------------------------===//

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string config_path = "/vagrant/config/firewall.json";  // ✅ Absolute path default
    bool test_config = false;
    bool verbose = false;
    bool quiet = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-v" || arg == "--version") {
            print_version();
            return 0;
        } else if (arg == "-c" || arg == "--config") {
            if (i + 1 < argc) {
                config_path = argv[++i];  // ✅ Use user-provided path
            } else {
                std::cerr << "[ERROR] --config requires a file path" << std::endl;
                return 1;
            }
        } else if (arg == "-t" || arg == "--test-config") {
            test_config = true;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--quiet") {
            quiet = true;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Day 50: Initialize Observability Infrastructure
    // ═══════════════════════════════════════════════════════════════════════

    try {
        // Determine log level
        bool enable_verbose = verbose && !quiet;  // Verbose overrides quiet

        // Initialize logger
        std::string log_path = "/vagrant/logs/firewall-acl-agent/firewall_detailed.log";
        mldefender::firewall::observability::g_logger =
            std::make_unique<mldefender::firewall::observability::ObservabilityLogger>(
                log_path, enable_verbose);
        // Initialize system state for crash diagnostics
        mldefender::firewall::diagnostics::g_system_state =
            std::make_unique<mldefender::firewall::diagnostics::SystemState>();
        FIREWALL_LOG_INFO("════════════════════════════════════════════════════════");
        FIREWALL_LOG_INFO("ML Defender - Firewall ACL Agent v1.0.0 (Day 50)");
        FIREWALL_LOG_INFO("High-Performance Packet DROP Agent");
        FIREWALL_LOG_INFO("Via Appia Quality: Built to last decades");
        FIREWALL_LOG_INFO("════════════════════════════════════════════════════════");

        FIREWALL_LOG_INFO("Observability initialized",
            "log_file", log_path,
            "verbose_mode", enable_verbose,
            "config_path", config_path);

        // Install crash diagnostics
        mldefender::firewall::diagnostics::install_crash_handlers();

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Failed to initialize observability: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "╔════════════════════════════════════════════════════════╗\n"
              << "║  ML Defender - Firewall ACL Agent (Day 50)            ║\n"
              << "║  High-Performance Packet DROP Agent                   ║\n"
              << "║  Target: 1M+ packets/sec                               ║\n"
              << "║  Enhanced: Comprehensive Observability                ║\n"
              << "╚════════════════════════════════════════════════════════╝\n"
              << std::endl;

    // ═══════════════════════════════════════════════════════════════════════
    // Configuration Loading
    // ═══════════════════════════════════════════════════════════════════════

    std::unique_ptr<mldefender::firewall::EtcdClient> etcd_client;
    FirewallAgentConfig config;

    try {
        FIREWALL_LOG_INFO("Loading configuration", "path", config_path);

        config = ConfigLoader::load_from_file(config_path);

        FIREWALL_LOG_INFO("Configuration loaded successfully",
            "config_file", config_path,
            "dry_run", config.operation.dry_run);

        // Show DRY-RUN status prominently
        if (config.operation.dry_run) {
            FIREWALL_LOG_WARN("DRY-RUN MODE ENABLED - No actual firewall modifications");
            std::cout << "\n╔════════════════════════════════════════════════════════╗\n"
                      << "║  🔍 DRY-RUN MODE ENABLED 🔍                           ║\n"
                      << "║  No actual firewall rules will be modified            ║\n"
                      << "╚════════════════════════════════════════════════════════╝\n"
                      << std::endl;
        }

        ConfigLoader::log_config_summary(config);

        // ═══════════════════════════════════════════════════════════════════════
        // ETCD Integration (Day 50: Fixed path bug)
        // ═══════════════════════════════════════════════════════════════════════

        if (config.etcd.enabled) {
            FIREWALL_LOG_INFO("etcd integration enabled",
                "endpoint", config.etcd.endpoints[0]);

            std::string etcd_endpoint = config.etcd.endpoints[0];

            try {
                etcd_client = std::make_unique<mldefender::firewall::EtcdClient>(
                    etcd_endpoint,
                    "firewall-acl-agent"
                );

                if (!etcd_client->initialize()) {
                    FIREWALL_LOG_WARN("etcd initialization failed - continuing without etcd");
                    etcd_client.reset();
                } else if (!etcd_client->registerService()) {
                    FIREWALL_LOG_WARN("etcd service registration failed - continuing without etcd");
                    etcd_client.reset();
                } else {
                    FIREWALL_LOG_INFO("etcd registration successful",
                        "service", "firewall-acl-agent",
                        "endpoint", etcd_endpoint);
                }
            } catch (const std::exception& e) {
                FIREWALL_LOG_ERROR("etcd client exception",
                    "error", e.what());
                etcd_client.reset();
            }
        } else {
            FIREWALL_LOG_INFO("etcd integration disabled in configuration");
        }

    } catch (const std::exception& e) {
        FIREWALL_LOG_CRASH("Configuration loading failed", "error", e.what());
        std::cerr << "[ERROR] Failed to load configuration: " << e.what() << std::endl;
        return 1;
    }

    FIREWALL_LOG_DEBUG("Configuration phase completed successfully");

    // Test config mode
    if (test_config) {
        FIREWALL_LOG_INFO("Configuration test mode - exiting");
        std::cout << "[INFO] Configuration test passed ✓" << std::endl;
        return 0;
    }

    // Permission check
    if (getuid() != 0 && !config.operation.dry_run) {
        FIREWALL_LOG_CRASH("Insufficient privileges - must run as root",
            "uid", getuid(),
            "dry_run", config.operation.dry_run);
        std::cerr << "[ERROR] This program must be run as root (or use dry_run mode)" << std::endl;
        return 1;
    }

    FIREWALL_LOG_DEBUG("Permission check passed",
        "uid", getuid(),
        "dry_run", config.operation.dry_run);

    // ═══════════════════════════════════════════════════════════════════════
    // Component Initialization (Day 50: Enhanced with logging)
    // ═══════════════════════════════════════════════════════════════════════

    try {
        FIREWALL_LOG_INFO("Starting component initialization phase");

        // ───────────────────────────────────────────────────────────────────
        // IPSet Configuration
        // ───────────────────────────────────────────────────────────────────

        FIREWALL_LOG_INFO("Configuring IPSet",
            "set_name", config.ipset.set_name,
            "hash_size", config.ipset.hash_size,
            "max_elements", config.ipset.max_elements,
            "timeout", config.ipset.timeout);

        IPSetConfig ipset_config;
        ipset_config.name = config.ipset.set_name;
        ipset_config.type = IPSetType::HASH_IP;
        ipset_config.family = IPSetFamily::INET;
        ipset_config.hashsize = config.ipset.hash_size;
        ipset_config.maxelem = config.ipset.max_elements;
        ipset_config.timeout = config.ipset.timeout;
        ipset_config.counters = true;
        ipset_config.comment = !config.ipset.comment.empty();

        FIREWALL_LOG_INFO("Initializing IPSet wrapper");
        IPSetWrapper ipset;
        ipset.set_dry_run(config.operation.dry_run);

        // Create primary ipset
        if (!ipset.set_exists(config.ipset.set_name)) {
            FIREWALL_LOG_INFO("Creating primary ipset", "name", config.ipset.set_name);

            if (!ipset.create_set(ipset_config)) {
                FIREWALL_LOG_ERROR("Failed to create primary ipset",
                    "name", config.ipset.set_name);
                DUMP_STATE_ON_ERROR("ipset_creation_failed");
                std::cerr << "[ERROR] Failed to create ipset" << std::endl;
                return 1;
            }

            FIREWALL_LOG_INFO("Primary ipset created successfully",
                "name", config.ipset.set_name);
        } else {
            FIREWALL_LOG_INFO("Primary ipset already exists",
                "name", config.ipset.set_name);
        }

        // Create additional ipsets
        if (!config.ipsets.empty()) {
            FIREWALL_LOG_INFO("Processing additional ipsets",
                "count", config.ipsets.size());

            for (const auto& [name, ipset_cfg] : config.ipsets) {
                if (!ipset.set_exists(ipset_cfg.set_name)) {
                    FIREWALL_LOG_INFO("Creating additional ipset",
                        "logical_name", name,
                        "set_name", ipset_cfg.set_name,
                        "timeout", ipset_cfg.timeout);

                    IPSetConfig additional_config;
                    additional_config.name = ipset_cfg.set_name;
                    additional_config.type = IPSetType::HASH_IP;
                    additional_config.family = IPSetFamily::INET;
                    additional_config.hashsize = ipset_cfg.hash_size;
                    additional_config.maxelem = ipset_cfg.max_elements;
                    additional_config.timeout = ipset_cfg.timeout;
                    additional_config.counters = true;
                    additional_config.comment = !ipset_cfg.comment.empty();

                    if (!ipset.create_set(additional_config)) {
                        FIREWALL_LOG_ERROR("Failed to create additional ipset",
                            "logical_name", name,
                            "set_name", ipset_cfg.set_name);
                        DUMP_STATE_ON_ERROR("additional_ipset_creation_failed");
                        return 1;
                    }

                    FIREWALL_LOG_INFO("Additional ipset created",
                        "logical_name", name,
                        "set_name", ipset_cfg.set_name);
                } else {
                    FIREWALL_LOG_DEBUG("Additional ipset already exists",
                        "logical_name", name,
                        "set_name", ipset_cfg.set_name);
                }
            }
        }

        // ───────────────────────────────────────────────────────────────────
        // IPTables Configuration
        // ───────────────────────────────────────────────────────────────────

        FIREWALL_LOG_INFO("Configuring IPTables",
            "chain", config.iptables.chain_name,
            "rate_limiting", config.iptables.enable_rate_limiting);

        FirewallConfig firewall_config;
        firewall_config.blacklist_chain = config.iptables.chain_name;
        firewall_config.blacklist_ipset = config.ipset.set_name;

        if (config.ipsets.count("whitelist") > 0) {
            firewall_config.whitelist_ipset = config.ipsets.at("whitelist").set_name;
            FIREWALL_LOG_INFO("Whitelist configured",
                "ipset_name", firewall_config.whitelist_ipset);
        }

        firewall_config.enable_rate_limiting = config.iptables.enable_rate_limiting;
        firewall_config.rate_limit_connections = config.iptables.rate_limit_connections;

        FIREWALL_LOG_INFO("Initializing IPTables wrapper");
        IPTablesWrapper iptables;
        iptables.set_dry_run(config.operation.dry_run);

        FIREWALL_LOG_INFO("Setting up IPTables rules");
        if (!iptables.setup_base_rules(firewall_config)) {
            FIREWALL_LOG_ERROR("Failed to setup IPTables rules");
            DUMP_STATE_ON_ERROR("iptables_setup_failed");

            if (!config.operation.dry_run) {
                std::cerr << "[ERROR] Failed to setup IPTables rules" << std::endl;
                return 1;
            }
        }

        FIREWALL_LOG_INFO("IPTables rules configured successfully");

        // ───────────────────────────────────────────────────────────────────
        // Batch Processor Configuration
        // ───────────────────────────────────────────────────────────────────

        FIREWALL_LOG_INFO("Configuring batch processor",
            "batch_size_threshold", config.batch_processor.batch_size_threshold,
            "batch_time_threshold_ms", config.batch_processor.batch_time_threshold_ms,
            "max_pending_ips", config.batch_processor.max_pending_ips,
            "min_confidence", config.batch_processor.min_confidence);

        BatchProcessorConfig batch_config;
        batch_config.batch_size_threshold = config.batch_processor.batch_size_threshold;
        batch_config.batch_time_threshold = std::chrono::milliseconds(config.batch_processor.batch_time_threshold_ms);
        batch_config.max_pending_ips = config.batch_processor.max_pending_ips;
        batch_config.confidence_threshold = config.batch_processor.min_confidence;

        FIREWALL_LOG_INFO("Initializing batch processor");
        BatchProcessor processor(ipset, batch_config);
        FIREWALL_LOG_INFO("Batch processor started successfully");

        // ───────────────────────────────────────────────────────────────────
        // ZMQ Configuration (Day 50: Enhanced transport logging)
        // ───────────────────────────────────────────────────────────────────

        FIREWALL_LOG_INFO("Configuring ZMQ subscriber",
            "endpoint", config.zmq.endpoint,
            "topic", config.zmq.topic,
            "recv_timeout_ms", config.zmq.recv_timeout_ms);

        ZMQSubscriber::Config zmq_config;
        zmq_config.endpoint = config.zmq.endpoint;
        zmq_config.topic = config.zmq.topic;
        zmq_config.recv_timeout_ms = config.zmq.recv_timeout_ms;
        zmq_config.linger_ms = config.zmq.linger_ms;
        zmq_config.reconnect_interval_ms = config.zmq.reconnect_interval_ms;
        zmq_config.max_reconnect_interval_ms = config.zmq.max_reconnect_interval_ms;
        zmq_config.enable_reconnect = config.zmq.enable_reconnect;

        // Transport configuration (compression)
        if (config.transport.compression.enabled) {
            zmq_config.compression_enabled = true;
            FIREWALL_LOG_INFO("LZ4 decompression enabled",
                "algorithm", config.transport.compression.algorithm);
        } else {
            zmq_config.compression_enabled = false;
            FIREWALL_LOG_INFO("LZ4 decompression disabled");
        }

        // Transport configuration (encryption)
        if (config.transport.encryption.enabled) {
            zmq_config.encryption_enabled = true;

            if (!etcd_client) {
                FIREWALL_LOG_CRASH("Encryption enabled but etcd not initialized");
                std::cerr << "[ERROR] Encryption enabled but etcd not initialized" << std::endl;
                std::cerr << "[ERROR] Set etcd.enabled = true in firewall.json" << std::endl;
                return 1;
            }

            FIREWALL_LOG_INFO("Retrieving crypto seed from etcd");
            zmq_config.crypto_token = etcd_client->get_crypto_seed();

            if (zmq_config.crypto_token.empty()) {
                FIREWALL_LOG_CRASH("Failed to retrieve crypto seed from etcd");
                std::cerr << "[ERROR] Failed to get crypto seed from etcd-server" << std::endl;
                return 1;
            }

            FIREWALL_LOG_INFO("ChaCha20-Poly1305 decryption enabled",
                "algorithm", config.transport.encryption.algorithm,
                "key_length", zmq_config.crypto_token.length());
        } else {
            zmq_config.encryption_enabled = false;
            FIREWALL_LOG_INFO("Decryption disabled");
        }

        FIREWALL_LOG_INFO("Initializing ZMQ subscriber");
        ZMQSubscriber subscriber(processor, zmq_config);

        // Start ZMQ subscriber thread
        FIREWALL_LOG_INFO("Starting ZMQ subscriber thread");
        mldefender::firewall::diagnostics::g_system_state->is_running.store(true);
        std::thread zmq_thread([&subscriber]() {
            subscriber.run();
        });
        zmq_thread.detach();

        FIREWALL_LOG_INFO("ZMQ subscriber initialized and running");

        // ───────────────────────────────────────────────────────────────────
        // Signal Handlers (Day 50: Enhanced)
        // ───────────────────────────────────────────────────────────────────

        setup_signal_handlers();

        // ───────────────────────────────────────────────────────────────────
        // Main Loop (Day 50: Enhanced with diagnostics)
        // ───────────────────────────────────────────────────────────────────

        FIREWALL_LOG_INFO("════════════════════════════════════════════════════════");
        FIREWALL_LOG_INFO("Firewall ACL Agent is now ACTIVE");
        FIREWALL_LOG_INFO("════════════════════════════════════════════════════════");
        FIREWALL_LOG_INFO("Configuration summary",
            "zmq_endpoint", config.zmq.endpoint,
            "topic_filter", config.zmq.topic.empty() ? "(all)" : config.zmq.topic,
            "ipset_name", config.ipset.set_name,
            "dry_run", config.operation.dry_run);

        std::cout << "\n[RUNNING] Firewall ACL Agent is now active\n"
                  << "[INFO] Listening on: " << config.zmq.endpoint << "\n"
                  << "[INFO] Topic filter: " << (config.zmq.topic.empty() ? "(all)" : config.zmq.topic) << "\n"
                  << "[INFO] Press Ctrl+C to stop\n"
                  << std::endl;

        auto last_health_check = std::chrono::steady_clock::now();
        auto last_metrics_export = std::chrono::steady_clock::now();

        while (g_running.load() &&
               mldefender::firewall::diagnostics::g_system_state->is_running.load()) {

            auto now = std::chrono::steady_clock::now();

            try {
                // Health checks every 30 seconds
                auto elapsed_health = std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_health_check).count();

                if (elapsed_health >= 30) {
                    perform_health_checks(config, ipset, iptables, subscriber);
                    last_health_check = now;
                }

                // Metrics export every 30 seconds
                auto elapsed_metrics = std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_metrics_export).count();

                if (elapsed_metrics >= 30) {
                    export_metrics(config, subscriber, processor);
                    last_metrics_export = now;
                }

            } catch (const std::exception& e) {
                FIREWALL_LOG_ERROR("Exception in main loop",
                    "error", e.what());
                DUMP_STATE_ON_ERROR("main_loop_exception");
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // ───────────────────────────────────────────────────────────────────
        // Graceful Shutdown (Day 50: Enhanced)
        // ───────────────────────────────────────────────────────────────────

        FIREWALL_LOG_INFO("════════════════════════════════════════════════════════");
        FIREWALL_LOG_INFO("Initiating graceful shutdown");
        FIREWALL_LOG_INFO("════════════════════════════════════════════════════════");

        std::cout << "\n[SHUTDOWN] Stopping components..." << std::endl;

        // Export final metrics
        export_metrics(config, subscriber, processor);
        FIREWALL_LOG_INFO("Final metrics exported");

        // Display final statistics
        const auto& zmq_stats = subscriber.get_stats();

        FIREWALL_LOG_INFO("Final statistics",
            "messages_received", zmq_stats.messages_received.load(),
            "detections_processed", zmq_stats.detections_processed.load(),
            "parse_errors", zmq_stats.parse_errors.load(),
            "reconnections", zmq_stats.reconnects.load());

        std::cout << "\n[STATS] Final Statistics:\n"
                  << "  Messages Received:     " << zmq_stats.messages_received.load() << "\n"
                  << "  Detections Processed:  " << zmq_stats.detections_processed.load() << "\n"
                  << "  Parse Errors:          " << zmq_stats.parse_errors.load() << "\n"
                  << "  Reconnections:         " << zmq_stats.reconnects.load() << "\n"
                  << std::endl;

        // Final state dump
        DUMP_STATE_ON_ERROR("clean_shutdown");

        FIREWALL_LOG_INFO("════════════════════════════════════════════════════════");
        FIREWALL_LOG_INFO("Firewall ACL Agent stopped cleanly");
        FIREWALL_LOG_INFO("════════════════════════════════════════════════════════");

        std::cout << "[SHUTDOWN] ✓ Firewall ACL Agent stopped cleanly" << std::endl;

    } catch (const std::exception& e) {
        FIREWALL_LOG_CRASH("Fatal exception in main",
            "error", e.what());
        DUMP_STATE_ON_ERROR("fatal_exception");

        std::cerr << "[FATAL] Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}