//===----------------------------------------------------------------------===//
// ML Defender - Firewall ACL Agent
// main.cpp - High-Performance Packet DROP Agent
//===----------------------------------------------------------------------===//

#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <ctime>

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
// Signal Handling
//===----------------------------------------------------------------------===//

std::atomic<bool> g_running{true};

void signal_handler(int signum) {
    std::cout << "\n[SIGNAL] Received signal " << signum << ", shutting down gracefully..." << std::endl;
    g_running.store(false);
}

void setup_signal_handlers() {
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}

//===----------------------------------------------------------------------===//
// Metrics Export
//===----------------------------------------------------------------------===//

void export_metrics(const FirewallAgentConfig& config, 
                   const ZMQSubscriber& subscriber,
                   const BatchProcessor& processor) {
    // Note: metrics export removed from new config - can be re-added if needed
    // For now, just log stats
    const auto& zmq_stats = subscriber.get_stats();
    
    std::cout << "[METRICS] Messages: " << zmq_stats.messages_received.load()
              << " | Detections: " << zmq_stats.detections_processed.load()
              << " | Errors: " << zmq_stats.parse_errors.load() << std::endl;
}

//===----------------------------------------------------------------------===//
// Health Checks
//===----------------------------------------------------------------------===//

bool perform_health_checks(const FirewallAgentConfig& config,
                          IPSetWrapper& ipset,
                          IPTablesWrapper& iptables,
                          const ZMQSubscriber& subscriber) {
    bool all_healthy = true;
    
    std::cout << "[HEALTH] Running health checks..." << std::endl;
    
    // Check IPSet
    if (!ipset.set_exists(config.ipset.set_name)) {
        std::cerr << "[HEALTH] ✗ IPSet '" << config.ipset.set_name << "' does not exist!" << std::endl;
        all_healthy = false;
    } else {
        std::cout << "[HEALTH] ✓ IPSet exists" << std::endl;
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
        std::cerr << "[HEALTH] ✗ IPTables rule for '" << config.ipset.set_name << "' not found!" << std::endl;
        all_healthy = false;
    } else {
        std::cout << "[HEALTH] ✓ IPTables rule exists" << std::endl;
    }
    
    // Check ZMQ connection
    const auto& stats = subscriber.get_stats();
    if (!stats.currently_connected.load()) {
        std::cerr << "[HEALTH] ✗ ZMQ not connected!" << std::endl;
        all_healthy = false;
    } else {
        std::cout << "[HEALTH] ✓ ZMQ connected" << std::endl;
    }
    
    return all_healthy;
}

//===----------------------------------------------------------------------===//
// Usage
//===----------------------------------------------------------------------===//

void print_usage(const char* program_name) {
    std::cout << "ML Defender - Firewall ACL Agent\n"
              << "High-Performance Packet DROP Agent (1M+ packets/sec)\n\n"
              << "Usage: " << program_name << " [options]\n\n"
              << "Options:\n"
              << "  -c, --config <file>    Configuration file\n"
              << "  -d, --daemonize        Run as daemon\n"
              << "  -t, --test-config      Test configuration and exit\n"
              << "  -v, --version          Show version\n"
              << "  -h, --help             Show this help\n"
              << std::endl;
}

void print_version() {
    std::cout << "ML Defender - Firewall ACL Agent v1.0.0\n"
              << "Via Appia Quality: Built to last decades\n"
              << std::endl;
}

//===----------------------------------------------------------------------===//
// Main Entry Point
//===----------------------------------------------------------------------===//

int main(int argc, char** argv) {
    std::string config_path = "../config/firewall.json";
    bool test_config = false;
    
    // Parse command line arguments
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
                config_path = argv[++i];
            } else {
                std::cerr << "[ERROR] --config requires a file path" << std::endl;
                return 1;
            }
        } else if (arg == "-t" || arg == "--test-config") {
            test_config = true;
        }
    }
    
    std::cout << "╔════════════════════════════════════════════════════════╗\n"
              << "║  ML Defender - Firewall ACL Agent                     ║\n"
              << "║  High-Performance Packet DROP Agent                   ║\n"
              << "║  Target: 1M+ packets/sec                               ║\n"
              << "╚════════════════════════════════════════════════════════╝\n"
              << std::endl;
    
    // Load configuration using new ConfigLoader
    FirewallAgentConfig config;
    try {
        config = ConfigLoader::load_from_file(config_path);
        std::cout << "[INFO] Configuration loaded from: " << config_path << std::endl;
        
        // Show DRY-RUN status prominently
        if (config.operation.dry_run) {
            std::cout << "\n╔════════════════════════════════════════════════════════╗\n"
                      << "║  🔍 DRY-RUN MODE ENABLED 🔍                           ║\n"
                      << "║  No actual firewall rules will be modified            ║\n"
                      << "╚════════════════════════════════════════════════════════╝\n"
                      << std::endl;
        }
        
        ConfigLoader::log_config_summary(config);

        // ═══════════════════════════════════════════════════════════════════════
        // ETCD INTEGRATION - Register component and upload config
        // ═══════════════════════════════════════════════════════════════════════
        std::unique_ptr<mldefender::firewall::EtcdClient> etcd_client;

        if (config.etcd.enabled) {
            std::string etcd_endpoint = config.etcd.endpoints[0];

            std::cout << "🔗 [etcd] Initializing connection to " << etcd_endpoint << std::endl;

            etcd_client = std::make_unique<mldefender::firewall::EtcdClient>(
                etcd_endpoint,
                "firewall-acl-agent"
            );

            if (!etcd_client->initialize()) {
                std::cerr << "⚠️  [etcd] Failed to initialize - continuing without etcd" << std::endl;
                etcd_client.reset();
            } else if (!etcd_client->registerService()) {
                std::cerr << "⚠️  [etcd] Failed to register service - continuing without etcd" << std::endl;
                etcd_client.reset();
            } else {
                std::cout << "✅ [etcd] firewall-acl-agent registered and config uploaded" << std::endl;
            }
        } else {
            std::cout << "⏭️  [etcd] etcd integration disabled in config" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load configuration: " << e.what() << std::endl;
        return 1;
    }
    
    if (test_config) {
        std::cout << "[INFO] Configuration test passed ✓" << std::endl;
        return 0;
    }
    
    if (getuid() != 0 && !config.operation.dry_run) {
        std::cerr << "[ERROR] This program must be run as root (or use dry_run mode)" << std::endl;
        return 1;
    }
    
    try {
        // Convert new config structs to old wrapper configs
        // TODO: Eventually refactor wrappers to use new config directly
        
        // IPSet configuration
        IPSetConfig ipset_config;
        ipset_config.name = config.ipset.set_name;
        ipset_config.type = IPSetType::HASH_IP; // From set_type string
        ipset_config.family = IPSetFamily::INET; // From family string
        ipset_config.hashsize = config.ipset.hash_size;
        ipset_config.maxelem = config.ipset.max_elements;
        ipset_config.timeout = config.ipset.timeout;
        ipset_config.counters = true;
        ipset_config.comment = !config.ipset.comment.empty();
        
        std::cout << "[INIT] Initializing IPSet wrapper..." << std::endl;
        IPSetWrapper ipset;
        ipset.set_dry_run(config.operation.dry_run); // Set dry-run mode
        
        // Create primary ipset (backward compatibility)
        if (!ipset.set_exists(config.ipset.set_name)) {
            std::cout << "[INIT] Creating ipset '" << config.ipset.set_name << "'..." << std::endl;
            if (!ipset.create_set(ipset_config)) {
                std::cerr << "[ERROR] Failed to create ipset" << std::endl;
                return 1;
            }
            std::cout << "[INIT] ✓ IPSet created successfully" << std::endl;
        } else {
            std::cout << "[INIT] ✓ IPSet already exists" << std::endl;
        }
        
        // Create additional ipsets from "ipsets" section
        if (!config.ipsets.empty()) {
            std::cout << "[INIT] Creating additional ipsets..." << std::endl;
            for (const auto& [name, ipset_cfg] : config.ipsets) {
                if (!ipset.set_exists(ipset_cfg.set_name)) {
                    std::cout << "[INIT] Creating ipset '" << ipset_cfg.set_name << "' (" << name << ")..." << std::endl;
                    
                    // Convert config to IPSetConfig
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
                        std::cerr << "[ERROR] Failed to create ipset '" << ipset_cfg.set_name << "'" << std::endl;
                        return 1;
                    }
                    std::cout << "[INIT] ✓ IPSet '" << ipset_cfg.set_name << "' created" << std::endl;
                } else {
                    std::cout << "[INIT] ✓ IPSet '" << ipset_cfg.set_name << "' already exists" << std::endl;
                }
            }
        }
        
        // IPTables configuration
        FirewallConfig firewall_config;
        firewall_config.blacklist_chain = config.iptables.chain_name;
        firewall_config.blacklist_ipset = config.ipset.set_name;
        
        // Configure whitelist if present in ipsets
        if (config.ipsets.count("whitelist") > 0) {
            firewall_config.whitelist_ipset = config.ipsets.at("whitelist").set_name;
            std::cout << "[INIT] Whitelist ipset configured: " << firewall_config.whitelist_ipset << std::endl;
        }
        
        firewall_config.enable_rate_limiting = config.iptables.enable_rate_limiting;
        firewall_config.rate_limit_connections = config.iptables.rate_limit_connections;
        
        std::cout << "[INIT] Initializing IPTables wrapper..." << std::endl;
        IPTablesWrapper iptables;
        iptables.set_dry_run(config.operation.dry_run); // Set dry-run mode
        
        std::cout << "[INIT] Setting up IPTables rules..." << std::endl;
        if (!iptables.setup_base_rules(firewall_config)) {
            std::cerr << "[ERROR] Failed to setup IPTables rules" << std::endl;
            if (!config.operation.dry_run) {
                return 1;
            }
        }
        std::cout << "[INIT] ✓ IPTables rules configured" << std::endl;
        
        // Batch processor configuration
        BatchProcessorConfig batch_config;
        batch_config.batch_size_threshold = config.batch_processor.batch_size_threshold;
        batch_config.batch_time_threshold = std::chrono::milliseconds(config.batch_processor.batch_time_threshold_ms);
        batch_config.max_pending_ips = config.batch_processor.max_pending_ips;
        batch_config.confidence_threshold = config.batch_processor.min_confidence;
        
        std::cout << "[INIT] Initializing batch processor..." << std::endl;
        BatchProcessor processor(ipset, batch_config);
        std::cout << "[INIT] ✓ Batch processor started" << std::endl;
        
        // ZMQ configuration
        ZMQSubscriber::Config zmq_config;
        zmq_config.endpoint = config.zmq.endpoint;
        zmq_config.topic = config.zmq.topic;
        zmq_config.recv_timeout_ms = config.zmq.recv_timeout_ms;
        zmq_config.linger_ms = config.zmq.linger_ms;
        zmq_config.reconnect_interval_ms = config.zmq.reconnect_interval_ms;
        zmq_config.max_reconnect_interval_ms = config.zmq.max_reconnect_interval_ms;
        zmq_config.enable_reconnect = config.zmq.enable_reconnect;

        // ✅ Day 23: Transport configuration (encryption + compression)
        if (config.transport.compression.enabled) {
            zmq_config.compression_enabled = true;
            std::cout << "[INIT] 📦 LZ4 decompression ENABLED" << std::endl;
        } else {
            zmq_config.compression_enabled = false;
            std::cout << "[INIT] ⏭️  LZ4 decompression DISABLED" << std::endl;
        }

        if (config.transport.encryption.enabled) {
            zmq_config.encryption_enabled = true;

            // TODO: Get crypto token from etcd in production
            // For now, use hardcoded test token (MUST MATCH ml-detector)
            zmq_config.crypto_token = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

            std::cout << "[INIT] 🔐 ChaCha20-Poly1305 decryption ENABLED" << std::endl;
            std::cout << "[INIT] ⚠️  Using HARDCODED crypto token (TODO: fetch from etcd)" << std::endl;
        } else {
            zmq_config.encryption_enabled = false;
            std::cout << "[INIT] ⏭️  Decryption DISABLED" << std::endl;
        }
        
        std::cout << "[INIT] Initializing ZMQ subscriber..." << std::endl;
        ZMQSubscriber subscriber(processor, zmq_config);
        
        // Start ZMQ subscriber thread
        std::thread zmq_thread([&subscriber]() {
            subscriber.run();
        });
        zmq_thread.detach();
        std::cout << "[INIT] ✓ ZMQ subscriber initialized" << std::endl;
        
        setup_signal_handlers();
        
        std::cout << "\n[RUNNING] Firewall ACL Agent is now active\n"
                  << "[INFO] Listening on: " << config.zmq.endpoint << "\n"
                  << "[INFO] Topic filter: " << (config.zmq.topic.empty() ? "(all)" : config.zmq.topic) << "\n"
                  << "[INFO] Press Ctrl+C to stop\n"
                  << std::endl;
        
        auto last_health_check = std::chrono::steady_clock::now();
        auto last_metrics_export = std::chrono::steady_clock::now();
        
        while (g_running.load()) {
            auto now = std::chrono::steady_clock::now();
            
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
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "\n[SHUTDOWN] Stopping components..." << std::endl;
        
        export_metrics(config, subscriber, processor);
        std::cout << "[SHUTDOWN] ✓ Final metrics exported" << std::endl;
        
        const auto& zmq_stats = subscriber.get_stats();
        std::cout << "\n[STATS] Final Statistics:\n"
                  << "  Messages Received:     " << zmq_stats.messages_received.load() << "\n"
                  << "  Detections Processed:  " << zmq_stats.detections_processed.load() << "\n"
                  << "  Parse Errors:          " << zmq_stats.parse_errors.load() << "\n"
                  << "  Reconnections:         " << zmq_stats.reconnects.load() << "\n"
                  << std::endl;
        
        std::cout << "[SHUTDOWN] ✓ Firewall ACL Agent stopped cleanly" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
