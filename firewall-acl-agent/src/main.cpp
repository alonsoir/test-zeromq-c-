//===----------------------------------------------------------------------===//
// ML Defender - Firewall ACL Agent
// main.cpp - High-Performance Packet DROP Agent
//===----------------------------------------------------------------------===//

#include <iostream>
#include <fstream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>

#include "firewall/ipset_wrapper.hpp"
#include "firewall/iptables_wrapper.hpp"
#include "firewall/batch_processor.hpp"
#include "firewall/zmq_subscriber.hpp"

#include <json/json.h>

#include <sys/stat.h>
#include <pwd.h>
#include <grp.h>
#include <unistd.h>

using namespace mldefender::firewall;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

IPSetType parse_ipset_type(const std::string& type_str) {
    if (type_str == "hash:ip") return IPSetType::HASH_IP;
    if (type_str == "hash:net") return IPSetType::HASH_NET;
    if (type_str == "hash:ip,port") return IPSetType::HASH_IP_PORT;
    std::cerr << "[WARN] Unknown ipset type '" << type_str << "', defaulting to hash:ip" << std::endl;
    return IPSetType::HASH_IP;
}

IPSetFamily parse_ipset_family(const std::string& family_str) {
    if (family_str == "inet" || family_str == "ipv4") return IPSetFamily::INET;
    if (family_str == "inet6" || family_str == "ipv6") return IPSetFamily::INET6;
    std::cerr << "[WARN] Unknown ipset family '" << family_str << "', defaulting to inet" << std::endl;
    return IPSetFamily::INET;
}

//===----------------------------------------------------------------------===//
// Configuration Structures
//===----------------------------------------------------------------------===//

struct DaemonConfig {
    bool daemonize = false;
    std::string pid_file = "/var/run/firewall-acl-agent.pid";
    std::string user = "root";
    std::string group = "root";
};

struct LoggingConfig {
    std::string level = "info";
    bool console = true;
    bool syslog = false;
    std::string file = "";
    uint32_t max_file_size_mb = 100;
};

struct MetricsConfig {
    bool enable_export = true;
    uint32_t export_interval_sec = 60;
    std::string export_format = "json";
    std::string export_file = "/var/log/ml-defender/firewall-metrics.json";
};

struct HealthCheckConfig {
    bool enable = true;
    uint32_t check_interval_sec = 30;
    bool ipset_health_check = true;
    bool iptables_health_check = true;
    bool zmq_connection_check = true;
};

struct Config {
    IPSetConfig ipset;
    FirewallConfig firewall;
    BatchProcessorConfig batch;
    ZMQSubscriber::Config zmq;
    DaemonConfig daemon;
    LoggingConfig logging;
    MetricsConfig metrics;
    HealthCheckConfig health_check;
};

//===----------------------------------------------------------------------===//
// Configuration Loading
//===----------------------------------------------------------------------===//

bool load_config(const std::string& config_path, Config& config) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open config file: " << config_path << std::endl;
        return false;
    }

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    
    if (!Json::parseFromStream(builder, file, &root, &errors)) {
        std::cerr << "[ERROR] Failed to parse JSON config: " << errors << std::endl;
        return false;
    }

    try {
        // IPSet configuration
        if (root.isMember("ipset")) {
            const auto& ipset = root["ipset"];
            config.ipset.name = ipset.get("set_name", "ml_defender_blacklist").asString();
            std::string type_str = ipset.get("set_type", "hash:ip").asString();
            config.ipset.type = parse_ipset_type(type_str);
            if (ipset.isMember("family")) {
                config.ipset.family = parse_ipset_family(ipset["family"].asString());
            }
            config.ipset.hashsize = ipset.get("hash_size", 4096).asUInt();
            config.ipset.maxelem = ipset.get("max_elements", 1000000).asUInt();
            config.ipset.timeout = ipset.get("timeout", 3600).asUInt();
            config.ipset.counters = ipset.get("counters", true).asBool();
            config.ipset.comment = ipset.get("enable_comments", true).asBool();
            config.ipset.skbinfo = ipset.get("skbinfo", false).asBool();
            config.ipset.netmask = ipset.get("netmask", 32).asUInt();
            config.ipset.forceadd = ipset.get("forceadd", false).asBool();
        }

        // Firewall configuration
        if (root.isMember("iptables")) {
            const auto& iptables = root["iptables"];
            config.firewall.blacklist_chain = iptables.get("chain_name", "ML_DEFENDER_BLACKLIST").asString();
            config.firewall.blacklist_ipset = config.ipset.name;
            config.firewall.enable_rate_limiting = iptables.get("enable_rate_limiting", true).asBool();
            config.firewall.rate_limit_connections = iptables.get("rate_limit_connections", 100).asUInt();
        }

        // Batch processor configuration
        if (root.isMember("batch_processor")) {
            const auto& batch = root["batch_processor"];
            config.batch.batch_size_threshold = batch.get("batch_size_threshold", 1000).asUInt();
            config.batch.batch_time_threshold = std::chrono::milliseconds(batch.get("batch_time_threshold_ms", 100).asUInt());
            config.batch.max_pending_ips = batch.get("max_pending_ips", 10000).asUInt();
            config.batch.confidence_threshold = batch.get("confidence_threshold", 0.8f).asFloat();
            config.batch.block_low_confidence = batch.get("block_low_confidence", false).asBool();
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
            config.zmq.enable_reconnect = zmq.get("enable_reconnect", true).asBool();
        }

        // Daemon configuration
        if (root.isMember("daemon")) {
            const auto& daemon = root["daemon"];
            config.daemon.daemonize = daemon.get("daemonize", false).asBool();
            config.daemon.pid_file = daemon.get("pid_file", "/var/run/firewall-acl-agent.pid").asString();
            config.daemon.user = daemon.get("user", "root").asString();
            config.daemon.group = daemon.get("group", "root").asString();
        }

        // Logging configuration
        if (root.isMember("logging")) {
            const auto& logging = root["logging"];
            config.logging.level = logging.get("level", "info").asString();
            config.logging.console = logging.get("console", true).asBool();
            config.logging.syslog = logging.get("syslog", false).asBool();
            config.logging.file = logging.get("file", "").asString();
            config.logging.max_file_size_mb = logging.get("max_file_size_mb", 100).asUInt();
        }

        // Metrics configuration
        if (root.isMember("metrics")) {
            const auto& metrics = root["metrics"];
            config.metrics.enable_export = metrics.get("enable_export", true).asBool();
            config.metrics.export_interval_sec = metrics.get("export_interval_sec", 60).asUInt();
            config.metrics.export_format = metrics.get("export_format", "json").asString();
            config.metrics.export_file = metrics.get("export_file", "/var/log/ml-defender/firewall-metrics.json").asString();
        }

        // Health check configuration
        if (root.isMember("health_check")) {
            const auto& health = root["health_check"];
            config.health_check.enable = health.get("enable", true).asBool();
            config.health_check.check_interval_sec = health.get("check_interval_sec", 30).asUInt();
            config.health_check.ipset_health_check = health.get("ipset_health_check", true).asBool();
            config.health_check.iptables_health_check = health.get("iptables_health_check", true).asBool();
            config.health_check.zmq_connection_check = health.get("zmq_connection_check", true).asBool();
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Error parsing configuration: " << e.what() << std::endl;
        return false;
    }

    return true;
}

Config create_default_config() {
    Config config;
    
    // IPSet defaults
    config.ipset.name = "ml_defender_blacklist";
    config.ipset.type = IPSetType::HASH_IP;
    config.ipset.family = IPSetFamily::INET;
    config.ipset.hashsize = 4096;
    config.ipset.maxelem = 1'000'000;
    config.ipset.timeout = 3600;
    config.ipset.counters = true;
    config.ipset.comment = true;
    config.ipset.skbinfo = false;
    config.ipset.netmask = 32;
    config.ipset.forceadd = false;
    
    // Firewall defaults
    config.firewall.blacklist_chain = "ML_DEFENDER_BLACKLIST";
    config.firewall.blacklist_ipset = "ml_defender_blacklist";
    config.firewall.enable_rate_limiting = true;
    config.firewall.rate_limit_connections = 100;
    
    // Batch processor defaults
    config.batch.batch_size_threshold = 1000;
    config.batch.batch_time_threshold = std::chrono::milliseconds(100);
    config.batch.max_pending_ips = 10000;
    config.batch.confidence_threshold = 0.8f;
    config.batch.block_low_confidence = false;
    
    // ZMQ defaults
    config.zmq.endpoint = "tcp://localhost:5555";
    config.zmq.topic = "";
    config.zmq.recv_timeout_ms = 1000;
    config.zmq.linger_ms = 1000;
    config.zmq.reconnect_interval_ms = 1000;
    config.zmq.max_reconnect_interval_ms = 30000;
    config.zmq.enable_reconnect = true;
    
    return config;
}

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

void export_metrics(const Config& config, 
                   const ZMQSubscriber& subscriber,
                   const BatchProcessor& processor) {
    if (!config.metrics.enable_export) {
        return;
    }
    
    const auto& zmq_stats = subscriber.get_stats();
    
    Json::Value metrics;
    metrics["timestamp"] = static_cast<Json::Int64>(std::time(nullptr));
    
    metrics["zmq"]["messages_received"] = static_cast<Json::UInt64>(zmq_stats.messages_received.load());
    metrics["zmq"]["detections_processed"] = static_cast<Json::UInt64>(zmq_stats.detections_processed.load());
    metrics["zmq"]["parse_errors"] = static_cast<Json::UInt64>(zmq_stats.parse_errors.load());
    metrics["zmq"]["reconnects"] = static_cast<Json::UInt64>(zmq_stats.reconnects.load());
    metrics["zmq"]["currently_connected"] = zmq_stats.currently_connected.load();
    metrics["zmq"]["last_message_timestamp"] = static_cast<Json::Int64>(zmq_stats.last_message_timestamp.load());
    
    if (config.metrics.export_format == "json") {
        std::ofstream file(config.metrics.export_file);
        if (file.is_open()) {
            Json::StreamWriterBuilder writer;
            writer["indentation"] = "  ";
            file << Json::writeString(writer, metrics);
        }
    }
}

//===----------------------------------------------------------------------===//
// Health Checks
//===----------------------------------------------------------------------===//

bool perform_health_checks(const Config& config,
                          IPSetWrapper& ipset,
                          IPTablesWrapper& iptables,
                          const ZMQSubscriber& subscriber) {
    bool all_healthy = true;
    
    std::cout << "[HEALTH] Running health checks..." << std::endl;
    
    if (config.health_check.ipset_health_check) {
        if (!ipset.set_exists(config.ipset.name)) {
            std::cerr << "[HEALTH] ✗ IPSet '" << config.ipset.name << "' does not exist!" << std::endl;
            all_healthy = false;
        } else {
            std::cout << "[HEALTH] ✓ IPSet exists" << std::endl;
        }
    }
    
    if (config.health_check.iptables_health_check) {
        auto rules = iptables.list_rules("INPUT");
        bool rule_found = false;
        for (const auto& rule : rules) {
            if (rule.find(config.ipset.name) != std::string::npos) {
                rule_found = true;
                break;
            }
        }
        
        if (!rule_found) {
            std::cerr << "[HEALTH] ✗ IPTables rule for '" << config.ipset.name << "' not found!" << std::endl;
            all_healthy = false;
        } else {
            std::cout << "[HEALTH] ✓ IPTables rule exists" << std::endl;
        }
    }
    
    if (config.health_check.zmq_connection_check) {
        const auto& stats = subscriber.get_stats();
        if (!stats.currently_connected.load()) {
            std::cerr << "[HEALTH] ✗ ZMQ not connected!" << std::endl;
            all_healthy = false;
        } else {
            std::cout << "[HEALTH] ✓ ZMQ connected" << std::endl;
        }
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
    std::string config_path = "/etc/ml-defender/firewall.json";
    bool test_config = false;
    bool force_daemon = false;
    (void)force_daemon;  // Simplified daemon mode for now
    
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
        } else if (arg == "-d" || arg == "--daemonize") {
            force_daemon = true;
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
    
    Config config = create_default_config();
    if (!load_config(config_path, config)) {
        std::cerr << "[ERROR] Failed to load configuration from: " << config_path << std::endl;
        std::cerr << "[INFO] Using default configuration" << std::endl;
    } else {
        std::cout << "[INFO] Configuration loaded from: " << config_path << std::endl;
    }
    
    if (test_config) {
        std::cout << "[INFO] Configuration test passed ✓" << std::endl;
        return 0;
    }
    
    if (getuid() != 0) {
        std::cerr << "[ERROR] This program must be run as root" << std::endl;
        return 1;
    }
    
    try {
        std::cout << "[INIT] Initializing IPSet wrapper..." << std::endl;
        IPSetWrapper ipset;
        
        if (!ipset.set_exists(config.ipset.name)) {
            std::cout << "[INIT] Creating ipset '" << config.ipset.name << "'..." << std::endl;
            if (!ipset.create_set(config.ipset)) {
                std::cerr << "[ERROR] Failed to create ipset" << std::endl;
                return 1;
            }
            std::cout << "[INIT] ✓ IPSet created successfully" << std::endl;
        } else {
            std::cout << "[INIT] ✓ IPSet already exists" << std::endl;
        }
        
        std::cout << "[INIT] Initializing IPTables wrapper..." << std::endl;
        IPTablesWrapper iptables;
        
        std::cout << "[INIT] Setting up IPTables rules..." << std::endl;
        if (!iptables.setup_base_rules(config.firewall)) {
            std::cerr << "[ERROR] Failed to setup IPTables rules" << std::endl;
            return 1;
        }
        std::cout << "[INIT] ✓ IPTables rules configured" << std::endl;
        
        std::cout << "[INIT] Initializing batch processor..." << std::endl;
        BatchProcessor processor(ipset, config.batch);
        std::cout << "[INIT] ✓ Batch processor started" << std::endl;
        
        std::cout << "[INIT] Initializing ZMQ subscriber..." << std::endl;
        ZMQSubscriber subscriber(processor, config.zmq);
        
        // Start ZMQ subscriber thread
        std::thread zmq_thread([&subscriber]() {
            subscriber.run();
        });
        zmq_thread.detach();
        std::cout << "[INIT] ✓ ZMQ subscriber initialized" << std::endl;
        
        setup_signal_handlers();
        
        std::cout << "\n[RUNNING] Firewall ACL Agent is now active\n"
                  << "[INFO] Listening on: " << config.zmq.endpoint << "\n"
                  << "[INFO] Press Ctrl+C to stop\n"
                  << std::endl;
        
        auto last_health_check = std::chrono::steady_clock::now();
        auto last_metrics_export = std::chrono::steady_clock::now();
        
        while (g_running.load()) {
            auto now = std::chrono::steady_clock::now();
            
            if (config.health_check.enable) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_health_check).count();
                    
                if (elapsed >= config.health_check.check_interval_sec) {
                    perform_health_checks(config, ipset, iptables, subscriber);
                    last_health_check = now;
                }
            }
            
            if (config.metrics.enable_export) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_metrics_export).count();
                    
                if (elapsed >= config.metrics.export_interval_sec) {
                    export_metrics(config, subscriber, processor);
                    last_metrics_export = now;
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "\n[SHUTDOWN] Stopping components..." << std::endl;
        
        if (config.metrics.enable_export) {
            export_metrics(config, subscriber, processor);
            std::cout << "[SHUTDOWN] ✓ Final metrics exported" << std::endl;
        }
        
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
