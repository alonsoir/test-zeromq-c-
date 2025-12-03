// main.cpp - Enhanced Sniffer v3.2 with Hybrid Filtering
// FECHA: 31 de Octubre de 2025
// FUNCIONALIDAD: Sistema de filtros híbridos con BPF Maps
// Stats handled by RingBufferConsumer internally
// sniffer/src/userspace/main.cpp

#include "config_types.h"
#include "main.h"
#include "network_security.pb.h"
#include "config_manager.hpp"
#include "zmq_pool_manager.hpp"
#include "ebpf_loader.hpp"
#include "ring_consumer.hpp"
#include "thread_manager.hpp"
#include "bpf_map_manager.h"

// Sistema y captura de red
#include <json/json.h>
#include <fstream>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <netinet/ether.h>
#include <linux/if_packet.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <errno.h>
#include <csignal>
#include <getopt.h>
#include <iomanip>
#include <random>
#include <fstream>
#include "feature_logger.hpp"
#include "dual_nic_manager.hpp"

FeatureLogger::VerbosityLevel g_verbosity = FeatureLogger::VerbosityLevel::NONE;

// ============================================================================
// FILTER CONFIGURATION PARSING FUNCTIONS
// ============================================================================

bool parse_filter_config(const Json::Value& root, FilterConfiguration& filter_config) {
    if (!root.isMember("capture") || !root["capture"].isMember("filter")) {
        std::cerr << "⚠️  No filter configuration found in JSON, using defaults" << std::endl;
        filter_config.mode = "blacklist";
        filter_config.default_action = "capture";
        return true;
    }

    const Json::Value& filter = root["capture"]["filter"];

    // Parse mode
    if (filter.isMember("mode")) {
        filter_config.mode = filter["mode"].asString();
        if (filter_config.mode != "blacklist" &&
            filter_config.mode != "whitelist" &&
            filter_config.mode != "hybrid") {
            std::cerr << "❌ Invalid filter mode: " << filter_config.mode << std::endl;
            return false;
        }
    } else {
        filter_config.mode = "hybrid";
    }

    // Parse excluded_ports
    if (filter.isMember("excluded_ports") && filter["excluded_ports"].isArray()) {
        for (const auto& port : filter["excluded_ports"]) {
            if (port.isInt()) {
                uint16_t port_num = static_cast<uint16_t>(port.asUInt());
                if (port_num > 0) {  // Quitamos check redundante de <= 65535
                    filter_config.excluded_ports.push_back(port_num);
                }
            }
        }
    }

    // Parse included_ports
    if (filter.isMember("included_ports") && filter["included_ports"].isArray()) {
        for (const auto& port : filter["included_ports"]) {
            if (port.isInt()) {
                uint16_t port_num = static_cast<uint16_t>(port.asUInt());
                if (port_num > 0) {  // Quitamos check redundante de <= 65535
                    filter_config.included_ports.push_back(port_num);
                }
            }
        }
    }

    // Parse default_action
    if (filter.isMember("default_action")) {
        filter_config.default_action = filter["default_action"].asString();
        if (filter_config.default_action != "capture" &&
            filter_config.default_action != "drop") {
            std::cerr << "❌ Invalid default_action" << std::endl;
            return false;
        }
    } else {
        filter_config.default_action = "capture";
    }

    return true;
}

bool validate_filter_mode(const FilterConfiguration& config) {
    if (config.mode == "blacklist") {
        if (config.excluded_ports.empty()) {
            std::cerr << "⚠️  Warning: blacklist mode with no excluded_ports" << std::endl;
        }
    } else if (config.mode == "whitelist") {
        if (config.included_ports.empty()) {
            std::cerr << "❌ Error: whitelist mode requires at least one included_port" << std::endl;
            return false;
        }
    } else if (config.mode == "hybrid") {
        if (config.excluded_ports.empty() && config.included_ports.empty()) {
            std::cerr << "⚠️  Warning: hybrid mode with no filters" << std::endl;
        }
    }
    return true;
}

// ============================================================================
// USAGE AND HELP
// ============================================================================

void print_usage(const char* program_name) {
    std::cout << "╔════════════════════════════════════════════╗\n";
    std::cout << "║   Enhanced Sniffer v3.2 - Hybrid Filters  ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n";
    std::cout << "Compilado: " << __DATE__ << " " << __TIME__ << "\n";
    std::cout << "Modo: JSON + BPF Maps dynamic filtering\n\n";

    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -v, --verbose           Incrementar nivel de verbosity (usar hasta 3 veces)\n";
    std::cout << "                          -v   : Resumen básico por paquete\n";
    std::cout << "                          -vv  : Features agrupadas por categoría\n";
    std::cout << "                          -vvv : Todas las features con detalle completo\n";
    std::cout << "  -h, --help              Mostrar ayuda\n";
    std::cout << "  -c, --config FILE       Archivo JSON (OBLIGATORIO)\n";
    std::cout << "  -i, --interface IFACE   Override interface\n";
    std::cout << "  -p, --profile PROFILE   Override perfil (lab/cloud/bare_metal)\n";
    std::cout << "\nFilter Configuration (in JSON):\n";
    std::cout << "  \"filter\": {\n";
    std::cout << "    \"mode\": \"hybrid\",\n";
    std::cout << "    \"excluded_ports\": [22, 4444, 8080],\n";
    std::cout << "    \"included_ports\": [8000],\n";
    std::cout << "    \"default_action\": \"capture\"\n";
    std::cout << "  }\n";
    std::cout << "\nModes: blacklist, whitelist, hybrid\n";
    std::cout << "Precedence: included_ports > excluded_ports > default_action\n";
}

// ============================================================================
// GLOBAL VARIABLES (existentes, mantener)
// ============================================================================

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

// Atomic flag para terminación limpia
std::atomic<bool> g_running{true};

// Configuración global
StrictSnifferConfig g_config;

// Punteros a componentes principales
sniffer::EbpfLoader* ebpf_loader_ptr = nullptr;
sniffer::RingBufferConsumer* ring_consumer_ptr = nullptr;
std::shared_ptr<sniffer::ThreadManager> thread_manager = nullptr;

// ============================================================================
// SIGNAL HANDLERS
// ============================================================================

void signal_handler(int signum) {
    std::cout << "\n[Signal] Received signal " << signum << " - initiating graceful shutdown..." << std::endl;
    g_running = false;
}

void setup_signal_handlers() {
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    std::cout << "✅ Signal handlers configured (SIGINT, SIGTERM)" << std::endl;
}

// ============================================================================
// MAIN FUNCTION - REFACTORIZADO CON INTEGRACIÓN DE FILTROS
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Enhanced Sniffer v3.2 - Hybrid Filtering System       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "Compilado: " << __DATE__ << " " << __TIME__ << "\n\n";

    // ============================================================================
    // PARSE COMMAND LINE ARGUMENTS
    // ============================================================================

    std::string config_path;
    std::string override_interface;
    std::string override_profile;

    // Parse command line arguments
    int opt;
    int option_index = 0;

    static struct option long_options[] = {
        {"config",    required_argument, 0, 'c'},
        {"interface", required_argument, 0, 'i'},
        {"profile",   required_argument, 0, 'p'},
        {"verbose",   no_argument,       0, 'v'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "c:i:p:vh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'c':
                config_path = optarg;
                break;
            case 'i':
                override_interface = optarg;
                break;
            case 'p':
                override_profile = optarg;
                break;
            case 'v':
                // Increment verbosity level
                if (g_verbosity == FeatureLogger::VerbosityLevel::NONE) {
                    g_verbosity = FeatureLogger::VerbosityLevel::BASIC;
                } else if (g_verbosity == FeatureLogger::VerbosityLevel::BASIC) {
                    g_verbosity = FeatureLogger::VerbosityLevel::GROUPED;
                } else if (g_verbosity == FeatureLogger::VerbosityLevel::GROUPED) {
                    g_verbosity = FeatureLogger::VerbosityLevel::DETAILED;
                }
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (config_path.empty()) {
        std::cerr << "❌ Error: Config file is required (-c/--config)\n";
        print_usage(argv[0]);
        return 1;
    }

    // ============================================================================
    // SETUP SIGNAL HANDLERS
    // ============================================================================

    setup_signal_handlers();

    // ============================================================================
    // INITIALIZE PROTOBUF
    // ============================================================================

    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::cout << "✅ Protobuf version: " << google::protobuf::internal::VersionString(GOOGLE_PROTOBUF_VERSION) << std::endl;

    try {
        // ============================================================================
        // LOAD JSON CONFIGURATION
        // ============================================================================

        std::cout << "\n[Config] Loading configuration from: " << config_path << std::endl;

        if (!strict_load_json_config(config_path, g_config, false)) {
            std::cerr << "❌ Failed to load configuration" << std::endl;
            return 1;
        }

        std::cout << "✅ Configuration loaded successfully" << std::endl;

        // ============================================================================
        // PARSE FILTER CONFIGURATION (v3.2)
        // ============================================================================
        std::cout << "\n[Filter] Parsing filter configuration..." << std::endl;

        // Re-parse JSON to get root object
        std::ifstream filter_config_file(config_path);
        if (!filter_config_file.is_open()) {
            std::cerr << "❌ Failed to re-open config file for filter parsing" << std::endl;
            return 1;
        }

        Json::Value json_root;
        Json::CharReaderBuilder reader_builder;
        std::string json_errors;

        if (!Json::parseFromStream(reader_builder, filter_config_file, &json_root, &json_errors)) {
            std::cerr << "❌ Failed to parse JSON for filters: " << json_errors << std::endl;
            filter_config_file.close();
            return 1;
        }
        filter_config_file.close();

        FilterConfiguration filter_config;
        if (!parse_filter_config(json_root, filter_config)) {
            std::cerr << "❌ Failed to parse filter configuration" << std::endl;
            return 1;
        }

        if (!validate_filter_mode(filter_config)) {
            std::cerr << "❌ Invalid filter configuration" << std::endl;
            return 1;
        }

        // Display filter configuration
        std::cout << "\n📋 Filter Configuration:" << std::endl;
        std::cout << "  Mode: " << filter_config.mode << std::endl;
        std::cout << "  Excluded ports: " << filter_config.excluded_ports.size() << std::endl;
        std::cout << "  Included ports: " << filter_config.included_ports.size() << std::endl;
        std::cout << "  Default action: " << filter_config.default_action << std::endl;

        /// ============================================================================
        // LOAD AND ATTACH EBPF PROGRAM
        // ============================================================================

        std::cout << "\n[eBPF] Loading and attaching eBPF program..." << std::endl;

        ebpf_loader_ptr = new sniffer::EbpfLoader();
        auto& ebpf_loader = *ebpf_loader_ptr;

        if (!ebpf_loader.load_program(g_config.kernel_space.ebpf_program)) {
            std::cerr << "❌ Failed to load eBPF program: " << g_config.kernel_space.ebpf_program << std::endl;
            return 1;
        }

        // ============================================================================
        // >>> NUEVO: DUAL-NIC DEPLOYMENT CONFIGURATION (Phase 1, Day 7)
        // ============================================================================
        std::cout << "\n[Dual-NIC] Configuring deployment mode..." << std::endl;

        try {
            sniffer::DualNICManager dual_nic_manager(json_root);
            dual_nic_manager.initialize();

            // Configure interface_configs BPF map
            int interface_configs_fd = ebpf_loader.get_interface_configs_fd();
            if (interface_configs_fd >= 0) {
                dual_nic_manager.configure_bpf_map(interface_configs_fd);

                // Setup network if needed (gateway mode)
                if (dual_nic_manager.is_dual_mode()) {
                    dual_nic_manager.enable_ip_forwarding();
                    dual_nic_manager.setup_nat_rules();
                }
            } else {
                std::cout << "[WARNING] interface_configs map not found - using legacy single-interface mode" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[WARNING] Dual-NIC configuration failed: " << e.what() << std::endl;
            std::cerr << "          Falling back to legacy single-interface mode" << std::endl;
        }

        std::cout << "✅ Deployment configuration complete\n" << std::endl;
        // ============================================================================
        // >>> FIN DUAL-NIC CONFIGURATION
        // ============================================================================

        // Usar capture_interface en lugar de g_config.capture.interface
        // Use attach method based on capture mode
        bool attached = false;
        if (g_config.capture_mode == "ebpf_skb") {
            std::cout << "[INFO] Using SKB mode (TC-based eBPF)" << std::endl;
            attached = ebpf_loader.attach_skb(g_config.capture_interface);
        } else {
            std::cout << "[INFO] Using XDP mode (native XDP)" << std::endl;
            attached = ebpf_loader.attach_xdp(g_config.capture_interface);
        }

        if (!attached) {
            std::cerr << "❌ Failed to attach to interface: " << g_config.capture_interface << std::endl;
            return 1;
        }

        std::cout << "✅ eBPF program attached to interface: " << g_config.capture_interface << std::endl;

        int ring_fd = ebpf_loader.get_ringbuf_fd();
        if (ring_fd < 0) {
            std::cerr << "❌ Failed to get ring buffer FD" << std::endl;
            return 1;
        }

        std::cout << "✅ Ring buffer FD: " << ring_fd << std::endl;

        // ============================================================================
        // LOAD FILTER CONFIGURATION TO BPF MAPS (v3.2)
        // ============================================================================
        std::cout << "\n[BPF Maps] Loading filter configuration to kernel..." << std::endl;

        sniffer::BPFMapManager bpf_map_manager;

        if (!bpf_map_manager.load_filter_config_with_fds(
            ebpf_loader.get_excluded_ports_fd(),
            ebpf_loader.get_included_ports_fd(),
            ebpf_loader.get_filter_settings_fd(),
            filter_config.excluded_ports,
            filter_config.included_ports,
            filter_config.get_default_action_value()
        )) {
            std::cerr << "❌ Failed to load filter configuration to BPF maps" << std::endl;
            std::cerr << "   Ensure the eBPF program defines the required maps:" << std::endl;
            std::cerr << "   - excluded_ports" << std::endl;
            std::cerr << "   - included_ports" << std::endl;
            std::cerr << "   - filter_settings" << std::endl;
            return 1;
        }

        std::cout << "✅ Filter configuration loaded to kernel space" << std::endl;
        std::cout << "✅ Sniffer ready with active filtering" << std::endl;

        // ============================================================================
        // INITIALIZE THREAD MANAGER
        // ============================================================================

        std::cout << "\n[ThreadManager] Initializing thread manager..." << std::endl;

        sniffer::ThreadingConfig threading_config;
        threading_config.ring_consumer_threads = g_config.threading.ring_consumer_threads;
        threading_config.feature_processor_threads = g_config.threading.feature_processor_threads;
        threading_config.zmq_sender_threads = g_config.threading.zmq_sender_threads;
        threading_config.statistics_collector_threads = g_config.threading.statistics_collector_threads;
        threading_config.total_worker_threads = g_config.threading.total_worker_threads;
        threading_config.cpu_affinity.enabled = g_config.threading.cpu_affinity_enabled;
        threading_config.cpu_affinity.ring_consumers = g_config.threading.ring_consumers_affinity;
        threading_config.cpu_affinity.processors = g_config.threading.processors_affinity;
        threading_config.cpu_affinity.zmq_senders = g_config.threading.zmq_senders_affinity;
        threading_config.cpu_affinity.statistics = g_config.threading.statistics_affinity;

        thread_manager = std::make_shared<sniffer::ThreadManager>(threading_config);

        if (!thread_manager->start()) {
            std::cerr << "❌ Failed to start thread manager" << std::endl;
            return 1;
        }
        std::cout << "✅ Thread manager started" << std::endl;

        // ============================================================================
        // INITIALIZE RING BUFFER CONSUMER
        // ============================================================================

        std::cout << "\n[RingBuffer] Initializing RingBufferConsumer..." << std::endl;

        sniffer::SnifferConfig sniffer_config;
        sniffer_config.node_id = g_config.node_id;
        sniffer_config.cluster_name = g_config.cluster_name;

        // Mapear buffers
        sniffer_config.buffers.ring_buffer_entries = g_config.buffers.ring_buffer_entries;
        sniffer_config.buffers.user_processing_queue_depth = g_config.buffers.user_processing_queue_depth;
        sniffer_config.buffers.protobuf_serialize_buffer_size = g_config.buffers.protobuf_serialize_buffer_size;
        sniffer_config.buffers.zmq_send_buffer_size = g_config.buffers.zmq_send_buffer_size;
        sniffer_config.buffers.flow_state_buffer_entries = g_config.buffers.flow_state_buffer_entries;
        sniffer_config.buffers.statistics_buffer_entries = g_config.buffers.statistics_buffer_entries;
        sniffer_config.buffers.batch_processing_size = g_config.buffers.batch_processing_size;

        // Mapear threading
        sniffer_config.threading.ring_consumer_threads = g_config.threading.ring_consumer_threads;
        sniffer_config.threading.feature_processor_threads = g_config.threading.feature_processor_threads;
        sniffer_config.threading.zmq_sender_threads = g_config.threading.zmq_sender_threads;
        sniffer_config.threading.statistics_collector_threads = g_config.threading.statistics_collector_threads;
        sniffer_config.threading.total_worker_threads = g_config.threading.total_worker_threads;

        // Mapear CPU affinity
        sniffer_config.threading.cpu_affinity.enabled = g_config.threading.cpu_affinity_enabled;
        sniffer_config.threading.cpu_affinity.ring_consumers = g_config.threading.ring_consumers_affinity;
        sniffer_config.threading.cpu_affinity.processors = g_config.threading.processors_affinity;
        sniffer_config.threading.cpu_affinity.zmq_senders = g_config.threading.zmq_senders_affinity;
        sniffer_config.threading.cpu_affinity.statistics = g_config.threading.statistics_affinity;

        // Mapear thread priorities
        sniffer_config.threading.thread_priorities["ring_consumers"] = g_config.threading.ring_consumers_priority;
        sniffer_config.threading.thread_priorities["processors"] = g_config.threading.processors_priority;
        sniffer_config.threading.thread_priorities["zmq_senders"] = g_config.threading.zmq_senders_priority;

        // Mapear ZMQ
        sniffer_config.zmq.worker_threads = g_config.zmq.worker_threads;
        sniffer_config.zmq.io_thread_pools = g_config.zmq.io_thread_pools;

        // Mapear socket pools
        sniffer_config.zmq.socket_pools.push_sockets = g_config.zmq.push_sockets;
        sniffer_config.zmq.socket_pools.load_balancing = g_config.zmq.load_balancing;
        sniffer_config.zmq.socket_pools.failover_enabled = g_config.zmq.failover_enabled;

        // Mapear queue management
        sniffer_config.zmq.queue_management.internal_queues = g_config.zmq.internal_queues;
        sniffer_config.zmq.queue_management.queue_size = g_config.zmq.queue_size;
        sniffer_config.zmq.queue_management.queue_timeout_ms = g_config.zmq.queue_timeout_ms;
        sniffer_config.zmq.queue_management.overflow_policy = g_config.zmq.overflow_policy;

        // Mapear connection settings
        sniffer_config.zmq.connection_settings.sndhwm = g_config.zmq.sndhwm;
        sniffer_config.zmq.connection_settings.linger_ms = g_config.zmq.linger_ms;
        sniffer_config.zmq.connection_settings.send_timeout_ms = g_config.zmq.send_timeout_ms;
        sniffer_config.zmq.connection_settings.rcvhwm = g_config.zmq.rcvhwm;
        sniffer_config.zmq.connection_settings.recv_timeout_ms = g_config.zmq.recv_timeout_ms;
        sniffer_config.zmq.connection_settings.tcp_keepalive = g_config.zmq.tcp_keepalive;
        sniffer_config.zmq.connection_settings.sndbuf = g_config.zmq.sndbuf;
        sniffer_config.zmq.connection_settings.rcvbuf = g_config.zmq.rcvbuf;
        sniffer_config.zmq.connection_settings.reconnect_interval_ms = g_config.zmq.reconnect_interval_ms;
        sniffer_config.zmq.connection_settings.max_reconnect_attempts = g_config.zmq.max_reconnect_attempts;

        // Mapear batch processing
        sniffer_config.zmq.batch_processing.enabled = g_config.zmq.batch_enabled;
        sniffer_config.zmq.batch_processing.batch_size = g_config.zmq.batch_size;
        sniffer_config.zmq.batch_processing.batch_timeout_ms = g_config.zmq.batch_timeout_ms;
        sniffer_config.zmq.batch_processing.max_batches_queued = g_config.zmq.max_batches_queued;

        // Mapear network
        sniffer_config.network.output_socket.address = g_config.network_output.address;
        sniffer_config.network.output_socket.port = g_config.network_output.port;
        sniffer_config.network.output_socket.mode = g_config.network_output.mode;
        sniffer_config.network.output_socket.socket_type = g_config.network_output.socket_type;
        sniffer_config.network.output_socket.high_water_mark = g_config.network_output.high_water_mark;

        // Mapear transport/compression
        sniffer_config.transport.compression.enabled = g_config.compression.enabled;
        sniffer_config.transport.compression.algorithm = g_config.compression.algorithm;
        sniffer_config.transport.compression.level = g_config.compression.level;
        sniffer_config.transport.compression.min_compress_size = g_config.compression.min_compress_size;
        sniffer_config.transport.compression.compression_ratio_threshold = g_config.compression.compression_ratio_threshold;
        sniffer_config.transport.compression.adaptive_compression = g_config.compression.adaptive_compression;

        ring_consumer_ptr = new sniffer::RingBufferConsumer(sniffer_config);
        auto& ring_consumer = *ring_consumer_ptr;

        // Configure stats interval from monitoring config
        ring_consumer.set_stats_interval(g_config.monitoring.stats_interval_seconds);

        if (!ring_consumer.initialize(ring_fd, thread_manager)) {
            std::cerr << "❌ Failed to initialize RingBufferConsumer" << std::endl;
            return 1;
        }

        if (!ring_consumer.start()) {
            std::cerr << "❌ Failed to start RingBufferConsumer" << std::endl;
            return 1;
        }

        std::cout << "✅ RingBufferConsumer started - capturing REAL packets from kernel" << std::endl;
        std::cout << "✅ Statistics will be displayed every " << g_config.monitoring.stats_interval_seconds
                  << " seconds" << std::endl;

        // ============================================================================
        // MAIN LOOP - WAIT FOR TERMINATION SIGNAL
        // ============================================================================

        std::cout << "\n🚀 Sniffer running with hybrid filtering enabled" << std::endl;
        std::cout << "   Press Ctrl+C to stop\n" << std::endl;

        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // ============================================================================
        // CLEANUP (SIN MEMORY LEAKS)
        // ============================================================================

        std::cout << "\n[Cleanup] Stopping components..." << std::endl;

        // Stop RingBufferConsumer first
        if (ring_consumer_ptr) {
            ring_consumer_ptr->stop();
            delete ring_consumer_ptr;
            ring_consumer_ptr = nullptr;
        }

        // Stop ThreadManager
        if (thread_manager) {
            thread_manager->stop();
            thread_manager.reset();  // smart_ptr cleanup
        }

        // Cleanup eBPF loader
        if (ebpf_loader_ptr) {
            delete ebpf_loader_ptr;
            ebpf_loader_ptr = nullptr;
        }

        std::cout << "✅ All components stopped cleanly" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ FATAL ERROR: " << e.what() << "\n";

        // Emergency cleanup
        if (ring_consumer_ptr) {
            delete ring_consumer_ptr;
            ring_consumer_ptr = nullptr;
        }

        if (ebpf_loader_ptr) {
            delete ebpf_loader_ptr;
            ebpf_loader_ptr = nullptr;
        }

        return 1;
    }

    google::protobuf::ShutdownProtobufLibrary();
    std::cout << "\n👋 Sniffer stopped correctly\n";
    return 0;
}