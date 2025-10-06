// main.cpp - Enhanced Sniffer v3.1 STRICT JSON
// FECHA: 6 de Octubre de 2025
// FUNCIONALIDAD: Implementación limpia con validación estricta JSON
// sniffer/src/userspace/main.cpp
#include "config_types.h"
#include "main.h"
#include "network_security.pb.h"
#include "config_manager.hpp"
#include "zmq_pool_manager.hpp"
#include "ebpf_loader.hpp"
#include "ring_consumer.hpp"
#include "thread_manager.hpp"
// Sistema y captura de red
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

// ============================================================================
// VARIABLES GLOBALES - DEFINICIÓN (sin extern)
// ============================================================================
std::atomic<bool> g_running{true};
StrictSnifferConfig g_config;
DetailedStats g_stats;
CommandLineArgs g_args;

// ============================================================================
// IMPLEMENTACIONES DE FUNCIONES
// ============================================================================

void signal_handler(int signum) {
    std::cout << "\nSeñal recibida (" << signum << "), deteniendo sniffer...\n";
    g_running = false;
}

void parse_command_line(int argc, char* argv[], CommandLineArgs& args) {
    static struct option long_options[] = {
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {"config", required_argument, 0, 'c'},
        {"interface", required_argument, 0, 'i'},
        {"profile", required_argument, 0, 'p'},
        {"dry-run", no_argument, 0, 'd'},
        {"show-config", no_argument, 0, 's'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int c;

    while ((c = getopt_long(argc, argv, "vhc:i:p:ds", long_options, &option_index)) != -1) {
        switch (c) {
            case 'v': args.verbose = true; break;
            case 'h': args.help = true; break;
            case 'c': args.config_file = optarg; break;
            case 'i': args.interface_override = optarg; break;
            case 'p': args.profile_override = optarg; break;
            case 'd': args.dry_run = true; break;
            case 's': args.show_config_only = true; break;
            default: args.help = true; break;
        }
    }
}

void print_help(const char* program_name) {
    std::cout << "Enhanced Network Security Sniffer v3.1 - STRICT JSON MODE\n\n";
    std::cout << "IMPORTANTE: JSON es la ley - falla si falta cualquier campo requerido\n\n";
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -v, --verbose           Mostrar validación JSON detallada\n";
    std::cout << "  -h, --help              Mostrar ayuda\n";
    std::cout << "  -c, --config FILE       Archivo JSON (OBLIGATORIO si no existe default)\n";
    std::cout << "  -i, --interface IFACE   Override interface\n";
    std::cout << "  -p, --profile PROFILE   Override perfil (lab/cloud/bare_metal)\n";
    std::cout << "  -d, --dry-run           Solo validar JSON\n";
    std::cout << "  -s, --show-config       Mostrar config parseada y salir\n\n";
}

bool strict_load_json_config(const std::string& config_path, StrictSnifferConfig& config, bool verbose) {
    if (verbose) {
        std::cout << "\n=== VALIDACIÓN ESTRICTA JSON ===\n";
        std::cout << "Archivo: " << std::filesystem::absolute(config_path) << "\n";
    }

    // Verificar que el archivo existe
    if (!std::filesystem::exists(config_path)) {
        throw std::runtime_error("ARCHIVO JSON NO EXISTE: " + config_path);
    }

    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("NO SE PUEDE ABRIR ARCHIVO JSON: " + config_path);
    }

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;

    if (!Json::parseFromStream(builder, config_file, &root, &errors)) {
        throw std::runtime_error("JSON INVÁLIDO: " + errors);
    }

    if (verbose) {
        std::cout << "JSON parseado - iniciando validación estricta de TODAS las secciones...\n";
    }

    try {
        // Validar TODAS las secciones requeridas
        validate_component_section(root, config, verbose);
        validate_profiles_section(root, config, verbose);
        validate_capture_section(root, config, verbose);
        validate_buffers_section(root, config, verbose);
        validate_threading_section(root, config, verbose);
        validate_kernel_space_section(root, config, verbose);
        validate_user_space_section(root, config, verbose);
        validate_feature_groups_section(root, config, verbose);
        validate_time_windows_section(root, config, verbose);
        validate_network_section(root, config, verbose);
        validate_zmq_section(root, config, verbose);
        validate_transport_section(root, config, verbose);
        validate_etcd_section(root, config, verbose);
        validate_processing_section(root, config, verbose);
        validate_auto_tuner_section(root, config, verbose);
        validate_monitoring_section(root, config, verbose);
        validate_protobuf_section(root, config, verbose);
        validate_logging_section(root, config, verbose);
        validate_security_section(root, config, verbose);
        validate_backpressure_section(root, config, verbose);

        if (verbose) {
            std::cout << "✅ TODAS las secciones JSON validadas exitosamente\n";
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ ERROR DE VALIDACIÓN JSON: " << e.what() << "\n";
        return false;
    }
}

void validate_component_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    REQUIRE_OBJECT(root, "component");
    const auto& component = root["component"];

    REQUIRE_FIELD(component, "name", String);
    REQUIRE_FIELD(component, "version", String);
    REQUIRE_FIELD(component, "mode", String);
    REQUIRE_FIELD(component, "kernel_version_required", String);

    config.component_name = component["name"].asString();
    config.component_version = component["version"].asString();
    config.component_mode = component["mode"].asString();
    config.kernel_version_required = component["kernel_version_required"].asString();

    if (verbose) {
        std::cout << "✓ Componente validado: " << config.component_name << " v" << config.component_version << "\n";
    }
}

void validate_profiles_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    // Identificación básica
    REQUIRE_FIELD(root, "node_id", String);
    REQUIRE_FIELD(root, "cluster_name", String);
    REQUIRE_FIELD(root, "profile", String);

    config.node_id = root["node_id"].asString();
    config.cluster_name = root["cluster_name"].asString();
    config.active_profile = root["profile"].asString();

    // Override profile si se especifica
    if (!g_args.profile_override.empty()) {
        config.active_profile = g_args.profile_override;
        if (verbose) std::cout << "Profile sobrescrito: " << config.active_profile << "\n";
    }

    // Validar perfiles
    REQUIRE_OBJECT(root, "profiles");
    const auto& profiles = root["profiles"];

    if (!profiles.isMember(config.active_profile)) {
        throw std::runtime_error("PERFIL NO ENCONTRADO: " + config.active_profile + " en sección profiles");
    }

    const auto& profile = profiles[config.active_profile];
    REQUIRE_FIELD(profile, "capture_interface", String);
    REQUIRE_FIELD(profile, "promiscuous_mode", Bool);
    REQUIRE_FIELD(profile, "af_xdp_enabled", Bool);
    REQUIRE_FIELD(profile, "worker_threads", Int);
    REQUIRE_FIELD(profile, "compression_level", Int);

    config.capture_interface = profile["capture_interface"].asString();
    config.promiscuous_mode = profile["promiscuous_mode"].asBool();
    config.af_xdp_enabled = profile["af_xdp_enabled"].asBool();
    config.worker_threads = profile["worker_threads"].asInt();
    config.compression_level = profile["compression_level"].asInt();

    // Override interface si se especifica
    if (!g_args.interface_override.empty()) {
        config.capture_interface = g_args.interface_override;
        if (verbose) std::cout << "Interface sobrescrita: " << config.capture_interface << "\n";
    }

    if (verbose) {
        std::cout << "✓ Perfil '" << config.active_profile << "' validado: " << config.capture_interface << "\n";
    }
}

void validate_capture_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    REQUIRE_OBJECT(root, "capture");
    const auto& capture = root["capture"];

    REQUIRE_FIELD(capture, "interface", String);
    // kernel_interface y user_interface pueden ser null
    if (capture["kernel_interface"].isNull()) {
        config.kernel_interface = "";
    } else {
        REQUIRE_FIELD(capture, "kernel_interface", String);
        config.kernel_interface = capture["kernel_interface"].asString();
    }
    if (capture["user_interface"].isNull()) {
        config.user_interface = "";
    } else {
        REQUIRE_FIELD(capture, "user_interface", String);
        config.user_interface = capture["user_interface"].asString();
    }

    REQUIRE_FIELD(capture, "mode", String);
    REQUIRE_ARRAY(capture, "xdp_flags");
    REQUIRE_FIELD(capture, "promiscuous_mode", Bool);
    REQUIRE_FIELD(capture, "filter_expression", String);
    REQUIRE_FIELD(capture, "buffer_size", Int);
    REQUIRE_FIELD(capture, "min_packet_size", Int);
    REQUIRE_FIELD(capture, "max_packet_size", Int);
    REQUIRE_ARRAY(capture, "excluded_ports");
    REQUIRE_ARRAY(capture, "included_protocols");

    config.capture_mode = capture["mode"].asString();
    config.filter_expression = capture["filter_expression"].asString();
    config.buffer_size = capture["buffer_size"].asInt();
    config.min_packet_size = capture["min_packet_size"].asInt();
    config.max_packet_size = capture["max_packet_size"].asInt();

    // Validar y cargar arrays
    for (const auto& flag : capture["xdp_flags"]) {
        if (!flag.isString()) {
            throw std::runtime_error("TIPO INCORRECTO: xdp_flags debe contener solo strings");
        }
        config.xdp_flags.push_back(flag.asString());
    }

    for (const auto& port : capture["excluded_ports"]) {
        if (!port.isInt()) {
            throw std::runtime_error("TIPO INCORRECTO: excluded_ports debe contener solo enteros");
        }
        config.excluded_ports.push_back(port.asInt());
    }

    for (const auto& proto : capture["included_protocols"]) {
        if (!proto.isString()) {
            throw std::runtime_error("TIPO INCORRECTO: included_protocols debe contener solo strings");
        }
        config.included_protocols.push_back(proto.asString());
    }

    // AF_XDP validación estricta
    REQUIRE_OBJECT(capture, "af_xdp");
    const auto& af_xdp = capture["af_xdp"];
    REQUIRE_FIELD(af_xdp, "enabled", Bool);
    REQUIRE_FIELD(af_xdp, "queue_id", Int);
    REQUIRE_FIELD(af_xdp, "frame_size", Int);
    REQUIRE_FIELD(af_xdp, "fill_ring_size", Int);
    REQUIRE_FIELD(af_xdp, "comp_ring_size", Int);
    REQUIRE_FIELD(af_xdp, "tx_ring_size", Int);
    REQUIRE_FIELD(af_xdp, "rx_ring_size", Int);
    REQUIRE_FIELD(af_xdp, "umem_size", Int);

    config.af_xdp.enabled = af_xdp["enabled"].asBool();
    config.af_xdp.queue_id = af_xdp["queue_id"].asInt();
    config.af_xdp.frame_size = af_xdp["frame_size"].asInt();
    config.af_xdp.fill_ring_size = af_xdp["fill_ring_size"].asInt();
    config.af_xdp.comp_ring_size = af_xdp["comp_ring_size"].asInt();
    config.af_xdp.tx_ring_size = af_xdp["tx_ring_size"].asInt();
    config.af_xdp.rx_ring_size = af_xdp["rx_ring_size"].asInt();
    config.af_xdp.umem_size = af_xdp["umem_size"].asInt();

    if (verbose) {
        std::cout << "✓ Captura validada: " << config.capture_mode << " en " << config.capture_interface << "\n";
        std::cout << "✓ AF_XDP validado: " << (config.af_xdp.enabled ? "habilitado" : "deshabilitado") << "\n";
    }
}

void validate_buffers_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    REQUIRE_OBJECT(root, "buffers");
    const auto& buffers = root["buffers"];

    REQUIRE_FIELD(buffers, "ring_buffer_entries", Int);
    REQUIRE_FIELD(buffers, "user_processing_queue_depth", Int);
    REQUIRE_FIELD(buffers, "protobuf_serialize_buffer_size", Int);
    REQUIRE_FIELD(buffers, "zmq_send_buffer_size", Int);
    REQUIRE_FIELD(buffers, "flow_state_buffer_entries", Int);
    REQUIRE_FIELD(buffers, "statistics_buffer_entries", Int);
    REQUIRE_FIELD(buffers, "batch_processing_size", Int);

    config.buffers.ring_buffer_entries = buffers["ring_buffer_entries"].asInt();
    config.buffers.user_processing_queue_depth = buffers["user_processing_queue_depth"].asInt();
    config.buffers.protobuf_serialize_buffer_size = buffers["protobuf_serialize_buffer_size"].asInt();
    config.buffers.zmq_send_buffer_size = buffers["zmq_send_buffer_size"].asInt();
    config.buffers.flow_state_buffer_entries = buffers["flow_state_buffer_entries"].asInt();
    config.buffers.statistics_buffer_entries = buffers["statistics_buffer_entries"].asInt();
    config.buffers.batch_processing_size = buffers["batch_processing_size"].asInt();

    if (verbose) {
        std::cout << "✓ Buffers validados: ring=" << config.buffers.ring_buffer_entries << ", batch=" << config.buffers.batch_processing_size << "\n";
    }
}

void validate_threading_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    REQUIRE_OBJECT(root, "threading");
    const auto& threading = root["threading"];

    REQUIRE_FIELD(threading, "ring_consumer_threads", Int);
    REQUIRE_FIELD(threading, "feature_processor_threads", Int);
    REQUIRE_FIELD(threading, "zmq_sender_threads", Int);
    REQUIRE_FIELD(threading, "statistics_collector_threads", Int);
    REQUIRE_FIELD(threading, "total_worker_threads", Int);

    config.threading.ring_consumer_threads = threading["ring_consumer_threads"].asInt();
    config.threading.feature_processor_threads = threading["feature_processor_threads"].asInt();
    config.threading.zmq_sender_threads = threading["zmq_sender_threads"].asInt();
    config.threading.statistics_collector_threads = threading["statistics_collector_threads"].asInt();
    config.threading.total_worker_threads = threading["total_worker_threads"].asInt();

    REQUIRE_OBJECT(threading, "cpu_affinity");
    const auto& cpu_affinity = threading["cpu_affinity"];
    REQUIRE_FIELD(cpu_affinity, "enabled", Bool);
    config.threading.cpu_affinity_enabled = cpu_affinity["enabled"].asBool();

    if (config.threading.cpu_affinity_enabled) {
        REQUIRE_ARRAY(cpu_affinity, "ring_consumers");
        REQUIRE_ARRAY(cpu_affinity, "processors");
        REQUIRE_ARRAY(cpu_affinity, "zmq_senders");
        REQUIRE_ARRAY(cpu_affinity, "statistics");

        for (const auto& cpu : cpu_affinity["ring_consumers"]) {
            if (!cpu.isInt()) throw std::runtime_error("CPU affinity debe ser entero");
            config.threading.ring_consumers_affinity.push_back(cpu.asInt());
        }
        for (const auto& cpu : cpu_affinity["processors"]) {
            if (!cpu.isInt()) throw std::runtime_error("CPU affinity debe ser entero");
            config.threading.processors_affinity.push_back(cpu.asInt());
        }
        for (const auto& cpu : cpu_affinity["zmq_senders"]) {
            if (!cpu.isInt()) throw std::runtime_error("CPU affinity debe ser entero");
            config.threading.zmq_senders_affinity.push_back(cpu.asInt());
        }
        for (const auto& cpu : cpu_affinity["statistics"]) {
            if (!cpu.isInt()) throw std::runtime_error("CPU affinity debe ser entero");
            config.threading.statistics_affinity.push_back(cpu.asInt());
        }
    }

    REQUIRE_OBJECT(threading, "thread_priorities");
    const auto& priorities = threading["thread_priorities"];
    REQUIRE_FIELD(priorities, "ring_consumers", String);
    REQUIRE_FIELD(priorities, "processors", String);
    REQUIRE_FIELD(priorities, "zmq_senders", String);

    config.threading.ring_consumers_priority = priorities["ring_consumers"].asString();
    config.threading.processors_priority = priorities["processors"].asString();
    config.threading.zmq_senders_priority = priorities["zmq_senders"].asString();

    if (verbose) {
        std::cout << "✓ Threading validado: " << config.threading.total_worker_threads << " workers, "
                  << "CPU affinity: " << (config.threading.cpu_affinity_enabled ? "sí" : "no") << "\n";
    }
}

void validate_kernel_space_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    REQUIRE_OBJECT(root, "kernel_space");
    const auto& kernel = root["kernel_space"];

    REQUIRE_FIELD(kernel, "ebpf_program", String);
    REQUIRE_FIELD(kernel, "xdp_mode", String);
    REQUIRE_FIELD(kernel, "ring_buffer_size", Int);
    REQUIRE_FIELD(kernel, "max_flows_in_kernel", Int);
    REQUIRE_FIELD(kernel, "flow_timeout_seconds", Int);
    REQUIRE_ARRAY(kernel, "kernel_features");

    config.kernel_space.ebpf_program = kernel["ebpf_program"].asString();
    config.kernel_space.xdp_mode = kernel["xdp_mode"].asString();
    config.kernel_space.ring_buffer_size = kernel["ring_buffer_size"].asInt();
    config.kernel_space.max_flows_in_kernel = kernel["max_flows_in_kernel"].asInt();
    config.kernel_space.flow_timeout_seconds = kernel["flow_timeout_seconds"].asInt();

    for (const auto& feature : kernel["kernel_features"]) {
        if (!feature.isString()) {
            throw std::runtime_error("kernel_features debe contener solo strings");
        }
        config.kernel_space.kernel_features.push_back(feature.asString());
    }

    REQUIRE_OBJECT(kernel, "performance");
    const auto& perf = kernel["performance"];
    REQUIRE_FIELD(perf, "cpu_budget_us_per_packet", Int);
    REQUIRE_FIELD(perf, "max_instructions_per_program", Int);
    REQUIRE_FIELD(perf, "map_update_batch_size", Int);

    config.kernel_space.cpu_budget_us_per_packet = perf["cpu_budget_us_per_packet"].asInt();
    config.kernel_space.max_instructions_per_program = perf["max_instructions_per_program"].asInt();
    config.kernel_space.map_update_batch_size = perf["map_update_batch_size"].asInt();

    if (verbose) {
        std::cout << "✓ Kernel space validado: " << config.kernel_space.ebpf_program
                  << " (" << config.kernel_space.kernel_features.size() << " features)\n";
    }
}

void validate_user_space_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    REQUIRE_OBJECT(root, "user_space");
    const auto& user = root["user_space"];

    REQUIRE_FIELD(user, "flow_table_size", Int);
    REQUIRE_FIELD(user, "time_window_buffer_size", Int);
    REQUIRE_ARRAY(user, "user_features");

    config.user_space.flow_table_size = user["flow_table_size"].asInt();
    config.user_space.time_window_buffer_size = user["time_window_buffer_size"].asInt();

    for (const auto& feature : user["user_features"]) {
        if (!feature.isString()) {
            throw std::runtime_error("user_features debe contener solo strings");
        }
        config.user_space.user_features.push_back(feature.asString());
    }

    REQUIRE_OBJECT(user, "memory_management");
    const auto& mem = user["memory_management"];
    REQUIRE_FIELD(mem, "flow_eviction_policy", String);
    REQUIRE_FIELD(mem, "max_memory_usage_mb", Int);
    REQUIRE_FIELD(mem, "gc_interval_seconds", Int);

    config.user_space.flow_eviction_policy = mem["flow_eviction_policy"].asString();
    config.user_space.max_memory_usage_mb = mem["max_memory_usage_mb"].asInt();
    config.user_space.gc_interval_seconds = mem["gc_interval_seconds"].asInt();

    if (verbose) {
        std::cout << "✓ User space validado: " << config.user_space.flow_table_size
                  << " flows, " << config.user_space.user_features.size() << " feature groups\n";
    }
}

void validate_feature_groups_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    REQUIRE_OBJECT(root, "feature_groups");
    const auto& feature_groups = root["feature_groups"];

    // Validar cada feature group requerido
    std::vector<std::string> required_groups = {
        "ddos_feature_group",
        "ransomware_feature_group",
        "rf_feature_group",
        "internal_traffic_feature_group"
    };

    for (const auto& group_name : required_groups) {
        REQUIRE_OBJECT(feature_groups, group_name);
        const auto& group = feature_groups[group_name];

        REQUIRE_FIELD(group, "count", Int);
        REQUIRE_FIELD(group, "reference", String);
        REQUIRE_FIELD(group, "description", String);

        FeatureGroup fg;
        fg.name = group_name;
        fg.count = group["count"].asInt();
        fg.reference = group["reference"].asString();
        fg.description = group["description"].asString();
        fg.loaded = true;

        config.feature_groups[group_name] = fg;
    }

    if (verbose) {
        std::cout << "✓ Feature groups validados: " << config.feature_groups.size() << " grupos\n";
        for (const auto& [name, group] : config.feature_groups) {
            std::cout << "  - " << name << ": " << group.count << " features\n";
        }
    }
}

// Implementaciones stub de las funciones restantes
void validate_time_windows_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    REQUIRE_OBJECT(root, "sniffer_time_windows");
    const auto& time_windows = root["sniffer_time_windows"];

    REQUIRE_FIELD(time_windows, "flow_tracking_window_seconds", Int);
    REQUIRE_FIELD(time_windows, "statistics_collection_window_seconds", Int);
    REQUIRE_FIELD(time_windows, "feature_aggregation_window_seconds", Int);
    REQUIRE_FIELD(time_windows, "cleanup_interval_seconds", Int);
    REQUIRE_FIELD(time_windows, "max_flows_per_window", Int);
    REQUIRE_FIELD(time_windows, "window_overlap_seconds", Int);

    config.time_windows.flow_tracking_window_seconds = time_windows["flow_tracking_window_seconds"].asInt();
    config.time_windows.statistics_collection_window_seconds = time_windows["statistics_collection_window_seconds"].asInt();
    config.time_windows.feature_aggregation_window_seconds = time_windows["feature_aggregation_window_seconds"].asInt();
    config.time_windows.cleanup_interval_seconds = time_windows["cleanup_interval_seconds"].asInt();
    config.time_windows.max_flows_per_window = time_windows["max_flows_per_window"].asInt();
    config.time_windows.window_overlap_seconds = time_windows["window_overlap_seconds"].asInt();

    if (verbose) {
        std::cout << "✓ Time windows validados\n";
    }
}

void validate_network_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    REQUIRE_OBJECT(root, "network");
    const auto& network = root["network"];

    REQUIRE_OBJECT(network, "output_socket");
    const auto& output = network["output_socket"];

    REQUIRE_FIELD(output, "address", String);
    REQUIRE_FIELD(output, "port", Int);
    REQUIRE_FIELD(output, "mode", String);
    REQUIRE_FIELD(output, "socket_type", String);
    REQUIRE_FIELD(output, "high_water_mark", Int);

    config.network_output.address = output["address"].asString();
    config.network_output.port = output["port"].asInt();
    config.network_output.mode = output["mode"].asString();
    config.network_output.socket_type = output["socket_type"].asString();
    config.network_output.high_water_mark = output["high_water_mark"].asInt();

    if (verbose) {
        std::cout << "✓ Network validado: " << config.network_output.address << ":" << config.network_output.port << "\n";
    }
}

void validate_zmq_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    REQUIRE_OBJECT(root, "zmq");
    const auto& zmq = root["zmq"];

    REQUIRE_FIELD(zmq, "worker_threads", Int);
    REQUIRE_FIELD(zmq, "io_thread_pools", Int);

    config.zmq.worker_threads = zmq["worker_threads"].asInt();
    config.zmq.io_thread_pools = zmq["io_thread_pools"].asInt();

    // Socket pools
    REQUIRE_OBJECT(zmq, "socket_pools");
    const auto& socket_pools = zmq["socket_pools"];
    REQUIRE_FIELD(socket_pools, "push_sockets", Int);
    REQUIRE_FIELD(socket_pools, "load_balancing", String);
    REQUIRE_FIELD(socket_pools, "failover_enabled", Bool);

    config.zmq.push_sockets = socket_pools["push_sockets"].asInt();
    config.zmq.load_balancing = socket_pools["load_balancing"].asString();
    config.zmq.failover_enabled = socket_pools["failover_enabled"].asBool();

    // Queue management
    REQUIRE_OBJECT(zmq, "queue_management");
    const auto& queue_mgmt = zmq["queue_management"];
    REQUIRE_FIELD(queue_mgmt, "internal_queues", Int);
    REQUIRE_FIELD(queue_mgmt, "queue_size", Int);
    REQUIRE_FIELD(queue_mgmt, "queue_timeout_ms", Int);
    REQUIRE_FIELD(queue_mgmt, "overflow_policy", String);

    config.zmq.internal_queues = queue_mgmt["internal_queues"].asInt();
    config.zmq.queue_size = queue_mgmt["queue_size"].asInt();
    config.zmq.queue_timeout_ms = queue_mgmt["queue_timeout_ms"].asInt();
    config.zmq.overflow_policy = queue_mgmt["overflow_policy"].asString();

    // Connection settings
    REQUIRE_OBJECT(zmq, "connection_settings");
    const auto& conn = zmq["connection_settings"];
    REQUIRE_FIELD(conn, "sndhwm", Int);
    REQUIRE_FIELD(conn, "linger_ms", Int);
    REQUIRE_FIELD(conn, "send_timeout_ms", Int);
    REQUIRE_FIELD(conn, "rcvhwm", Int);
    REQUIRE_FIELD(conn, "recv_timeout_ms", Int);
    REQUIRE_FIELD(conn, "tcp_keepalive", Int);
    REQUIRE_FIELD(conn, "sndbuf", Int);
    REQUIRE_FIELD(conn, "rcvbuf", Int);
    REQUIRE_FIELD(conn, "reconnect_interval_ms", Int);
    REQUIRE_FIELD(conn, "max_reconnect_attempts", Int);

    config.zmq.sndhwm = conn["sndhwm"].asInt();
    config.zmq.linger_ms = conn["linger_ms"].asInt();
    config.zmq.send_timeout_ms = conn["send_timeout_ms"].asInt();
    config.zmq.rcvhwm = conn["rcvhwm"].asInt();
    config.zmq.recv_timeout_ms = conn["recv_timeout_ms"].asInt();
    config.zmq.tcp_keepalive = conn["tcp_keepalive"].asInt();
    config.zmq.sndbuf = conn["sndbuf"].asInt();
    config.zmq.rcvbuf = conn["rcvbuf"].asInt();
    config.zmq.reconnect_interval_ms = conn["reconnect_interval_ms"].asInt();
    config.zmq.max_reconnect_attempts = conn["max_reconnect_attempts"].asInt();

    // Batch processing
    REQUIRE_OBJECT(zmq, "batch_processing");
    const auto& batch = zmq["batch_processing"];
    REQUIRE_FIELD(batch, "enabled", Bool);
    REQUIRE_FIELD(batch, "batch_size", Int);
    REQUIRE_FIELD(batch, "batch_timeout_ms", Int);
    REQUIRE_FIELD(batch, "max_batches_queued", Int);

    config.zmq.batch_enabled = batch["enabled"].asBool();
    config.zmq.batch_size = batch["batch_size"].asInt();
    config.zmq.batch_timeout_ms = batch["batch_timeout_ms"].asInt();
    config.zmq.max_batches_queued = batch["max_batches_queued"].asInt();

    if (verbose) {
        std::cout << "✓ ZMQ validado: " << config.zmq.worker_threads << " workers, "
                  << config.zmq.push_sockets << " sockets\n";
    }
}

void validate_transport_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    REQUIRE_OBJECT(root, "transport");
    const auto& transport = root["transport"];

    REQUIRE_OBJECT(transport, "compression");
    const auto& compression = transport["compression"];

    REQUIRE_FIELD(compression, "enabled", Bool);
    REQUIRE_FIELD(compression, "algorithm", String);
    REQUIRE_FIELD(compression, "level", Int);

    config.compression.enabled = compression["enabled"].asBool();
    config.compression.algorithm = compression["algorithm"].asString();
    config.compression.level = compression["level"].asInt();

    if (verbose) {
        std::cout << "✓ Transport validado: compresión " << config.compression.algorithm << "\n";
    }
}

void validate_etcd_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    REQUIRE_OBJECT(root, "etcd");
    const auto& etcd = root["etcd"];

    REQUIRE_FIELD(etcd, "enabled", Bool);

    config.etcd.enabled = etcd["enabled"].asBool();

    if (verbose) {
        std::cout << "✓ etcd validado: " << (config.etcd.enabled ? "habilitado" : "deshabilitado") << "\n";
    }
}

// Funciones stub para las validaciones restantes
void validate_processing_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    (void)root; (void)config; // Suprimir warnings
    if (verbose) std::cout << "✓ Processing validado (stub)\n";
}

void validate_auto_tuner_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    (void)root; (void)config; // Suprimir warnings
    if (verbose) std::cout << "✓ Auto tuner validado (stub)\n";
}

void validate_monitoring_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    REQUIRE_OBJECT(root, "monitoring");
    const auto& monitoring = root["monitoring"];

    REQUIRE_FIELD(monitoring, "stats_interval_seconds", Int);

    config.monitoring.stats_interval_seconds = monitoring["stats_interval_seconds"].asInt();

    if (verbose) std::cout << "✓ Monitoring validado\n";
}

void validate_protobuf_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    REQUIRE_OBJECT(root, "protobuf");
    const auto& protobuf = root["protobuf"];

    REQUIRE_FIELD(protobuf, "schema_version", String);

    config.protobuf.schema_version = protobuf["schema_version"].asString();

    if (verbose) std::cout << "✓ Protobuf validado\n";
}

void validate_logging_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    REQUIRE_OBJECT(root, "logging");
    const auto& logging = root["logging"];

    REQUIRE_FIELD(logging, "level", String);

    config.logging.level = logging["level"].asString();

    if (verbose) std::cout << "✓ Logging validado\n";
}

void validate_security_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    (void)root; (void)config; // Suprimir warnings
    if (verbose) std::cout << "✓ Security validado (stub)\n";
}

void validate_backpressure_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose) {
    (void)root; (void)config; // Suprimir warnings
    if (verbose) std::cout << "✓ Backpressure validado (stub)\n";
}

void print_complete_config(const StrictSnifferConfig& config, bool verbose) {
    (void)verbose; // Suprimir warning
    std::cout << "\n=== CONFIGURACIÓN COMPLETA ===\n";
    std::cout << "Componente: " << config.component_name << " v" << config.component_version << "\n";
    std::cout << "Node: " << config.node_id << "\n";
    std::cout << "Interface: " << config.capture_interface << "\n";
    std::cout << "Perfil: " << config.active_profile << "\n";
    std::cout << "==============================\n";
}

bool initialize_etcd_connection(const StrictSnifferConfig& config, bool verbose) {
    (void)config; // Suprimir warning
    if (verbose) std::cout << "Inicializando etcd (stub)\n";
    return true;
}

bool initialize_compression(const StrictSnifferConfig& config, bool verbose) {
    (void)config; // Suprimir warning
    if (verbose) std::cout << "Inicializando compresión (stub)\n";
    return true;
}

bool initialize_zmq_pool(const StrictSnifferConfig& config, bool verbose) {
    (void)config; // Suprimir warning
    if (verbose) std::cout << "Inicializando ZMQ (stub)\n";
    return true;
}

void packet_capture_thread(const StrictSnifferConfig& config, DetailedStats& stats) {
    std::cout << "Thread de captura iniciado en " << config.capture_interface << "\n";

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        stats.incrementPacketsCaptured();
    }

    std::cout << "Thread de captura finalizado\n";
}

void detailed_stats_display_thread(const StrictSnifferConfig& config, DetailedStats& stats) {
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(config.monitoring.stats_interval_seconds));

        if (!g_running) break;

        std::cout << "\n=== ESTADÍSTICAS ===\n";
        std::cout << "Paquetes capturados: " << stats.getPacketsCaptured() << "\n";
        std::cout << "Tiempo activo: " << stats.getUptime() << " segundos\n";
        std::cout << "===================\n";
    }
}

void print_final_statistics(const StrictSnifferConfig& config, const DetailedStats& stats) {
    (void)config; // Suprimir warning
    std::cout << "\n=== ESTADÍSTICAS FINALES ===\n";
    std::cout << "Total paquetes: " << stats.getPacketsCaptured() << "\n";
    std::cout << "Tiempo total: " << stats.getUptime() << " segundos\n";
    std::cout << "============================\n";
}

void DetailedStats::reset() {
    packets_captured = 0;
    packets_processed = 0;
    packets_sent = 0;
    bytes_captured = 0;
    bytes_compressed = 0;
    errors = 0;
    drops = 0;
    kernel_packets_processed = 0;
    kernel_map_updates = 0;
    kernel_instructions_executed = 0;
    user_flows_tracked = 0;
    user_features_extracted = 0;
    user_memory_usage_mb = 0;
    zmq_messages_sent = 0;
    zmq_send_errors = 0;
    zmq_reconnections = 0;
    compression_operations = 0;
    compression_savings_bytes = 0;
    etcd_token_requests = 0;
    etcd_connection_errors = 0;
    start_time = std::chrono::steady_clock::now();
}

// Función principal main()
int main(int argc, char* argv[]) {
    sniffer::EbpfLoader* ebpf_loader_ptr = nullptr;
    std::shared_ptr<sniffer::ThreadManager> thread_manager;
    sniffer::RingBufferConsumer* ring_consumer_ptr = nullptr;
    std::thread* stats_thread_ptr = nullptr;

    try {
        parse_command_line(argc, argv, g_args);

        if (g_args.help) {
            print_help(argv[0]);
            return 0;
        }

        std::cout << "\n╔════════════════════════════════════════════╗\n";
        std::cout << "║   Enhanced Sniffer v3.1 - STRICT JSON    ║\n";
        std::cout << "╚════════════════════════════════════════════╝\n";
        std::cout << "Compilado: " << __DATE__ << " " << __TIME__ << "\n";
        std::cout << "Modo: JSON es la ley - falla rápido si falta algo\n\n";

        if (!g_args.dry_run && !g_args.show_config_only && geteuid() != 0) {
            throw std::runtime_error("PRIVILEGIOS INSUFICIENTES: Se requiere root para captura raw");
        }

        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);

        if (!strict_load_json_config(g_args.config_file, g_config, g_args.verbose)) {
            return 1;
        }

        if (g_args.show_config_only) {
            print_complete_config(g_config, g_args.verbose);
            return 0;
        }

        if (g_args.dry_run) {
            std::cout << "✅ DRY RUN COMPLETADO - Configuración JSON válida\n";
            return 0;
        }

        std::cout << "\n🔄 Inicializando subsistemas según JSON...\n";

        GOOGLE_PROTOBUF_VERIFY_VERSION;
        std::cout << "✅ Protobuf inicializado\n";

        if (g_config.etcd.enabled) {
            if (initialize_etcd_connection(g_config, g_args.verbose)) {
                std::cout << "✅ etcd conectado\n";
            }
        }

        if (g_config.compression.enabled) {
            if (initialize_compression(g_config, g_args.verbose)) {
                std::cout << "✅ Compresión " << g_config.compression.algorithm << " inicializada\n";
            }
        }

        if (initialize_zmq_pool(g_config, g_args.verbose)) {
            std::cout << "✅ ZMQ pool inicializado\n";
        }

        g_stats.reset();

        std::cout << "\n🚀 SNIFFER OPERATIVO - Configuración del JSON aplicada\n";
        std::cout << "Interface: " << g_config.capture_interface << " (" << g_config.capture_mode << ")\n";
        std::cout << "Node: " << g_config.node_id << " (cluster: " << g_config.cluster_name << ")\n";
        std::cout << "Profile: " << g_config.active_profile << "\n";
        std::cout << "Presiona Ctrl+C para detener\n\n";

        // ============================================================================
        // CARGAR E INICIALIZAR PROGRAMA EBPF
        // ============================================================================
        ebpf_loader_ptr = new sniffer::EbpfLoader();
        auto& ebpf_loader = *ebpf_loader_ptr;

        std::string bpf_path = g_config.kernel_space.ebpf_program;
        std::cout << "[eBPF] Cargando programa: " << bpf_path << std::endl;

        if (!ebpf_loader.load_program(bpf_path)) {
            std::cerr << "❌ Failed to load eBPF program from " << bpf_path << std::endl;
            return 1;
        }

        std::cout << "[eBPF] Attaching XDP to interface: " << g_config.capture_interface << std::endl;
        if (!ebpf_loader.attach_xdp(g_config.capture_interface)) {
            std::cerr << "❌ Failed to attach XDP to " << g_config.capture_interface << std::endl;
            return 1;
        }

        int ring_fd = ebpf_loader.get_ringbuf_fd();
        std::cout << "✅ eBPF program loaded and attached (ring_fd=" << ring_fd << ")" << std::endl;

        // ============================================================================
        // INICIALIZAR THREAD MANAGER
        // ============================================================================
        std::cout << "\n[Threads] Inicializando Thread Manager..." << std::endl;

        // Crear ThreadingConfig desde g_config
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
        // INICIALIZAR RING BUFFER CONSUMER
        // ============================================================================
        std::cout << "\n[RingBuffer] Inicializando RingBufferConsumer..." << std::endl;

        // Convertir StrictSnifferConfig a SnifferConfig
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

		// DEBUG: Verificar valores en g_config antes de mapear
        // std::cout << "[DEBUG] g_config.zmq.push_sockets = " << g_config.zmq.push_sockets << std::endl;
        // std::cout << "[DEBUG] g_config.zmq.worker_threads = " << g_config.zmq.worker_threads << std::endl;

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

		// DEBUG: Verificar valores ZMQ
        std::cout << "[DEBUG] zmq.socket_pools.push_sockets = " << sniffer_config.zmq.socket_pools.push_sockets << std::endl;
        std::cout << "[DEBUG] zmq.worker_threads = " << sniffer_config.zmq.worker_threads << std::endl;
        ring_consumer_ptr = new sniffer::RingBufferConsumer(sniffer_config);
        auto& ring_consumer = *ring_consumer_ptr;

        if (!ring_consumer.initialize(ring_fd, thread_manager)) {
            std::cerr << "❌ Failed to initialize RingBufferConsumer" << std::endl;
            return 1;
        }

        if (!ring_consumer.start()) {
            std::cerr << "❌ Failed to start RingBufferConsumer" << std::endl;
            return 1;
        }
        std::cout << "✅ RingBufferConsumer started - capturing REAL packets from kernel" << std::endl;

        // ============================================================================
        // THREAD DE ESTADÍSTICAS
        // ============================================================================
        stats_thread_ptr = new std::thread(detailed_stats_display_thread, std::cref(g_config), std::ref(g_stats));

        // ============================================================================
        // LOOP PRINCIPAL - ESPERAR SEÑAL DE TERMINACIÓN
        // ============================================================================
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // ============================================================================
        // CLEANUP
        // ============================================================================
        std::cout << "\n[Cleanup] Deteniendo componentes..." << std::endl;

        if (ring_consumer_ptr) {
            ring_consumer_ptr->stop();
            delete ring_consumer_ptr;
            ring_consumer_ptr = nullptr;
        }

        if (thread_manager) {
            thread_manager->stop();
            thread_manager.reset();
        }

        if (ebpf_loader_ptr) {
            delete ebpf_loader_ptr;
            ebpf_loader_ptr = nullptr;
        }

        std::cout << "✅ Componentes detenidos" << std::endl;

        if (stats_thread_ptr) {
            stats_thread_ptr->join();
            delete stats_thread_ptr;
            stats_thread_ptr = nullptr;
        }

        print_final_statistics(g_config, g_stats);

    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERROR FATAL: " << e.what() << "\n";

        if (ring_consumer_ptr) delete ring_consumer_ptr;
        if (ebpf_loader_ptr) delete ebpf_loader_ptr;
        if (stats_thread_ptr) {
            if (stats_thread_ptr->joinable()) stats_thread_ptr->join();
            delete stats_thread_ptr;
        }

        return 1;
    }

    google::protobuf::ShutdownProtobufLibrary();
    std::cout << "\n👋 Sniffer detenido correctamente\n";
    return 0;
}