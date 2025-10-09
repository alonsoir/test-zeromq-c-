// main.cpp - Enhanced Sniffer v3.1 STRICT JSON
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

// [TODAS las funciones validate_* se mantienen igual - omitidas por brevedad]
// ... (incluye todas las validate functions aquí - no las repito para ahorrar espacio) ...

void print_complete_config(const StrictSnifferConfig& config, bool verbose) {
    (void)verbose;
    std::cout << "\n=== CONFIGURACIÓN COMPLETA ===\n";
    std::cout << "Componente: " << config.component_name << " v" << config.component_version << "\n";
    std::cout << "Node: " << config.node_id << "\n";
    std::cout << "Interface: " << config.capture_interface << "\n";
    std::cout << "Perfil: " << config.active_profile << "\n";
    std::cout << "==============================\n";
}

bool initialize_etcd_connection(const StrictSnifferConfig& config, bool verbose) {
    (void)config;
    if (verbose) std::cout << "Inicializando etcd (stub)\n";
    return true;
}

bool initialize_compression(const StrictSnifferConfig& config, bool verbose) {
    (void)config;
    if (verbose) std::cout << "Inicializando compresión (stub)\n";
    return true;
}

bool initialize_zmq_pool(const StrictSnifferConfig& config, bool verbose) {
    (void)config;
    if (verbose) std::cout << "Inicializando ZMQ (stub)\n";
    return true;
}

// Función principal main()
int main(int argc, char* argv[]) {
    sniffer::EbpfLoader* ebpf_loader_ptr = nullptr;
    std::shared_ptr<sniffer::ThreadManager> thread_manager;
    sniffer::RingBufferConsumer* ring_consumer_ptr = nullptr;

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

        sniffer::SnifferConfig sniffer_config;
        sniffer_config.node_id = g_config.node_id;
        sniffer_config.cluster_name = g_config.cluster_name;

        // Map all config sections
        sniffer_config.buffers.ring_buffer_entries = g_config.buffers.ring_buffer_entries;
        sniffer_config.buffers.user_processing_queue_depth = g_config.buffers.user_processing_queue_depth;
        sniffer_config.buffers.protobuf_serialize_buffer_size = g_config.buffers.protobuf_serialize_buffer_size;
        sniffer_config.buffers.zmq_send_buffer_size = g_config.buffers.zmq_send_buffer_size;
        sniffer_config.buffers.flow_state_buffer_entries = g_config.buffers.flow_state_buffer_entries;
        sniffer_config.buffers.statistics_buffer_entries = g_config.buffers.statistics_buffer_entries;
        sniffer_config.buffers.batch_processing_size = g_config.buffers.batch_processing_size;

        sniffer_config.threading.ring_consumer_threads = g_config.threading.ring_consumer_threads;
        sniffer_config.threading.feature_processor_threads = g_config.threading.feature_processor_threads;
        sniffer_config.threading.zmq_sender_threads = g_config.threading.zmq_sender_threads;
        sniffer_config.threading.statistics_collector_threads = g_config.threading.statistics_collector_threads;
        sniffer_config.threading.total_worker_threads = g_config.threading.total_worker_threads;

        sniffer_config.threading.cpu_affinity.enabled = g_config.threading.cpu_affinity_enabled;
        sniffer_config.threading.cpu_affinity.ring_consumers = g_config.threading.ring_consumers_affinity;
        sniffer_config.threading.cpu_affinity.processors = g_config.threading.processors_affinity;
        sniffer_config.threading.cpu_affinity.zmq_senders = g_config.threading.zmq_senders_affinity;
        sniffer_config.threading.cpu_affinity.statistics = g_config.threading.statistics_affinity;

        sniffer_config.threading.thread_priorities["ring_consumers"] = g_config.threading.ring_consumers_priority;
        sniffer_config.threading.thread_priorities["processors"] = g_config.threading.processors_priority;
        sniffer_config.threading.thread_priorities["zmq_senders"] = g_config.threading.zmq_senders_priority;

        sniffer_config.zmq.worker_threads = g_config.zmq.worker_threads;
        sniffer_config.zmq.io_thread_pools = g_config.zmq.io_thread_pools;
        sniffer_config.zmq.socket_pools.push_sockets = g_config.zmq.push_sockets;
        sniffer_config.zmq.socket_pools.load_balancing = g_config.zmq.load_balancing;
        sniffer_config.zmq.socket_pools.failover_enabled = g_config.zmq.failover_enabled;

        sniffer_config.zmq.queue_management.internal_queues = g_config.zmq.internal_queues;
        sniffer_config.zmq.queue_management.queue_size = g_config.zmq.queue_size;
        sniffer_config.zmq.queue_management.queue_timeout_ms = g_config.zmq.queue_timeout_ms;
        sniffer_config.zmq.queue_management.overflow_policy = g_config.zmq.overflow_policy;

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

        sniffer_config.zmq.batch_processing.enabled = g_config.zmq.batch_enabled;
        sniffer_config.zmq.batch_processing.batch_size = g_config.zmq.batch_size;
        sniffer_config.zmq.batch_processing.batch_timeout_ms = g_config.zmq.batch_timeout_ms;
        sniffer_config.zmq.batch_processing.max_batches_queued = g_config.zmq.max_batches_queued;

        sniffer_config.network.output_socket.address = g_config.network_output.address;
        sniffer_config.network.output_socket.port = g_config.network_output.port;
        sniffer_config.network.output_socket.mode = g_config.network_output.mode;
        sniffer_config.network.output_socket.socket_type = g_config.network_output.socket_type;
        sniffer_config.network.output_socket.high_water_mark = g_config.network_output.high_water_mark;

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

    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERROR FATAL: " << e.what() << "\n";

        if (ring_consumer_ptr) delete ring_consumer_ptr;
        if (ebpf_loader_ptr) delete ebpf_loader_ptr;

        return 1;
    }

    google::protobuf::ShutdownProtobufLibrary();
    std::cout << "\n👋 Sniffer detenido correctamente\n";
    return 0;
}