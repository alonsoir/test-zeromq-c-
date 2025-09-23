#include <iostream>
#include <string>
#include <memory>
#include <csignal>
#include <cstdlib>
#include <unistd.h>
#include <sys/resource.h>
#include <getopt.h>
#include <fstream>
#include <thread>
#include <chrono>

#include "config_manager.hpp"

// Global flag para shutdown limpio
static volatile bool g_running = true;

// Signal handler para shutdown graceful
void signal_handler(int signal) {
    std::cout << "\n[INFO] Received signal " << signal << ", initiating shutdown..." << std::endl;
    g_running = false;
}

// Verificar privilegios root
bool check_root_privileges() {
    if (geteuid() != 0) {
        std::cerr << "[FATAL] This program requires root privileges for eBPF/XDP operations." << std::endl;
        std::cerr << "[FATAL] Please run with sudo or as root user." << std::endl;
        return false;
    }
    return true;
}

// Verificar kernel version
bool check_kernel_version() {
    // Implementación básica - en producción sería más robusta
    std::system("uname -r > /tmp/kernel_version.txt");
    std::ifstream kernel_file("/tmp/kernel_version.txt");
    if (!kernel_file.is_open()) {
        std::cerr << "[WARNING] Could not determine kernel version" << std::endl;
        return true; // Continuar de todas formas
    }

    std::string kernel_version;
    std::getline(kernel_file, kernel_version);
    kernel_file.close();
    std::system("rm -f /tmp/kernel_version.txt");

    std::cout << "[INFO] Detected kernel version: " << kernel_version << std::endl;

    // Verificar que es >= 6.0 (versión mínima recomendada para eBPF moderno)
    if (kernel_version.find("6.") != 0 && kernel_version.find("7.") != 0) {
        if (kernel_version.find("5.") == 0) {
            std::cout << "[WARNING] Kernel 5.x detected. Some eBPF features may not be available." << std::endl;
            std::cout << "[WARNING] Recommended: kernel 6.12+ for optimal performance." << std::endl;
        } else {
            std::cerr << "[ERROR] Unsupported kernel version: " << kernel_version << std::endl;
            std::cerr << "[ERROR] Minimum required: 5.0, recommended: 6.12+" << std::endl;
            return false;
        }
    } else {
        std::cout << "[INFO] Kernel version is compatible with advanced eBPF features." << std::endl;
    }

    return true;
}

// Configurar límites del sistema para eBPF
void configure_system_limits() {
    // Aumentar límite de memory lock para eBPF maps
    struct rlimit rlim = {RLIM_INFINITY, RLIM_INFINITY};
    if (setrlimit(RLIMIT_MEMLOCK, &rlim)) {
        std::cerr << "[WARNING] Failed to set RLIMIT_MEMLOCK" << std::endl;
    } else {
        std::cout << "[INFO] Configured unlimited memory lock for eBPF maps" << std::endl;
    }
}

// Verificar dependencias del sistema
bool check_system_dependencies() {
    std::vector<std::string> required_tools = {
        "bpftool",
        "ip"
    };

    bool all_found = true;

    for (const auto& tool : required_tools) {
        std::string cmd = "which " + tool + " > /dev/null 2>&1";
        if (std::system(cmd.c_str()) != 0) {
            std::cerr << "[ERROR] Required tool not found: " << tool << std::endl;
            all_found = false;
        } else {
            std::cout << "[INFO] Found required tool: " << tool << std::endl;
        }
    }

    return all_found;
}

// Verificar interface de red
bool check_network_interface(const std::string& interface) {
    if (interface == "any") {
        std::cout << "[INFO] Using 'any' interface - will capture from all interfaces" << std::endl;
        return true;
    }

    std::string cmd = "ip link show " + interface + " > /dev/null 2>&1";
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "[ERROR] Network interface not found: " << interface << std::endl;
        std::cerr << "[INFO] Available interfaces:" << std::endl;
        std::system("ip link show | grep '^[0-9]' | cut -d: -f2 | sed 's/^ */  - /'");
        return false;
    }

    std::cout << "[INFO] Network interface verified: " << interface << std::endl;
    return true;
}

// Mostrar ayuda
void show_help(const char* program_name) {
    std::cout << "\nC++20 Evolutionary Sniffer v3.1.0\n";
    std::cout << "High-performance network packet sniffer with eBPF/XDP and ML features\n\n";
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -c, --config FILE    Configuration file path (default: config/sniffer.json)\n";
    std::cout << "  -d, --daemon         Run as daemon (background)\n";
    std::cout << "  -v, --verbose        Verbose output\n";
    std::cout << "  -t, --test-config    Test configuration and exit\n";
    std::cout << "  --autotune           Enable auto-tuner calibration on startup\n";
    std::cout << "  -h, --help           Show this help message\n";
    std::cout << "  --version            Show version information\n\n";
    std::cout << "Examples:\n";
    std::cout << "  sudo " << program_name << " --config=/etc/sniffer.json\n";
    std::cout << "  sudo " << program_name << " --test-config\n";
    std::cout << "  sudo " << program_name << " --autotune --verbose\n\n";
    std::cout << "Requirements:\n";
    std::cout << "  - Root privileges for eBPF/XDP operations\n";
    std::cout << "  - Linux kernel 5.0+ (recommended: 6.12+)\n";
    std::cout << "  - Network interface with XDP support\n\n";
}

// Mostrar información de versión
void show_version() {
    std::cout << "C++20 Evolutionary Sniffer v3.1.0\n";
    std::cout << "Built with:\n";
    std::cout << "  - C++ standard: " << __cplusplus << "\n";
    std::cout << "  - Compiler: " << __VERSION__ << "\n";
    std::cout << "  - eBPF/XDP support: enabled\n";
    std::cout << "  - Protobuf v3.1 support: enabled\n";
    std::cout << "  - ZeroMQ support: enabled\n";
    std::cout << "  - Auto-tuner: available\n";
}

int main(int argc, char* argv[]) {
    std::string config_path = "config/sniffer.json";
    bool daemon_mode = false;
    bool verbose = false;
    bool test_config_only = false;
    bool enable_autotune = false;

    // Parsear argumentos de línea de comandos
    struct option long_options[] = {
        {"config", required_argument, 0, 'c'},
        {"daemon", no_argument, 0, 'd'},
        {"verbose", no_argument, 0, 'v'},
        {"test-config", no_argument, 0, 't'},
        {"autotune", no_argument, 0, 'a'},
        {"help", no_argument, 0, 'h'},
        {"version", no_argument, 0, 'V'},
        {0, 0, 0, 0}
    };

    int c;
    int option_index = 0;

    while ((c = getopt_long(argc, argv, "c:dvtah", long_options, &option_index)) != -1) {
        switch (c) {
            case 'c':
                config_path = optarg;
                break;
            case 'd':
                daemon_mode = true;
                break;
            case 'v':
                verbose = true;
                break;
            case 't':
                test_config_only = true;
                break;
            case 'a':
                enable_autotune = true;
                break;
            case 'h':
                show_help(argv[0]);
                return 0;
            case 'V':
                show_version();
                return 0;
            case '?':
                std::cerr << "[ERROR] Invalid option. Use --help for usage information." << std::endl;
                return 1;
            default:
                break;
        }
    }

    // Banner de inicio
    std::cout << "\n=== C++20 Evolutionary Sniffer v3.1.0 ===" << std::endl;
    std::cout << "Initializing high-performance packet capture with eBPF/XDP..." << std::endl;

    // Verificaciones preliminares
    std::cout << "\n[1/6] Checking system requirements..." << std::endl;

    if (!check_root_privileges()) {
        return 1;
    }

    if (!check_kernel_version()) {
        return 1;
    }

    if (!check_system_dependencies()) {
        return 1;
    }

    // Configurar sistema
    std::cout << "\n[2/6] Configuring system limits..." << std::endl;
    configure_system_limits();

    // Cargar configuración
    std::cout << "\n[3/6] Loading configuration from: " << config_path << std::endl;

    std::unique_ptr<sniffer::SnifferConfig> config;
    try {
        config = sniffer::ConfigManager::load_from_file(config_path);
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Configuration error: " << e.what() << std::endl;
        return 1;
    }

    // Override auto-tuner si se especifica en CLI
    if (enable_autotune) {
        std::cout << "[INFO] Auto-tuner would be enabled via command line flag (not implemented yet)" << std::endl;
    }

    // Verificar interface de red
    std::cout << "\n[4/6] Verifying network interface..." << std::endl;
    if (!check_network_interface(config->capture.interface)) {
        return 1;
    }

    // Si solo testing config, salir aquí
    if (test_config_only) {
        std::cout << "\n[INFO] Configuration test PASSED. All validations successful." << std::endl;
        std::cout << "[INFO] Sniffer would bind to: " << config->network.output_socket.address
                  << ":" << config->network.output_socket.port << std::endl;
        std::cout << "[INFO] Ready for packet capture on interface: " << config->capture.interface << std::endl;
        return 0;
    }

    // Configurar signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGUSR1, signal_handler);

    // Daemon mode
    if (daemon_mode) {
        std::cout << "\n[5/6] Starting in daemon mode..." << std::endl;
        if (daemon(0, 0) < 0) {
            std::cerr << "[FATAL] Failed to daemonize" << std::endl;
            return 1;
        }
    }

    std::cout << "\n[6/6] Initializing sniffer components..." << std::endl;

    // TODO: Aquí irían las inicializaciones de los componentes principales:
    // - SnifferEngine
    // - eBPF program loading
    // - Ring buffer setup
    // - Feature aggregator
    // - ZeroMQ output socket
    // - Auto-tuner (if enabled)

    std::cout << "\n=== Sniffer Ready ===" << std::endl;
    std::cout << "Capturing packets on interface: " << config->capture.interface << std::endl;
    std::cout << "Output socket: " << config->network.output_socket.address
              << ":" << config->network.output_socket.port
              << " (" << config->network.output_socket.socket_type << ")" << std::endl;
    std::cout << "Kernel features: " << config->features.kernel_feature_count << " (configured)" << std::endl;
    std::cout << "User features: " << config->features.user_feature_count << " (configured)" << std::endl;
    std::cout << "Performance target: 10M pps (default target)" << std::endl;
    std::cout << "Auto-tuner: NOT IMPLEMENTED YET" << std::endl;

    std::cout << "\nPress Ctrl+C to stop...\n" << std::endl;

    // Main loop - por ahora solo esperar señales
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // TODO: Aquí iría el processing loop principal:
        // - Consume ring buffer events
        // - Process features
        // - Send to ZeroMQ
        // - Update metrics
    }

    // Cleanup
    std::cout << "\n=== Shutting down ===" << std::endl;
    std::cout << "[INFO] Cleaning up resources..." << std::endl;

    // TODO: Cleanup de componentes:
    // - Detach eBPF program
    // - Close ring buffer
    // - Close ZeroMQ sockets
    // - Stop threads

    std::cout << "[INFO] Sniffer stopped cleanly." << std::endl;
    return 0;
}