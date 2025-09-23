#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <json/json.h>

namespace sniffer {

    // Exception para errores de configuración
    class ConfigException : public std::runtime_error {
    public:
        explicit ConfigException(const std::string& message)
            : std::runtime_error("Configuration Error: " + message) {}
    };

    // Configuración de red simplificada
    struct NetworkConfig {
        struct OutputSocket {
            std::string address;
            int port;
            std::string socket_type;
        } output_socket;
    };

    // Configuración de captura simplificada
    struct CaptureConfig {
        std::string interface;
    };

    // Configuración de features simplificada
    struct FeaturesConfig {
        bool extraction_enabled;
        int kernel_feature_count;
        int user_feature_count;
    };

    // Configuración completa simplificada
    struct SnifferConfig {
        // Información básica del componente
        std::string component_name;
        std::string version;
        std::string node_id;
        std::string cluster_name;

        // Configuraciones básicas
        NetworkConfig network;
        CaptureConfig capture;
        FeaturesConfig features;

        // Validación básica
        bool is_valid() const;
        std::vector<std::string> validate() const;
    };

    // Manager simplificado para configuración
    class ConfigManager {
    public:
        // Cargar configuración desde archivo JSON
        static std::unique_ptr<SnifferConfig> load_from_file(const std::string& config_path);

        // Validar configuración
        static void validate_config(const SnifferConfig& config);

        // Utilidades
        static void fail_fast(const std::string& error_message);
        static void log_config_summary(const SnifferConfig& config);
    };

} // namespace sniffer