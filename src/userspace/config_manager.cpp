#include "config_manager.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <filesystem>

namespace sniffer {

// Cargar configuración desde archivo JSON
std::unique_ptr<SnifferConfig> ConfigManager::load_from_file(const std::string& config_path) {
    if (!std::filesystem::exists(config_path)) {
        fail_fast("Configuration file not found: " + config_path);
    }

    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        fail_fast("Cannot open configuration file: " + config_path);
    }

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;

    if (!Json::parseFromStream(builder, config_file, &root, &errors)) {
        fail_fast("JSON parsing error: " + errors);
    }

    auto config = std::make_unique<SnifferConfig>();

    // Parsear campos básicos con valores por defecto seguros
    config->component_name = root.get("component", Json::Value{})["name"].asString();
    config->version = root.get("component", Json::Value{})["version"].asString();
    config->node_id = root.get("node_id", "default_node").asString();
    config->cluster_name = root.get("cluster_name", "default_cluster").asString();

    // Configuración de red básica
    auto network = root["network"]["output_socket"];
    config->network.output_socket.address = network.get("address", "localhost").asString();
    config->network.output_socket.port = network.get("port", 5571).asInt();
    config->network.output_socket.socket_type = network.get("socket_type", "PUSH").asString();

    // Features básicas
    config->features.extraction_enabled = true;
    config->features.kernel_feature_count = 25;
    config->features.user_feature_count = 58;

    // Captura básica
    config->capture.interface = root["capture"].get("interface", "eth0").asString();

    std::cout << "[INFO] Configuration loaded successfully from: " + config_path << std::endl;
    log_config_summary(*config);

    return config;
}

void ConfigManager::validate_config(const SnifferConfig& config) {
    if (config.component_name.empty()) {
        fail_fast("Component name cannot be empty");
    }

    if (config.node_id.empty()) {
        fail_fast("Node ID cannot be empty");
    }

    if (config.network.output_socket.port <= 0 || config.network.output_socket.port > 65535) {
        fail_fast("Invalid port number: " + std::to_string(config.network.output_socket.port));
    }

    std::cout << "[INFO] Configuration validation passed" << std::endl;
}

void ConfigManager::fail_fast(const std::string& error_message) {
    std::cerr << "[FATAL] " << error_message << std::endl;
    std::exit(1);
}

void ConfigManager::log_config_summary(const SnifferConfig& config) {
    std::cout << "\n=== Configuration Summary ===" << std::endl;
    std::cout << "Component: " << config.component_name << " v" << config.version << std::endl;
    std::cout << "Node ID: " << config.node_id << std::endl;
    std::cout << "Cluster: " << config.cluster_name << std::endl;
    std::cout << "Interface: " << config.capture.interface << std::endl;
    std::cout << "Output socket: " << config.network.output_socket.address
              << ":" << config.network.output_socket.port
              << " (" << config.network.output_socket.socket_type << ")" << std::endl;
    std::cout << "==============================\n" << std::endl;
}

bool SnifferConfig::is_valid() const {
    return !component_name.empty() && !node_id.empty();
}

std::vector<std::string> SnifferConfig::validate() const {
    std::vector<std::string> errors;
    if (component_name.empty()) errors.push_back("Component name required");
    if (node_id.empty()) errors.push_back("Node ID required");
    return errors;
}

} // namespace sniffer