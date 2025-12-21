#include "firewall/etcd_client.hpp"
#include <etcd_client/etcd_client.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

namespace mldefender::firewall {

// ============================================================================
// Implementación del Adapter (PIMPL)
// ============================================================================

struct EtcdClient::Impl {
    std::unique_ptr<etcd_client::EtcdClient> client_;
    std::string endpoint_;
    std::string component_name_;
    std::string host_;
    int port_;

    Impl(const std::string& endpoint, const std::string& component_name)
        : endpoint_(endpoint), component_name_(component_name) {

        // Parse endpoint (formato: "host:port" o "http://host:port")
        parseEndpoint(endpoint);

        // Configurar etcd-client
        etcd_client::Config config;
        config.component_name = component_name;
        config.host = host_;
        config.port = port_;
        config.encryption_enabled = true;
        config.compression_enabled = true;

        client_ = std::make_unique<etcd_client::EtcdClient>(config);
    }

    void parseEndpoint(const std::string& endpoint) {
        // Remover http:// si existe
        std::string clean_endpoint = endpoint;
        if (endpoint.find("http://") == 0) {
            clean_endpoint = endpoint.substr(7);
        }

        // Buscar ':'
        size_t colon_pos = clean_endpoint.find(':');
        if (colon_pos != std::string::npos) {
            host_ = clean_endpoint.substr(0, colon_pos);
            port_ = std::stoi(clean_endpoint.substr(colon_pos + 1));
        } else {
            host_ = clean_endpoint;
            port_ = 2379; // Puerto por defecto
        }

        std::cout << "📡 [firewall-acl-agent] Parsed endpoint: " << host_ << ":" << port_ << std::endl;
    }
};

// ============================================================================
// Implementación pública del Adapter
// ============================================================================

EtcdClient::EtcdClient(const std::string& endpoint, const std::string& component_name)
    : pImpl(std::make_unique<Impl>(endpoint, component_name)) {
    std::cout << "🔧 [firewall-acl-agent] EtcdClient adapter created for: " << component_name << std::endl;
}

EtcdClient::~EtcdClient() = default;

bool EtcdClient::initialize() {
    std::cout << "🔗 [firewall-acl-agent] Initializing etcd client..." << std::endl;

    // Conectar y registrar (obtiene clave automáticamente)
    if (!pImpl->client_->connect()) {
        std::cerr << "❌ [firewall-acl-agent] Failed to connect to etcd-server" << std::endl;
        return false;
    }

    std::cout << "✅ [firewall-acl-agent] Connected to etcd-server" << std::endl;
    std::cout << "🔑 [firewall-acl-agent] Encryption key received automatically" << std::endl;

    return true;
}

bool EtcdClient::registerService() {
    std::cout << "📝 [firewall-acl-agent] Registering service in etcd..." << std::endl;

    // Cargar TODO el firewall.json
    std::ifstream config_file("../config/firewall.json");  // ← CAMBIAR A ../config/
    if (!config_file.is_open()) {
        std::cerr << "❌ [firewall-acl-agent] Failed to open ../config/firewall.json" << std::endl;
        return false;
    }

    nlohmann::json full_config;
    try {
        config_file >> full_config;
    } catch (const std::exception& e) {
        std::cerr << "❌ [firewall-acl-agent] Failed to parse firewall.json: " << e.what() << std::endl;
        return false;
    }
    config_file.close();

    // Añadir metadata de registro
    full_config["_registration"] = {
        {"timestamp", std::time(nullptr)},
        {"component", pImpl->component_name_},
        {"type", "firewall"},
        {"status", "active"}
    };

    // Extraer capabilities para log
    bool dry_run = full_config["operation"].value("dry_run", false);
    bool ipset_check = full_config["health_check"].value("ipset_health_check", false);
    bool iptables_check = full_config["health_check"].value("iptables_health_check", false);
    std::string zmq_endpoint = full_config["zmq"].value("endpoint", "unknown");

    std::cout << "📋 [firewall-acl-agent] Capabilities:" << std::endl;
    std::cout << "   - Mode: " << (dry_run ? "DRY-RUN" : "PRODUCTION") << std::endl;
    std::cout << "   - ZMQ Endpoint: " << zmq_endpoint << std::endl;
    std::cout << "   - IPSet Health Check: " << (ipset_check ? "✅" : "❌") << std::endl;
    std::cout << "   - IPTables Health Check: " << (iptables_check ? "✅" : "❌") << std::endl;
    std::cout << "   - Config Size: " << full_config.dump().size() << " bytes" << std::endl;

    // Registrar componente
    if (!pImpl->client_->register_component()) {
        std::cerr << "❌ [firewall-acl-agent] Failed to register component" << std::endl;
        return false;
    }

    // Subir config completo con put_config() (automáticamente cifrado+comprimido)
    if (!pImpl->client_->put_config(full_config.dump(2))) {
        std::cerr << "❌ [firewall-acl-agent] Failed to upload config" << std::endl;
        return false;
    }

    std::cout << "✅ [firewall-acl-agent] Service registered successfully" << std::endl;
    std::cout << "🔐 [firewall-acl-agent] Config uploaded encrypted + compressed" << std::endl;

    return true;
}

} // namespace mldefender::firewall