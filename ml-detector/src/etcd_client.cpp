#include "etcd_client.hpp"
#include <etcd_client/etcd_client.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

namespace ml_detector {

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

        std::cout << "📡 [ml-detector] Parsed endpoint: " << host_ << ":" << port_ << std::endl;
    }
};

// ============================================================================
// Implementación pública del Adapter
// ============================================================================

EtcdClient::EtcdClient(const std::string& endpoint, const std::string& component_name)
    : pImpl(std::make_unique<Impl>(endpoint, component_name)) {
    std::cout << "🔧 [ml-detector] EtcdClient adapter created for: " << component_name << std::endl;
}

EtcdClient::~EtcdClient() = default;

bool EtcdClient::initialize() {
    std::cout << "🔗 [ml-detector] Initializing etcd client..." << std::endl;

    // Conectar y registrar (obtiene clave automáticamente)
    if (!pImpl->client_->connect()) {
        std::cerr << "❌ [ml-detector] Failed to connect to etcd-server" << std::endl;
        return false;
    }

    std::cout << "✅ [ml-detector] Connected to etcd-server" << std::endl;
    std::cout << "🔑 [ml-detector] Encryption key received automatically" << std::endl;

    return true;
}

bool EtcdClient::registerService() {
    std::cout << "📝 [ml-detector] Registering ml-detector service in etcd..." << std::endl;

    // Cargar TODO el ml_detector_config.json
    std::ifstream config_file("config/ml_detector_config.json");
    if (!config_file.is_open()) {
        std::cerr << "❌ [ml-detector] Failed to open ml_detector_config.json" << std::endl;
        return false;
    }

    nlohmann::json full_config;
    try {
        config_file >> full_config;
    } catch (const std::exception& e) {
        std::cerr << "❌ [ml-detector] Failed to parse ml_detector_config.json: " << e.what() << std::endl;
        return false;
    }
    config_file.close();

    // Añadir metadata de registro
    full_config["_registration"] = {
        {"timestamp", std::time(nullptr)},
        {"component", pImpl->component_name_},
        {"type", "ml-detector"},
        {"status", "active"}
    };

    // Extraer capabilities para log
    std::string profile = full_config.value("profile", "unknown");
    bool level1_enabled = full_config["ml"]["level1"].value("enabled", false);
    bool ddos_enabled = full_config["ml"]["level2"]["ddos"].value("enabled", false);
    bool ransomware_enabled = full_config["ml"]["level2"]["ransomware"].value("enabled", false);
    bool traffic_enabled = full_config["ml"]["level3"]["web"].value("enabled", false);
    bool internal_enabled = full_config["ml"]["level3"]["internal"].value("enabled", false);

    std::cout << "📋 [ml-detector] Capabilities:" << std::endl;
    std::cout << "   - Profile: " << profile << std::endl;
    std::cout << "   - Level 1 (Attack): " << (level1_enabled ? "✅" : "❌") << std::endl;
    std::cout << "   - Level 2 DDoS: " << (ddos_enabled ? "✅" : "❌") << std::endl;
    std::cout << "   - Level 2 Ransomware: " << (ransomware_enabled ? "✅" : "❌") << std::endl;
    std::cout << "   - Level 3 Traffic: " << (traffic_enabled ? "✅" : "❌") << std::endl;
    std::cout << "   - Level 3 Internal: " << (internal_enabled ? "✅" : "❌") << std::endl;
    std::cout << "   - Config Size: " << full_config.dump().size() << " bytes" << std::endl;

    // Registrar componente
    if (!pImpl->client_->register_component()) {
        std::cerr << "❌ [ml-detector] Failed to register component" << std::endl;
        return false;
    }

    // Subir config completo con put_config() (automáticamente cifrado+comprimido)
    if (!pImpl->client_->put_config(full_config.dump(2))) {
        std::cerr << "❌ [ml-detector] Failed to upload config" << std::endl;
        return false;
    }

    std::cout << "✅ [ml-detector] Service registered successfully" << std::endl;
    std::cout << "🔐 [ml-detector] Config uploaded encrypted + compressed" << std::endl;

    return true;
}

std::string EtcdClient::get_encryption_seed() const {
    if (!pImpl || !pImpl->client_) {
        std::cerr << "❌ [ml-detector] EtcdClient not initialized" << std::endl;
        return "";
    }

    std::string seed = pImpl->client_->get_encryption_key();

    if (seed.empty()) {
        std::cerr << "❌ [ml-detector] Failed to get encryption seed" << std::endl;
    } else {
        std::cout << "🔑 [ml-detector] Retrieved encryption seed ("
                  << seed.size() << " bytes)" << std::endl;
    }

    return seed;
}

    std::string get_hmac_key() {
    // Path: /secrets/{short_component_name}
    // Equivale a /secrets/ml-detector para component_name = "ml-detector"
    std::string path = "/secrets/" + short_name_;  // mismo short_name_ que usa /register

    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(5);

    auto res = cli.Get(path.c_str());

    if (!res || res->status != 200) {
        std::cerr << "[etcd] Failed to get HMAC key from " << path
                  << " status=" << (res ? res->status : -1) << std::endl;
        return "";
    }

    try {
        auto j = nlohmann::json::parse(res->body);
        // etcd-server returns: {"key_hex": "...", "component": "...", ...}
        if (j.contains("key_hex")) {
            std::string key_hex = j["key_hex"].get<std::string>();
            std::cout << "[etcd] HMAC key received for " << path
                      << " (" << key_hex.size() << " chars)" << std::endl;
            return key_hex;
        }
        // Fallback: some versions return "key" directly
        if (j.contains("key")) {
            return j["key"].get<std::string>();
        }
        std::cerr << "[etcd] Unexpected HMAC key response format" << std::endl;
        return "";
    } catch (const std::exception& e) {
        std::cerr << "[etcd] Failed to parse HMAC key response: " << e.what() << std::endl;
        return "";
    }
}

std::string EtcdClient::get_hmac_key() const {
    return pImpl->get_hmac_key();
}

} // namespace ml_detector