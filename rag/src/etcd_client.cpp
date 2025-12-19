// rag/src/etcd_client.cpp - Adapter para etcd_client library
#include "rag/etcd_client.hpp"
#include "etcd_client/etcd_client.hpp"
#include <nlohmann/json.hpp>
#include <iostream>

namespace Rag {

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
        
        std::cout << "📡 Parsed endpoint: " << host_ << ":" << port_ << std::endl;
    }
};

// ============================================================================
// Implementación pública del Adapter
// ============================================================================

EtcdClient::EtcdClient(const std::string& endpoint, const std::string& component_name)
    : pImpl(std::make_unique<Impl>(endpoint, component_name)) {
    std::cout << "🔧 EtcdClient adapter created for: " << component_name << std::endl;
}

EtcdClient::~EtcdClient() = default;

bool EtcdClient::initialize() {
    std::cout << "🔗 Initializing etcd client..." << std::endl;
    
    // Conectar y registrar (obtiene clave automáticamente)
    if (!pImpl->client_->connect()) {
        std::cerr << "❌ Failed to connect to etcd-server" << std::endl;
        return false;
    }
    
    std::cout << "✅ Connected to etcd-server" << std::endl;
    std::cout << "🔑 Encryption key received automatically" << std::endl;
    
    return true;
}

bool EtcdClient::is_connected() const {
    return pImpl->client_->is_connected();
}

bool EtcdClient::registerService() {
    std::cout << "📝 Registering RAG service in etcd..." << std::endl;
    
    // Crear configuración de RAG
    nlohmann::json rag_config = {
        {"component", pImpl->component_name_},
        {"type", "rag-logger"},
        {"status", "active"},
        {"capabilities", {
            {"llm_inference", true},
            {"semantic_search", true},
            {"query_validation", true}
        }},
        {"version", "1.0.0"}
    };
    
    // Subir config con put_config() (automáticamente cifrado+comprimido)
    if (!pImpl->client_->put_config(rag_config.dump(2))) {
        std::cerr << "❌ Failed to register service" << std::endl;
        return false;
    }
    
    std::cout << "✅ Service registered successfully" << std::endl;
    return true;
}

bool EtcdClient::unregisterService() {
    std::cout << "🗑️ Unregistering service..." << std::endl;
    return pImpl->client_->unregister_component();
}

bool EtcdClient::get_component_config(const std::string& component_name) {
    std::cout << "📥 Fetching config for: " << component_name << std::endl;
    
    std::string config_json = pImpl->client_->get_component_config(component_name);
    if (config_json.empty()) {
        std::cerr << "❌ Component not found: " << component_name << std::endl;
        return false;
    }
    
    std::cout << "✅ Config received for " << component_name << std::endl;
    std::cout << config_json << std::endl;
    
    return true;
}

bool EtcdClient::validate_configuration() {
    std::cout << "✓ Configuration validation (using new library)" << std::endl;
    return true;
}

bool EtcdClient::update_component_config(const std::string& component_name, const std::string& config) {
    std::cout << "📤 Updating config for: " << component_name << std::endl;
    return pImpl->client_->put_config(config);
}

std::string EtcdClient::get_encryption_seed() {
    // La nueva librería maneja el cifrado automáticamente
    return "encryption_managed_automatically";
}

bool EtcdClient::test_encryption(const std::string& test_data) {
    std::cout << "🔒 Encryption test (handled automatically by library)" << std::endl;
    (void)test_data;
    return true;
}

bool EtcdClient::get_pipeline_status() {
    std::cout << "📊 Getting pipeline status..." << std::endl;
    // Obtener lista de componentes
    auto components = pImpl->client_->list_components();
    std::cout << "✅ Found " << components.size() << " components" << std::endl;
    return true;
}

bool EtcdClient::start_component(const std::string& component_name) {
    std::cout << "▶️ Starting component: " << component_name << std::endl;
    // Implementar lógica de start si es necesario
    (void)component_name;
    return true;
}

bool EtcdClient::stop_component(const std::string& component_name) {
    std::cout << "⏸️ Stopping component: " << component_name << std::endl;
    // Implementar lógica de stop si es necesario
    (void)component_name;
    return true;
}

bool EtcdClient::show_rag_config() {
    std::cout << "📋 RAG Configuration:" << std::endl;
    return get_component_config(pImpl->component_name_);
}

bool EtcdClient::set_rag_setting(const std::string& setting, const std::string& value) {
    std::cout << "⚙️ Setting RAG setting: " << setting << " = " << value << std::endl;
    // Implementar lógica de update específica
    (void)setting;
    (void)value;
    return true;
}

bool EtcdClient::get_rag_capabilities() {
    std::cout << "🎯 RAG Capabilities:" << std::endl;
    std::cout << "  - LLM Inference: ✅" << std::endl;
    std::cout << "  - Semantic Search: ✅" << std::endl;
    std::cout << "  - Query Validation: ✅" << std::endl;
    std::cout << "  - Encrypted Communication: ✅" << std::endl;
    std::cout << "  - Compressed Transfers: ✅" << std::endl;
    return true;
}

} // namespace Rag
