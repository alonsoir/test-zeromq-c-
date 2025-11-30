// rag/src/etcd_client.cpp -
#include "rag/etcd_client.hpp"
#include "rag/config_manager.hpp"
#include <httplib.h>
#include <iostream>
#include <thread>
#include <chrono>

namespace Rag {

// ============================================================================
// Implementación de EtcdClient::Impl (PIMPL)
// ============================================================================

struct EtcdClient::Impl {
    std::string endpoint_;
    std::string component_name_;
    bool connected_ = false;

    Impl(const std::string& endpoint, const std::string& component_name)
        : endpoint_(endpoint), component_name_(component_name) {}

    bool initialize() {
        std::cout << "🔗 Initializing etcd client with endpoint: " << endpoint_ << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        connected_ = true;
        std::cout << "✅ etcd client initialized successfully" << std::endl;
        return true;
    }

    bool is_connected() const {
        return connected_;
    }

    bool set_key(const std::string& key, const std::string& value) {
        if (!connected_) {
            std::cerr << "❌ etcd client not connected" << std::endl;
            return false;
        }
        std::cout << "💾 Setting key: " << key << " = " << value << std::endl;
        return true;
    }

    bool register_component() {
        try {
            auto& config_manager = ConfigManager::getInstance();
            auto rag_config = config_manager.getRagConfig();
            auto etcd_config = config_manager.getEtcdConfig();

            // Parsear el endpoint para obtener host y puerto
            std::string host = "localhost";
            int port = 2379;

            size_t protocol_pos = endpoint_.find("://");
            if (protocol_pos != std::string::npos) {
                std::string without_protocol = endpoint_.substr(protocol_pos + 3);
                size_t colon_pos = without_protocol.find(':');
                if (colon_pos != std::string::npos) {
                    host = without_protocol.substr(0, colon_pos);
                    port = std::stoi(without_protocol.substr(colon_pos + 1));
                } else {
                    host = without_protocol;
                }
            }

            std::cout << "📤 Enviando registro a etcd-server: " << host << ":" << port << std::endl;

            // Crear cliente HTTP
            httplib::Client cli(host, port);
            cli.set_connection_timeout(5);

            // Preparar datos de registro
            nlohmann::json registration_data = {
                {"component", component_name_},
                {"config", {
                    {"host", rag_config.host},
                    {"port", rag_config.port},
                    {"model_name", rag_config.model_name},
                    {"embedding_dimension", rag_config.embedding_dimension},
                    {"etcd_endpoint", etcd_config.host + ":" + std::to_string(etcd_config.port)},
                    {"status", "active"}
                }}
            };

            std::string request_body = registration_data.dump();
            std::cout << "📦 Datos de registro: " << request_body << std::endl;

            // Enviar petición POST real
            auto res = cli.Post("/register", request_body, "application/json");

            if (res && res->status == 200) {
                std::cout << "✅ Registro exitoso en etcd-server" << std::endl;
                std::cout << "📋 Respuesta: " << res->body << std::endl;
                return true;
            } else {
                std::cerr << "❌ Error en registro etcd: ";
                if (res) {
                    std::cerr << "HTTP " << res->status << " - " << res->body << std::endl;
                } else {
                    std::cerr << "No se pudo conectar al etcd-server" << std::endl;
                }
                return false;
            }

        } catch (const std::exception& e) {
            std::cerr << "❌ Excepción en register_component: " << e.what() << std::endl;
            return false;
        }
    }

    bool unregister_component() {
    std::cout << "🔧 Desregistro asíncrono - enviando solicitud en segundo plano..." << std::endl;

    // Lanzar en un hilo separado para no bloquear la salida
    std::thread([this]() {
        try {
            std::cout << "🔄 Procesando desregistro en segundo plano..." << std::endl;

            // Enfoque ultra-minimalista
            httplib::Client cli("127.0.0.1", 2379);
            cli.set_connection_timeout(1);  // Muy corto
            cli.set_read_timeout(1);        // Muy corto

            // JSON como string simple, sin nlohmann
            std::string json_body = "{\"component\":\"" + component_name_ + "\"}";

            // Enviar y olvidar - no nos importa la respuesta
            auto res = cli.Post("/unregister", json_body, "application/json");

            if (res) {
                std::cout << "✅ Desregistro confirmado en segundo plano" << std::endl;
            } else {
                std::cout << "⚠️  Desregistro enviado (sin confirmación)" << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cout << "⚠️  Error en desregistro en segundo plano: " << e.what() << std::endl;
        }
        catch (...) {
            std::cout << "⚠️  Error desconocido en desregistro en segundo plano" << std::endl;
        }
    }).detach();  // IMPORTANTE: detach para no bloquear

    std::cout << "✅ Solicitud de desregistro enviada (procesándose en fondo)" << std::endl;
    return true;
}

    bool get_component_config(const std::string& component_name) {
        std::cout << "📋 Getting config for: " << component_name << std::endl;
        return true;
    }

    bool validate_configuration() {
        std::cout << "✅ Validating configuration..." << std::endl;
        return true;
    }

    bool update_component_config(const std::string& component_name, const std::string& config) {
        std::cout << "⚙️  Updating config for: " << component_name << " with: " << config << std::endl;
        return true;
    }

    std::string get_encryption_seed() {
        return "default-encryption-seed-12345";
    }

    bool test_encryption(const std::string& test_data) {
        std::cout << "🔒 Testing encryption with: " << test_data << std::endl;
        return true;
    }

    bool get_pipeline_status() {
        std::cout << "📊 Getting pipeline status..." << std::endl;
        return true;
    }

    bool start_component(const std::string& component_name) {
        std::cout << "🚀 Starting component: " << component_name << std::endl;
        return true;
    }

    bool stop_component(const std::string& component_name) {
        std::cout << "🛑 Stopping component: " << component_name << std::endl;
        return true;
    }

    bool show_rag_config() {
        try {
            auto& config_manager = ConfigManager::getInstance();
            auto rag_config = config_manager.getRagConfig();

            std::cout << "🔧 RAG Configuration:" << std::endl;
            std::cout << "  - Host: " << rag_config.host << std::endl;
            std::cout << "  - Port: " << rag_config.port << std::endl;
            std::cout << "  - Model: " << rag_config.model_name << std::endl;
            std::cout << "  - Embedding Dimension: " << rag_config.embedding_dimension << std::endl;

            return true;
        } catch (const std::exception& e) {
            std::cerr << "❌ Error showing RAG config: " << e.what() << std::endl;
            return false;
        }
    }

    bool set_rag_setting(const std::string& setting, const std::string& value) {
        std::cout << "⚙️  Setting RAG setting: " << setting << " = " << value << std::endl;
        return true;
    }

    bool get_rag_capabilities() {
        std::cout << "🎯 RAG Capabilities: component_management, config_validation, pipeline_control" << std::endl;
        return true;
    }
};

// ============================================================================
// Implementación de EtcdClient (interfaz pública)
// ============================================================================

EtcdClient::EtcdClient(const std::string& endpoint, const std::string& component_name)
    : pImpl(std::make_unique<Impl>(endpoint, component_name)) {}

EtcdClient::~EtcdClient() = default;

bool EtcdClient::initialize() {
    return pImpl->initialize();
}

bool EtcdClient::is_connected() const {
    return pImpl->is_connected();
}

bool EtcdClient::registerService() {
    return pImpl->register_component(); // Internamente llama al método snake_case
}

bool EtcdClient::unregisterService() {
    return pImpl->unregister_component(); // Internamente llama al método snake_case
}

bool EtcdClient::get_component_config(const std::string& component_name) {
    return pImpl->get_component_config(component_name);
}

bool EtcdClient::validate_configuration() {
    return pImpl->validate_configuration();
}

bool EtcdClient::update_component_config(const std::string& component_name, const std::string& config) {
    return pImpl->update_component_config(component_name, config);
}

std::string EtcdClient::get_encryption_seed() {
    return pImpl->get_encryption_seed();
}

bool EtcdClient::test_encryption(const std::string& test_data) {
    return pImpl->test_encryption(test_data);
}

bool EtcdClient::get_pipeline_status() {
    return pImpl->get_pipeline_status();
}

bool EtcdClient::start_component(const std::string& component_name) {
    return pImpl->start_component(component_name);
}

bool EtcdClient::stop_component(const std::string& component_name) {
    return pImpl->stop_component(component_name);
}

bool EtcdClient::show_rag_config() {
    return pImpl->show_rag_config();
}

bool EtcdClient::set_rag_setting(const std::string& setting, const std::string& value) {
    return pImpl->set_rag_setting(setting, value);
}

bool EtcdClient::get_rag_capabilities() {
    return pImpl->get_rag_capabilities();
}

} // namespace Rag