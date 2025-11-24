#include "rag/etcd_client.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>

using json = nlohmann::json;

namespace Rag {

class EtcdClient::Impl {
private:
    std::string endpoint_;
    std::string component_name_;
    std::string encryption_seed_;
    bool connected_ = false;
    std::unique_ptr<httplib::Client> client_;

public:
    Impl(const std::string& endpoint, const std::string& component_name)
        : endpoint_(endpoint), component_name_(component_name) {

        // Parsear endpoint (ej: "http://localhost:2379")
        size_t scheme_end = endpoint.find("://");
        if (scheme_end == std::string::npos) {
            throw std::runtime_error("Endpoint inválido: " + endpoint);
        }

        std::string scheme = endpoint.substr(0, scheme_end);
        std::string host_port = endpoint.substr(scheme_end + 3);

        size_t colon_pos = host_port.find(':');
        std::string host = host_port.substr(0, colon_pos);
        int port = 2379; // default

        if (colon_pos != std::string::npos) {
            port = std::stoi(host_port.substr(colon_pos + 1));
        }

        client_ = std::make_unique<httplib::Client>(host, port);
        client_->set_connection_timeout(5); // 5 segundos timeout
        client_->set_read_timeout(10);

        std::cout << "[ETCD-CLIENT] Cliente configurado para: " << host << ":" << port << std::endl;
    }

    bool initialize() {
        std::cout << "[ETCD-CLIENT] Inicializando cliente para: " << component_name_ << std::endl;

        // 1. Probar conexión básica
        if (!test_connection()) {
            std::cerr << "[ETCD-CLIENT] ❌ No se puede conectar al etcd-server" << std::endl;
            return false;
        }

        // 2. Registrar componente
        if (!register_component()) {
            std::cerr << "[ETCD-CLIENT] ❌ Error registrando componente" << std::endl;
            return false;
        }

        // 3. Obtener seed de cifrado
        if (!retrieve_encryption_seed()) {
            std::cerr << "[ETCD-CLIENT] ❌ Error obteniendo seed de cifrado" << std::endl;
            return false;
        }

        connected_ = true;
        std::cout << "[ETCD-CLIENT] ✅ Cliente inicializado correctamente" << std::endl;
        std::cout << "[ETCD-CLIENT] 🔑 Seed obtenida: " << encryption_seed_ << std::endl;
        return true;
    }

    bool is_connected() const {
        return connected_;
    }

    bool get_component_config(const std::string& component_name) {
        auto res = client_->Get("/config/" + component_name);

        if (res && res->status == 200) {
            std::cout << "[ETCD-CLIENT] 📋 Configuración de " << component_name << ": " << std::endl;
            std::cout << "  " << res->body << std::endl;
            return true;
        } else if (res && res->status == 404) {
            std::cout << "[ETCD-CLIENT] ⚠️  Componente no encontrado: " << component_name << std::endl;
        } else {
            std::cerr << "[ETCD-CLIENT] ❌ Error obteniendo configuración de " << component_name << std::endl;
            if (res) {
                std::cerr << "  Status: " << res->status << ", Body: " << res->body << std::endl;
            }
        }
        return false;
    }

    bool validate_configuration() {
        auto res = client_->Get("/validate");

        if (res && res->status == 200) {
            auto response = json::parse(res->body);
            std::cout << "[ETCD-CLIENT] 🔍 Validación del sistema:" << std::endl;
            std::cout << "  Status: " << response["status"] << std::endl;
            std::cout << "  Componentes registrados: " << response["components_registered"] << std::endl;
            std::cout << "  Anomalías detectadas: " << response["anomalies_detected"] << std::endl;

            if (response.contains("anomalies") && !response["anomalies"].empty()) {
                std::cout << "  ⚠️  Anomalías:" << std::endl;
                for (const auto& anomaly : response["anomalies"]) {
                    std::cout << "    - " << anomaly << std::endl;
                }
            }
            return true;
        }

        std::cerr << "[ETCD-CLIENT] ❌ Error validando configuración" << std::endl;
        return false;
    }

    bool update_component_config(const std::string& component_name,
                                const std::string& path,
                                const std::string& value) {
        json request = {
            {"path", path},
            {"value", value}
        };

        auto res = client_->Put("/config/" + component_name, request.dump(), "application/json");

        if (res && res->status == 200) {
            std::cout << "[ETCD-CLIENT] ✅ Configuración actualizada: "
                      << component_name << "[" << path << "] = " << value << std::endl;
            return true;
        }

        std::cerr << "[ETCD-CLIENT] ❌ Error actualizando configuración de " << component_name << std::endl;
        return false;
    }

    std::string get_encryption_seed() {
        return encryption_seed_;
    }

    bool test_encryption(const std::string& test_data) {
        // Por ahora solo probamos que podemos obtener la seed
        // En una implementación futura, probaríamos cifrado/descifrado real
        std::cout << "[ETCD-CLIENT] 🧪 Test de cifrado - Seed disponible: "
                  << (!encryption_seed_.empty() ? "✅" : "❌") << std::endl;
        std::cout << "[ETCD-CLIENT]   Seed: " << encryption_seed_ << std::endl;
        return !encryption_seed_.empty();
    }

    bool get_pipeline_status() {
        std::cout << "[ETCD-CLIENT] 🔍 Obteniendo estado del pipeline..." << std::endl;

        // Obtener configuración de todos los componentes principales
        bool success = true;
        success &= get_component_config("sniffer");
        success &= get_component_config("ml-detector");
        success &= get_component_config("firewall");
        success &= validate_configuration();

        return success;
    }

    bool start_component(const std::string& component_name) {
        std::cout << "[ETCD-CLIENT] 🚀 Simulando inicio de componente: " << component_name << std::endl;
        // TODO: Implementar comando real de inicio cuando etcd-server lo soporte
        return true;
    }

    bool stop_component(const std::string& component_name) {
        std::cout << "[ETCD-CLIENT] 🛑 Simulando parada de componente: " << component_name << std::endl;
        // TODO: Implementar comando real de parada cuando etcd-server lo soporte
        return true;
    }

    // Nuevos métodos para comandos específicos del RAG
    bool show_rag_config() {
        return get_component_config("rag");
    }

    bool set_rag_setting(const std::string& setting, const std::string& value) {
        return update_component_config("rag", setting, value);
    }

    bool get_rag_capabilities() {
        auto res = client_->Get("/config/rag");

        if (res && res->status == 200) {
            try {
                auto config = json::parse(res->body);
                std::cout << "[ETCD-CLIENT] 🎯 Capacidades del RAG:" << std::endl;

                if (config.contains("capabilities") && config["capabilities"].is_array()) {
                    for (const auto& capability : config["capabilities"]) {
                        std::cout << "  ✅ " << capability << std::endl;
                    }
                }

                if (config.contains("whitelist_commands")) {
                    std::cout << "  🔐 Whitelist commands: "
                              << (config["whitelist_commands"] ? "ACTIVADO" : "DESACTIVADO") << std::endl;
                }

                // Mostrar otras configuraciones importantes
                if (config.contains("compression")) {
                    std::cout << "  📦 Compresión: " << config["compression"] << std::endl;
                }
                if (config.contains("encryption")) {
                    std::cout << "  🔒 Cifrado: " << config["encryption"] << std::endl;
                }
                if (config.contains("version")) {
                    std::cout << "  🏷️  Versión: " << config["version"] << std::endl;
                }

                return true;
            } catch (const std::exception& e) {
                std::cerr << "[ETCD-CLIENT] ❌ Error parseando configuración RAG: " << e.what() << std::endl;
            }
        }
        return false;
    }

private:
    bool test_connection() {
        auto res = client_->Get("/health");
        if (res && res->status == 200) {
            std::cout << "[ETCD-CLIENT] ✅ Conexión establecida con etcd-server" << std::endl;
            return true;
        }
        return false;
    }

    bool register_component() {
        // Cargar configuración real del RAG - FAIL FAST
        Rag::ConfigManager config_manager;

        std::string config_path = "../config/rag-config.json";

        if (!config_manager.loadFromFile(config_path)) {
            std::cerr << "[ETCD-CLIENT] ❌ CRITICAL: Cannot load RAG config from: " << config_path << std::endl;
            std::cerr << "[ETCD-CLIENT] ❌ SYSTEM WILL EXIT - Configuration is required" << std::endl;
            return false; // Esto causará que el sistema falle rápidamente
        }

        // Usar configuración real para el registro
        auto rag_config = config_manager.getConfigForEtcd();

        auto res = client_->Post("/register", rag_config.dump(), "application/json");

        if (res && res->status == 200) {
            auto response = json::parse(res->body);
            if (response["status"] == "success") {
                std::cout << "[ETCD-CLIENT] ✅ Componente RAG registrado con configuración real" << std::endl;
                std::cout << "[ETCD-CLIENT]   Config loaded: " << config_path << std::endl;
                return true;
            }
        }

        std::cerr << "[ETCD-CLIENT] ❌ CRITICAL: Error registrando componente: "
                  << (res ? res->body : "No response") << std::endl;
        return false; // Fallar rápidamente
    }

    bool retrieve_encryption_seed() {
        auto res = client_->Get("/seed");

        if (res && res->status == 200) {
            auto response = json::parse(res->body);
            encryption_seed_ = response["seed"];
            std::cout << "[ETCD-CLIENT] 🔑 Seed obtenida correctamente" << std::endl;
            return true;
        }

        std::cerr << "[ETCD-CLIENT] ❌ Error obteniendo seed" << std::endl;
        return false;
    }
};

// Implementación de la interfaz pública
EtcdClient::EtcdClient(const std::string& endpoint, const std::string& component_name)
    : impl_(std::make_unique<Impl>(endpoint, component_name)) {}

EtcdClient::~EtcdClient() = default;

bool EtcdClient::initialize() {
    bool result = impl_->initialize();
    connected_ = result;
    return result;
}

bool EtcdClient::is_connected() const {
    return impl_->is_connected();
}

bool EtcdClient::get_component_config(const std::string& component_name) {
    return impl_->get_component_config(component_name);
}

bool EtcdClient::validate_configuration() {
    return impl_->validate_configuration();
}

bool EtcdClient::update_component_config(const std::string& component_name,
                                        const std::string& path,
                                        const std::string& value) {
    return impl_->update_component_config(component_name, path, value);
}

std::string EtcdClient::get_encryption_seed() {
    return impl_->get_encryption_seed();
}

bool EtcdClient::test_encryption(const std::string& test_data) {
    return impl_->test_encryption(test_data);
}

bool EtcdClient::get_pipeline_status() {
    return impl_->get_pipeline_status();
}

bool EtcdClient::start_component(const std::string& component_name) {
    return impl_->start_component(component_name);
}

bool EtcdClient::stop_component(const std::string& component_name) {
    return impl_->stop_component(component_name);
}

// Implementación de los nuevos métodos
bool EtcdClient::show_rag_config() {
    return impl_->show_rag_config();
}

bool EtcdClient::set_rag_setting(const std::string& setting, const std::string& value) {
    return impl_->set_rag_setting(setting, value);
}

bool EtcdClient::get_rag_capabilities() {
    return impl_->get_rag_capabilities();
}

} // namespace Rag