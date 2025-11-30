#pragma once

#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "crypto_manager.hpp"
//etcd-server/include/etcd_server/component_registry.hpp
using json = nlohmann::json;

class ComponentRegistry {
private:
    std::unordered_map<std::string, json> components_;
    std::unique_ptr<CryptoManager> crypto_manager_;
    bool compression_enabled_ = true;
    bool encryption_enabled_ = true;

public:
    ComponentRegistry();

    // Gestión de componentes
    bool register_component(const std::string& name, const std::string& config_json);
    bool unregister_component(const std::string& name);
    bool update_component_config(const std::string& name, const std::string& path, const std::string& value);
    std::string get_component_config(const std::string& name) const;

    // Cifrado
    std::string get_encryption_seed() const;
    std::string encrypt_data(const std::string& plaintext);
    std::string decrypt_data(const std::string& ciphertext);

    // Configuración global
    bool set_compression_mode(bool enabled);
    bool set_encryption_mode(bool enabled);
    void rotate_encryption_key();

    // Validación
    std::string validate_configuration() const;
    std::vector<std::string> detect_configuration_anomalies() const;

    // Información del sistema
    size_t get_component_count() const;
    std::vector<std::string> get_registered_components() const;

private:
    bool validate_component_config(const json& config) const;
};