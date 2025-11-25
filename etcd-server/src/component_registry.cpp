#include "etcd_server/component_registry.hpp"
#include <iostream>
#include <algorithm>
//etcd-server/src/component_registry.cpp
ComponentRegistry::ComponentRegistry() {
    try {
        crypto_manager_ = std::make_unique<CryptoManager>();
        std::cout << "[REGISTRY] ComponentRegistry inicializado con cifrado" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[REGISTRY] ❌ Error inicializando CryptoManager: " << e.what() << std::endl;
        throw;
    }
}

bool ComponentRegistry::register_component(const std::string& name, const std::string& config_json) {
    try {
        json config = json::parse(config_json);

        if (!validate_component_config(config)) {
            std::cerr << "[REGISTRY] Configuración inválida para componente: " << name << std::endl;
            return false;
        }

        // Asegurar que tiene los campos mínimos
        if (!config.contains("component")) {
            config["component"] = name;
        }
        if (!config.contains("version")) {
            config["version"] = "1.0";
        }

        components_[name] = config;
        std::cout << "[REGISTRY] ✅ Componente registrado: " << name << std::endl;
        std::cout << "[REGISTRY]   Configuración: " << config.dump(2) << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[REGISTRY] ❌ Error registrando componente " << name << ": " << e.what() << std::endl;
        return false;
    }
}

bool ComponentRegistry::unregister_component(const std::string& name) {
    auto it = components_.find(name);
    if (it == components_.end()) {
        std::cout << "[REGISTRY] ❌ Componente no encontrado para desregistro: " << name << std::endl;
        return false;
    }

    components_.erase(it);
    std::cout << "[REGISTRY] 🗑️  Componente desregistrado: " << name << std::endl;
    std::cout << "[REGISTRY]   Componentes restantes: " << components_.size() << std::endl;
    return true;
}

bool ComponentRegistry::update_component_config(const std::string& name, const std::string& path, const std::string& value) {
    auto it = components_.find(name);
    if (it == components_.end()) {
        std::cerr << "[REGISTRY] Componente no encontrado: " << name << std::endl;
        return false;
    }

    try {
        // Actualizar configuración (implementación básica)
        it->second[path] = value;
        std::cout << "[REGISTRY] Configuración actualizada: " << name << "[" << path << "] = " << value << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[REGISTRY] Error actualizando configuración: " << e.what() << std::endl;
        return false;
    }
}

std::string ComponentRegistry::get_component_config(const std::string& name) const {
    auto it = components_.find(name);
    if (it != components_.end()) {
        return it->second.dump();
    }
    return "{}";
}

std::string ComponentRegistry::get_encryption_seed() const {
    if (!crypto_manager_) {
        throw std::runtime_error("CryptoManager no inicializado");
    }
    return crypto_manager_->get_current_seed();
}

std::string ComponentRegistry::encrypt_data(const std::string& plaintext) {
    if (!encryption_enabled_ || !crypto_manager_) {
        return plaintext; // Devolver sin cifrar si está desactivado
    }

    try {
        std::string ciphertext = crypto_manager_->encrypt(plaintext);
        std::cout << "[REGISTRY] 🔒 Datos cifrados (" << plaintext.length() << " bytes -> "
                  << ciphertext.length() << " bytes hex)" << std::endl;
        return ciphertext;
    } catch (const std::exception& e) {
        std::cerr << "[REGISTRY] ❌ Error cifrando datos: " << e.what() << std::endl;
        throw;
    }
}

std::string ComponentRegistry::decrypt_data(const std::string& ciphertext) {
    if (!encryption_enabled_ || !crypto_manager_) {
        return ciphertext; // Devolver sin descifrar si está desactivado
    }

    try {
        if (!crypto_manager_->validate_ciphertext(ciphertext)) {
            throw std::runtime_error("Ciphertext inválido");
        }

        std::string plaintext = crypto_manager_->decrypt(ciphertext);
        std::cout << "[REGISTRY] 🔓 Datos descifrados (" << ciphertext.length()
                  << " bytes hex -> " << plaintext.length() << " bytes)" << std::endl;
        return plaintext;
    } catch (const std::exception& e) {
        std::cerr << "[REGISTRY] ❌ Error descifrando datos: " << e.what() << std::endl;
        throw;
    }
}

bool ComponentRegistry::set_compression_mode(bool enabled) {
    compression_enabled_ = enabled;
    std::cout << "[REGISTRY] Modo compresión: " << (enabled ? "ACTIVADO" : "DESACTIVADO") << std::endl;
    return true;
}

bool ComponentRegistry::set_encryption_mode(bool enabled) {
    encryption_enabled_ = enabled;
    std::cout << "[REGISTRY] Modo cifrado: " << (enabled ? "ACTIVADO" : "DESACTIVADO") << std::endl;
    return true;
}

void ComponentRegistry::rotate_encryption_key() {
    if (crypto_manager_) {
        crypto_manager_->rotate_key();
        std::cout << "[REGISTRY] 🔄 Clave de cifrado rotada" << std::endl;
    }
}

std::string ComponentRegistry::validate_configuration() const {
    json result;
    result["status"] = "valid";
    result["components_registered"] = components_.size();
    result["encryption_enabled"] = encryption_enabled_;
    result["compression_enabled"] = compression_enabled_;
    result["encryption_seed_set"] = (crypto_manager_ != nullptr);

    // Detectar anomalías
    auto anomalies = detect_configuration_anomalies();
    result["anomalies_detected"] = anomalies.size();
    result["anomalies"] = anomalies;

    return result.dump();
}

std::vector<std::string> ComponentRegistry::detect_configuration_anomalies() const {
    std::vector<std::string> anomalies;

    if (components_.empty()) {
        anomalies.push_back("No hay componentes registrados");
    }

    // Verificar que todos los componentes tienen configuración básica
    for (const auto& [name, config] : components_) {
        if (!config.contains("component")) {
            anomalies.push_back("Componente " + name + " no tiene campo 'component'");
        }
        if (!config.contains("version")) {
            anomalies.push_back("Componente " + name + " no tiene campo 'version'");
        }
    }

    return anomalies;
}

size_t ComponentRegistry::get_component_count() const {
    return components_.size();
}

std::vector<std::string> ComponentRegistry::get_registered_components() const {
    std::vector<std::string> components;
    for (const auto& [name, _] : components_) {
        components.push_back(name);
    }
    return components;
}

bool ComponentRegistry::validate_component_config(const json& config) const {
    // Validación básica - puede expandirse según necesidades
    if (!config.is_object()) {
        return false;
    }

    // Verificar campos requeridos
    if (!config.contains("component") || !config["component"].is_string()) {
        return false;
    }

    return true;
}