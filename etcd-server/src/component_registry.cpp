#include "etcd_server/component_registry.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>

//etcd-server/src/component_registry.cpp

ComponentRegistry::ComponentRegistry() {
    try {
        crypto_manager_ = std::make_unique<CryptoManager>();
        std::cout << "[REGISTRY] ComponentRegistry inicializado con cifrado" << std::endl;

        // Iniciar monitor de heartbeats
        start_heartbeat_monitor();
        std::cout << "[REGISTRY] 💓 Monitor de heartbeats iniciado" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[REGISTRY] ❌ Error inicializando CryptoManager: " << e.what() << std::endl;
        throw;
    }
}

ComponentRegistry::~ComponentRegistry() {
    stop_heartbeat_monitor();
    std::cout << "[REGISTRY] ComponentRegistry destruido" << std::endl;
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

        // Inicializar heartbeat para el componente
        {
            std::lock_guard<std::mutex> lock(heartbeat_mutex_);
            heartbeats_[name] = ComponentHeartbeat();
            std::cout << "[REGISTRY] 💓 Heartbeat inicializado para: " << name << std::endl;
        }

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

    // Limpiar heartbeat también
    {
        std::lock_guard<std::mutex> lock(heartbeat_mutex_);
        heartbeats_.erase(name);
        std::cout << "[REGISTRY] 💔 Heartbeat removido para: " << name << std::endl;
    }

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

std::string ComponentRegistry::get_encryption_key() const {
    if (!crypto_manager_) {
        throw std::runtime_error("CryptoManager no inicializado");
    }
    return crypto_manager_->get_encryption_key();
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
    if (!config.is_object()) {
        return false;
    }

    // Aceptar tanto "component" como string o como objeto con "name"
    if (config.contains("component")) {
        if (config["component"].is_string()) {
            return true;  // Formato simple: {"component": "sniffer"}
        }
        if (config["component"].is_object() && config["component"].contains("name")) {
            return true;  // Formato completo: {"component": {"name": "sniffer", ...}}
        }
        return false;  // component existe pero formato incorrecto
    }

    return false;  // No tiene "component"
}

// ============================================================================
// HEARTBEAT IMPLEMENTATION
// ============================================================================

bool ComponentRegistry::update_heartbeat(const std::string& component_name) {
    std::lock_guard<std::mutex> lock(heartbeat_mutex_);

    auto it = heartbeats_.find(component_name);
    if (it == heartbeats_.end()) {
        std::cerr << "[REGISTRY] ⚠️  Componente no registrado para heartbeat: " << component_name << std::endl;
        return false;
    }

    it->second.last_heartbeat = std::time(nullptr);
    it->second.status = "active";

    std::cout << "[REGISTRY] 💓 Heartbeat actualizado: " << component_name
              << " [timestamp: " << it->second.last_heartbeat << "]" << std::endl;

    return true;
}

std::vector<std::string> ComponentRegistry::get_dead_components() {
    std::lock_guard<std::mutex> lock(heartbeat_mutex_);
    std::vector<std::string> dead;

    for (const auto& [name, hb] : heartbeats_) {
        if (hb.status == "dead") {
            dead.push_back(name);
        }
    }

    return dead;
}

void ComponentRegistry::start_heartbeat_monitor() {
    if (monitor_running_) {
        std::cout << "[REGISTRY] ⚠️  Monitor ya está ejecutándose" << std::endl;
        return;
    }

    monitor_running_ = true;
    monitor_thread_ = std::thread(&ComponentRegistry::monitor_heartbeats, this);
    std::cout << "[REGISTRY] 🚀 Monitor de heartbeats iniciado (check cada "
              << check_interval_seconds_ << "s, timeout: "
              << (heartbeat_interval_seconds_ * timeout_multiplier_) << "s)" << std::endl;
}

void ComponentRegistry::stop_heartbeat_monitor() {
    if (!monitor_running_) {
        return;
    }

    monitor_running_ = false;
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    std::cout << "[REGISTRY] 🛑 Monitor de heartbeats detenido" << std::endl;
}

std::time_t ComponentRegistry::get_last_heartbeat(const std::string& component_name) const {
    std::lock_guard<std::mutex> lock(heartbeat_mutex_);

    auto it = heartbeats_.find(component_name);
    if (it != heartbeats_.end()) {
        return it->second.last_heartbeat;
    }

    return 0;
}

void ComponentRegistry::monitor_heartbeats() {
    std::cout << "[REGISTRY] 🔄 Thread de monitoreo iniciado" << std::endl;

    while (monitor_running_) {
        std::this_thread::sleep_for(std::chrono::seconds(check_interval_seconds_));

        std::vector<std::string> components_to_check;
        {
            std::lock_guard<std::mutex> lock(heartbeat_mutex_);
            for (const auto& [name, _] : heartbeats_) {
                components_to_check.push_back(name);
            }
        }

        // Verificar cada componente
        for (const auto& name : components_to_check) {
            if (is_component_dead(name)) {
                handle_dead_component(name);
            }
        }
    }

    std::cout << "[REGISTRY] 🔄 Thread de monitoreo finalizado" << std::endl;
}

bool ComponentRegistry::is_component_dead(const std::string& component_name) {
    std::lock_guard<std::mutex> lock(heartbeat_mutex_);

    auto it = heartbeats_.find(component_name);
    if (it == heartbeats_.end()) {
        return false;  // No existe, no está "muerto"
    }

    std::time_t now = std::time(nullptr);
    std::time_t timeout_threshold = heartbeat_interval_seconds_ * timeout_multiplier_;
    std::time_t elapsed = now - it->second.last_heartbeat;

    if (elapsed > timeout_threshold && it->second.status == "active") {
        std::cout << "[REGISTRY] ⚠️  Componente sin heartbeat: " << component_name
                  << " (última señal hace " << elapsed << "s, timeout: "
                  << timeout_threshold << "s)" << std::endl;
        return true;
    }

    return false;
}

void ComponentRegistry::handle_dead_component(const std::string& component_name) {
    std::cout << "[REGISTRY] ☠️  Manejando componente muerto: " << component_name << std::endl;

    {
        std::lock_guard<std::mutex> lock(heartbeat_mutex_);
        auto it = heartbeats_.find(component_name);
        if (it != heartbeats_.end()) {
            it->second.status = "dead";
            it->second.restart_attempts++;

            // Verificar límite de reintentos
            if (it->second.restart_attempts > max_restart_attempts_) {
                std::cerr << "[REGISTRY] ❌ Máximo de reintentos alcanzado para: "
                          << component_name << " (" << it->second.restart_attempts
                          << "/" << max_restart_attempts_ << ")" << std::endl;
                return;
            }
        }
    }

    // Desregistrar componente muerto
    std::cout << "[REGISTRY] 🗑️  Desregistrando componente muerto: " << component_name << std::endl;
    unregister_component(component_name);

    // Auto-restart usando systemd
    if (auto_restart_enabled_) {
        std::cout << "[REGISTRY] 🔄 Esperando " << restart_delay_seconds_
                  << "s antes de reiniciar..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(restart_delay_seconds_));

        std::string service_name = component_name + ".service";
        std::string cmd = "systemctl restart " + service_name;

        std::cout << "[REGISTRY] 🚀 Ejecutando: " << cmd << std::endl;
        int result = system(cmd.c_str());

        if (result == 0) {
            std::cout << "[REGISTRY] ✅ Servicio reiniciado exitosamente: " << service_name << std::endl;
        } else {
            std::cerr << "[REGISTRY] ❌ Error reiniciando servicio (exit code: "
                      << WEXITSTATUS(result) << "): " << service_name << std::endl;
            std::cerr << "[REGISTRY]    Verificar:" << std::endl;
            std::cerr << "[REGISTRY]    - Archivo /etc/systemd/system/" << service_name << " existe" << std::endl;
            std::cerr << "[REGISTRY]    - Permisos sudo en /etc/sudoers.d/ml-defender-restart" << std::endl;
            std::cerr << "[REGISTRY]    - systemctl daemon-reload ejecutado" << std::endl;
        }
    } else {
        std::cout << "[REGISTRY] ℹ️  Auto-restart deshabilitado en configuración" << std::endl;
    }
}