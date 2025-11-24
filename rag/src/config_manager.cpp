#include "rag/config_manager.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace Rag {

ConfigManager::ConfigManager() {
    // NO configurar valores por defecto - la verdad está en el JSON
}

ConfigManager::~ConfigManager() = default;

bool ConfigManager::loadFromFile(const std::string& config_path) {
    try {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            std::cerr << "❌ CRITICAL: Cannot open config file: " << config_path << std::endl;
            return false;
        }

        file >> config_;
        current_config_path_ = config_path;

        std::cout << "✅ Loaded RAG config from: " << config_path << std::endl;

        // Validación estricta - si falla, salimos
        if (!validateConfig()) {
            std::cerr << "❌ CRITICAL: RAG config validation failed" << std::endl;
            return false;
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ CRITICAL: Error loading RAG config: " << e.what() << std::endl;
        return false;
    }
}

bool ConfigManager::saveToFile(const std::string& config_path) {
    std::string path = config_path.empty() ? current_config_path_ : config_path;
    if (path.empty()) {
        std::cerr << "❌ CRITICAL: No config path specified for save" << std::endl;
        return false;
    }

    try {
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "❌ CRITICAL: Cannot open config file for writing: " << path << std::endl;
            return false;
        }

        file << config_.dump(4);
        std::cout << "✅ Saved RAG config to: " << path << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ CRITICAL: Error saving RAG config: " << e.what() << std::endl;
        return false;
    }
}

// Métodos de acceso a configuraciones - FAIL FAST
std::string ConfigManager::getModelPath() const {
    if (!config_.contains("model_path") || !config_["model_path"].is_string()) {
        throw std::runtime_error("CRITICAL: model_path not found in config");
    }
    return config_["model_path"].get<std::string>();
}

int ConfigManager::getSecurityLevel() const {
    if (!config_.contains("security_level") || !config_["security_level"].is_number()) {
        throw std::runtime_error("CRITICAL: security_level not found in config");
    }
    return config_["security_level"].get<int>();
}

bool ConfigManager::useLLM() const {
    if (!config_.contains("use_llm") || !config_["use_llm"].is_boolean()) {
        throw std::runtime_error("CRITICAL: use_llm not found in config");
    }
    return config_["use_llm"].get<bool>();
}

std::string ConfigManager::getEtcdEndpoint() const {
    if (!config_.contains("etcd_endpoint") || !config_["etcd_endpoint"].is_string()) {
        throw std::runtime_error("CRITICAL: etcd_endpoint not found in config");
    }
    return config_["etcd_endpoint"].get<std::string>();
}

std::string ConfigManager::getLogLevel() const {
    if (!config_.contains("log_level") || !config_["log_level"].is_string()) {
        throw std::runtime_error("CRITICAL: log_level not found in config");
    }
    return config_["log_level"].get<std::string>();
}

// Métodos de modificación - FAIL FAST
bool ConfigManager::updateSetting(const std::string& path, const std::string& value) {
    try {
        // Verificar que la configuración está cargada
        if (config_.empty()) {
            throw std::runtime_error("Config not loaded");
        }

        // Manejar paths simples (sin puntos)
        if (path.find('.') == std::string::npos) {
            if (!config_.contains(path)) {
                throw std::runtime_error("Setting '" + path + "' not found in config");
            }
            config_[path] = value;
        } else {
            // Manejar paths con puntos (llama_config.context_size)
            std::istringstream iss(path);
            std::string token;
            nlohmann::json* current = &config_;

            while (std::getline(iss, token, '.')) {
                if (current->is_object() && current->contains(token)) {
                    current = &(*current)[token];
                } else {
                    throw std::runtime_error("Config path '" + path + "' not found");
                }
            }
            *current = value;
        }

        std::cout << "✅ Updated config: " << path << " = " << value << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ CRITICAL: Error updating config '" << path << "': " << e.what() << std::endl;
        return false;
    }
}

bool ConfigManager::updateSetting(const std::string& path, int value) {
    return updateSetting(path, std::to_string(value));
}

bool ConfigManager::updateSetting(const std::string& path, bool value) {
    return updateSetting(path, value ? "true" : "false");
}

// Para registro en etcd - FAIL FAST
nlohmann::json ConfigManager::getConfigForEtcd() const {
    if (config_.empty()) {
        throw std::runtime_error("CRITICAL: Config not loaded for etcd registration");
    }

    // Validar campos mínimos requeridos para etcd
    if (!config_.contains("id") || !config_["id"].is_string()) {
        throw std::runtime_error("CRITICAL: 'id' field required for etcd registration");
    }

    return config_; // Devolver configuración completa
}

std::string ConfigManager::getComponentId() const {
    if (!config_.contains("id") || !config_["id"].is_string()) {
        throw std::runtime_error("CRITICAL: 'id' field not found in config");
    }
    return config_["id"].get<std::string>();
}

bool ConfigManager::validateConfig() const {
    // Validaciones CRÍTICAS - si falla alguna, el sistema no puede funcionar

    if (!config_.contains("id") || !config_["id"].is_string()) {
        std::cerr << "❌ CRITICAL: 'id' field is required and must be a string" << std::endl;
        return false;
    }

    if (!config_.contains("model_path") || !config_["model_path"].is_string()) {
        std::cerr << "❌ CRITICAL: 'model_path' field is required and must be a string" << std::endl;
        return false;
    }

    if (!config_.contains("etcd_endpoint") || !config_["etcd_endpoint"].is_string()) {
        std::cerr << "❌ CRITICAL: 'etcd_endpoint' field is required and must be a string" << std::endl;
        return false;
    }

    if (!config_.contains("security_level") || !config_["security_level"].is_number()) {
        std::cerr << "❌ CRITICAL: 'security_level' field is required and must be a number" << std::endl;
        return false;
    }

    int security_level = config_["security_level"].get<int>();
    if (security_level < 1 || security_level > 5) {
        std::cerr << "❌ CRITICAL: security_level must be between 1 and 5" << std::endl;
        return false;
    }

    std::cout << "✅ RAG config validation passed - all critical fields present" << std::endl;
    return true;
}

} // namespace Rag