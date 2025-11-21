// src/whitelist_manager.cpp
#include "rag/whitelist_manager.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace rag {

void WhitelistManager::WhitelistConfig::clear() {
    allowed_commands.clear();
    allowed_pattern_strings.clear();  // ✅ Limpiar strings también
    allowed_patterns.clear();
    restricted_keys.clear();
    max_query_length = 1000;
}

WhitelistManager::WhitelistManager() {
    config_.max_query_length = 1000;
}

WhitelistManager::~WhitelistManager() = default;

bool WhitelistManager::loadFromFile(const std::string& config_path) {
    try {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            std::cerr << "❌ Cannot open whitelist file: " << config_path << std::endl;
            return false;
        }

        nlohmann::json json_config;
        file >> json_config;
        current_config_path_ = config_path;

        return loadFromJson(json_config);

    } catch (const std::exception& e) {
        std::cerr << "❌ Error loading whitelist: " << e.what() << std::endl;
        return false;
    }
}

bool WhitelistManager::loadFromJson(const nlohmann::json& json_config) {
    config_.clear();

    try {
        // Cargar comandos permitidos
        if (json_config.contains("allowed_commands") && json_config["allowed_commands"].is_array()) {
            for (const auto& cmd : json_config["allowed_commands"]) {
                if (cmd.is_string()) {
                    config_.allowed_commands.insert(cmd.get<std::string>());
                }
            }
        }

        // Cargar patrones permitidos
        if (json_config.contains("allowed_patterns") && json_config["allowed_patterns"].is_array()) {
            for (const auto& pattern : json_config["allowed_patterns"]) {
                if (pattern.is_string()) {
                    std::string pattern_str = pattern.get<std::string>();
                    try {
                        config_.allowed_patterns.emplace_back(pattern_str);
                        config_.allowed_pattern_strings.push_back(pattern_str);  // ✅ Guardar string también
                    } catch (const std::regex_error& e) {
                        std::cerr << "❌ Invalid regex pattern: " << pattern_str << " - " << e.what() << std::endl;
                    }
                }
            }
        }

        // Cargar keys restringidas
        if (json_config.contains("restricted_keys") && json_config["restricted_keys"].is_array()) {
            for (const auto& key : json_config["restricted_keys"]) {
                if (key.is_string()) {
                    config_.restricted_keys.insert(key.get<std::string>());
                }
            }
        }

        // Cargar longitud máxima
        if (json_config.contains("max_query_length") && json_config["max_query_length"].is_number()) {
            config_.max_query_length = json_config["max_query_length"].get<size_t>();
        }

        std::cout << "✅ Loaded whitelist: " << config_.allowed_commands.size()
                  << " commands, " << config_.allowed_patterns.size()
                  << " patterns, " << config_.restricted_keys.size()
                  << " restricted keys" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error parsing whitelist JSON: " << e.what() << std::endl;
        return false;
    }
}

bool WhitelistManager::saveToFile(const std::string& config_path) {
    std::string path = config_path.empty() ? current_config_path_ : config_path;
    if (path.empty()) {
        std::cerr << "❌ No config path specified for save" << std::endl;
        return false;
    }

    try {
        nlohmann::json json_config;

        // Comandos permitidos
        json_config["allowed_commands"] = nlohmann::json::array();
        for (const auto& cmd : config_.allowed_commands) {
            json_config["allowed_commands"].push_back(cmd);
        }

        // Patrones permitidos - usar las strings almacenadas ✅
        json_config["allowed_patterns"] = nlohmann::json::array();
        for (const auto& pattern_str : config_.allowed_pattern_strings) {
            json_config["allowed_patterns"].push_back(pattern_str);
        }

        // Keys restringidas
        json_config["restricted_keys"] = nlohmann::json::array();
        for (const auto& key : config_.restricted_keys) {
            json_config["restricted_keys"].push_back(key);
        }

        // Longitud máxima
        json_config["max_query_length"] = config_.max_query_length;

        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "❌ Cannot open whitelist file for writing: " << path << std::endl;
            return false;
        }

        file << json_config.dump(4);
        std::cout << "✅ Saved whitelist to: " << path << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error saving whitelist: " << e.what() << std::endl;
        return false;
    }
}

bool WhitelistManager::isCommandAllowed(const std::string& command) const {
    return config_.allowed_commands.find(command) != config_.allowed_commands.end();
}

bool WhitelistManager::isKeyAllowed(const std::string& key) const {
    // Verificar keys restringidas
    if (config_.restricted_keys.find(key) != config_.restricted_keys.end()) {
        return false;
    }

    // Verificar patrones permitidos
    for (const auto& pattern : config_.allowed_patterns) {
        if (validatePattern(key, pattern)) {
            return true;
        }
    }

    return false;
}

bool WhitelistManager::isQueryValid(const std::string& query) const {
    return query.length() <= config_.max_query_length;
}

void WhitelistManager::addCommand(const std::string& command) {
    config_.allowed_commands.insert(command);
}

void WhitelistManager::removeCommand(const std::string& command) {
    config_.allowed_commands.erase(command);
}

void WhitelistManager::addPattern(const std::string& pattern) {
    try {
        config_.allowed_patterns.emplace_back(pattern);
        config_.allowed_pattern_strings.push_back(pattern);  // ✅ Guardar string también
    } catch (const std::regex_error& e) {
        std::cerr << "❌ Invalid regex pattern: " << pattern << " - " << e.what() << std::endl;
    }
}

void WhitelistManager::addRestrictedKey(const std::string& key) {
    config_.restricted_keys.insert(key);
}

std::vector<std::string> WhitelistManager::getAllowedCommands() const {
    return std::vector<std::string>(
        config_.allowed_commands.begin(),
        config_.allowed_commands.end()
    );
}

void WhitelistManager::addAuditEntry(const AuditEntry& entry) {
    audit_log_.push_back(entry);
    // Mantener solo las últimas 1000 entradas
    if (audit_log_.size() > 1000) {
        audit_log_.erase(audit_log_.begin());
    }
}

std::vector<WhitelistManager::AuditEntry> WhitelistManager::getAuditLog() const {
    return audit_log_;
}

bool WhitelistManager::validatePattern(const std::string& input, const std::regex& pattern) const {
    try {
        return std::regex_match(input, pattern);
    } catch (const std::regex_error& e) {
        std::cerr << "❌ Regex error: " << e.what() << std::endl;
        return false;
    }
}

std::string WhitelistManager::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

} // namespace rag