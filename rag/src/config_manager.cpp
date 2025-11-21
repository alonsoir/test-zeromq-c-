// src/config_manager.cpp
#include "rag/config_manager.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace rag {

ConfigManager::ConfigManager() = default;
ConfigManager::~ConfigManager() = default;

bool ConfigManager::loadConfig(const std::string& config_path) {
    try {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            std::cerr << "❌ Cannot open config file: " << config_path << std::endl;
            return false;
        }

        file >> config_;
        current_config_path_ = config_path;

        std::cout << "✅ Loaded config from: " << config_path << std::endl;
        return validateConfig();

    } catch (const std::exception& e) {
        std::cerr << "❌ Error loading config: " << e.what() << std::endl;
        return false;
    }
}

bool ConfigManager::saveConfig(const std::string& config_path) {
    std::string path = config_path.empty() ? current_config_path_ : config_path;
    if (path.empty()) {
        std::cerr << "❌ No config path specified for save" << std::endl;
        return false;
    }

    try {
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "❌ Cannot open config file for writing: " << path << std::endl;
            return false;
        }

        file << config_.dump(4);
        std::cout << "✅ Saved config to: " << path << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error saving config: " << e.what() << std::endl;
        return false;
    }
}

std::string ConfigManager::getString(const std::string& key, const std::string& default_val) const {
    const nlohmann::json* node = getNode(key);
    return (node && node->is_string()) ? node->get<std::string>() : default_val;
}

int ConfigManager::getInt(const std::string& key, int default_val) const {
    const nlohmann::json* node = getNode(key);
    return (node && node->is_number()) ? node->get<int>() : default_val;
}

bool ConfigManager::getBool(const std::string& key, bool default_val) const {
    const nlohmann::json* node = getNode(key);
    return (node && node->is_boolean()) ? node->get<bool>() : default_val;
}

std::vector<std::string> ConfigManager::getStringArray(const std::string& key) const {
    const nlohmann::json* node = getNode(key);
    if (node && node->is_array()) {
        return node->get<std::vector<std::string>>();
    }
    return {};
}

void ConfigManager::setString(const std::string& key, const std::string& value) {
    nlohmann::json* node = getNode(key);
    if (node) {
        *node = value;
    }
}

void ConfigManager::setInt(const std::string& key, int value) {
    nlohmann::json* node = getNode(key);
    if (node) {
        *node = value;
    }
}

void ConfigManager::setBool(const std::string& key, bool value) {
    nlohmann::json* node = getNode(key);
    if (node) {
        *node = value;
    }
}

bool ConfigManager::validateConfig() const {
    // Validaciones básicas
    if (getString("etcd.endpoints").empty()) {
        std::cerr << "❌ etcd.endpoints is required" << std::endl;
        return false;
    }

    if (getString("llama.model_path").empty()) {
        std::cerr << "❌ llama.model_path is required" << std::endl;
        return false;
    }

    if (getString("security.whitelist_file").empty()) {
        std::cerr << "❌ security.whitelist_file is required" << std::endl;
        return false;
    }

    return true;
}

nlohmann::json* ConfigManager::getNode(const std::string& key_path) {
    nlohmann::json* current = &config_;
    std::istringstream iss(key_path);
    std::string token;

    while (std::getline(iss, token, '.')) {
        if (current->is_object() && current->contains(token)) {
            current = &(*current)[token];
        } else {
            return nullptr;
        }
    }
    return current;
}

const nlohmann::json* ConfigManager::getNode(const std::string& key_path) const {
    const nlohmann::json* current = &config_;
    std::istringstream iss(key_path);
    std::string token;

    while (std::getline(iss, token, '.')) {
        if (current->is_object() && current->contains(token)) {
            current = &(*current)[token];
        } else {
            return nullptr;
        }
    }
    return current;
}

} // namespace rag