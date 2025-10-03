#include "config_manager.h"
#include <fstream>
#include <iostream>
//service3/src/config_manager.cpp
ConfigManager::ConfigManager(const std::string& config_file)
    : config_file_(config_file) {
}

bool ConfigManager::loadConfig() {
    std::ifstream file(config_file_);
    if (!file.is_open()) {
        std::cerr << "[ConfigManager] ❌ No se pudo abrir: " << config_file_ << std::endl;
        return false;
    }

    Json::CharReaderBuilder builder;
    std::string errors;

    if (!Json::parseFromStream(builder, file, &config_, &errors)) {
        std::cerr << "[ConfigManager] ❌ Error parseando JSON: " << errors << std::endl;
        return false;
    }

    std::cout << "[ConfigManager] ✅ Configuración cargada desde: " << config_file_ << std::endl;
    return true;
}

std::string ConfigManager::getSnifferEndpoint() const {
    return config_["connection"].get("sniffer_endpoint", "tcp://host.docker.internal:5571").asString();
}

std::string ConfigManager::getSocketType() const {
    return config_["connection"].get("socket_type", "PULL").asString();
}

std::string ConfigManager::getConnectionType() const {
    return config_["connection"].get("connection_type", "connect").asString();
}

int ConfigManager::getReceiveTimeout() const {
    return config_["connection"].get("receive_timeout_ms", 1000).asInt();
}

int ConfigManager::getReceiveHighWaterMark() const {
    return config_["connection"].get("receive_hwm", 10000).asInt();
}

int ConfigManager::getStatsIntervalSeconds() const {
    return config_.get("stats_interval_seconds", 5).asInt();
}

bool ConfigManager::isVerboseMode() const {
    return config_.get("verbose", true).asBool();
}

std::string ConfigManager::getNodeId() const {
    return config_.get("node_id", "service3_consumer_001").asString();
}

std::string ConfigManager::getClusterName() const {
    return config_.get("cluster_name", "upgraded-happiness-cluster").asString();
}