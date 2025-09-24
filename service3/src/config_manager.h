#pragma once

#include <jsoncpp/json/json.h>
#include <string>

class ConfigManager {
private:
    std::string config_file_;
    Json::Value config_;

public:
    explicit ConfigManager(const std::string& config_file);

    bool loadConfig();

    // Getters para configuración
    std::string getSnifferEndpoint() const;
    std::string getSocketType() const;
    std::string getConnectionType() const;
    int getReceiveTimeout() const;
    int getReceiveHighWaterMark() const;
    int getStatsIntervalSeconds() const;
    bool isVerboseMode() const;
    std::string getNodeId() const;
    std::string getClusterName() const;
    std::string getProtocolFormat() const;
    bool shouldFallbackToJson() const;
};