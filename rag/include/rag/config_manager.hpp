#pragma once
#include <string>
#include <nlohmann/json.hpp>

namespace Rag {

    class ConfigManager {
    public:
        ConfigManager();
        ~ConfigManager();

        bool loadFromFile(const std::string& config_path = "../config/rag-config.json");
        bool saveToFile(const std::string& config_path = "");

        // Acceso a configuraciones
        std::string getModelPath() const;
        int getSecurityLevel() const;
        bool useLLM() const;
        std::string getEtcdEndpoint() const;
        std::string getLogLevel() const;

        // Modificación de configuraciones
        bool updateSetting(const std::string& path, const std::string& value);
        bool updateSetting(const std::string& path, int value);
        bool updateSetting(const std::string& path, bool value);

        // Para registro en etcd
        nlohmann::json getConfigForEtcd() const;
        std::string getComponentId() const;

    private:
        nlohmann::json config_;
        std::string current_config_path_;

        bool validateConfig() const;
    };

} // namespace Rag