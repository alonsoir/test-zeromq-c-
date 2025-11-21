// include/rag/config_manager.hpp
#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace rag {

    class ConfigManager {
    public:
        ConfigManager();
        ~ConfigManager();

        bool loadConfig(const std::string& config_path);
        bool saveConfig(const std::string& config_path = "");

        std::string getString(const std::string& key, const std::string& default_val = "") const;
        int getInt(const std::string& key, int default_val = 0) const;
        bool getBool(const std::string& key, bool default_val = false) const;
        std::vector<std::string> getStringArray(const std::string& key) const;

        void setString(const std::string& key, const std::string& value);
        void setInt(const std::string& key, int value);
        void setBool(const std::string& key, bool value);

        bool validateConfig() const;

    private:
        nlohmann::json config_;
        std::string current_config_path_;

        nlohmann::json* getNode(const std::string& key_path);
        const nlohmann::json* getNode(const std::string& key_path) const;
    };

} // namespace rag