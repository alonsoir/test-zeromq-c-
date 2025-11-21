// include/rag/whitelist_manager.hpp
#pragma once
#include <string>
#include <unordered_set>
#include <vector>
#include <regex>
#include <nlohmann/json.hpp>

namespace rag {

    class WhitelistManager {
    public:
        struct WhitelistConfig {
            std::unordered_set<std::string> allowed_commands;
            std::vector<std::string> allowed_pattern_strings;  // ✅ Almacenar strings también
            std::vector<std::regex> allowed_patterns;
            std::unordered_set<std::string> restricted_keys;
            size_t max_query_length = 1000;

            void clear();
        };

        struct AuditEntry {
            std::string timestamp;
            std::string command;
            std::string key;
            bool allowed;
            std::string reason;
        };

        WhitelistManager();
        ~WhitelistManager();

        bool loadFromFile(const std::string& config_path);
        bool loadFromJson(const nlohmann::json& json_config);
        bool saveToFile(const std::string& config_path = "");

        bool isCommandAllowed(const std::string& command) const;
        bool isKeyAllowed(const std::string& key) const;
        bool isQueryValid(const std::string& query) const;

        void addCommand(const std::string& command);
        void removeCommand(const std::string& command);
        void addPattern(const std::string& pattern);
        void addRestrictedKey(const std::string& key);

        std::vector<std::string> getAllowedCommands() const;

        void addAuditEntry(const AuditEntry& entry);
        std::vector<AuditEntry> getAuditLog() const;

    private:
        WhitelistConfig config_;
        std::vector<AuditEntry> audit_log_;
        std::string current_config_path_;

        bool validatePattern(const std::string& input, const std::regex& pattern) const;
        std::string getCurrentTimestamp() const;
    };

} // namespace rag