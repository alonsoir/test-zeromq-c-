#pragma once
#include <string>
#include <vector>
#include <memory>
#include "rag/etcd_client.hpp"
#include "rag/config_manager.hpp"

namespace Rag {

    class RagCommandManager {
    public:
        RagCommandManager(std::shared_ptr<EtcdClient> etcd_client,
                         std::shared_ptr<ConfigManager> config_manager);
        ~RagCommandManager();

        // Comandos específicos del RAG
        bool showConfig();
        bool showCapabilities();
        bool updateSetting(const std::string& setting, const std::string& value);
        bool updateSecurityLevel(int level);
        bool toggleLLM(bool enable);
        bool changeLogLevel(const std::string& level);

        // Ayuda y validación
        std::vector<std::string> getAvailableCommands() const;
        bool isValidSetting(const std::string& setting) const;

    private:
        std::shared_ptr<EtcdClient> etcd_client_;
        std::shared_ptr<ConfigManager> config_manager_;

        bool sendConfigUpdateToEtcd();
    };

} // namespace Rag