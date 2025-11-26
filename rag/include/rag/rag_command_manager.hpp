#pragma once
#include <string>
#include <memory>
#include <vector>

namespace Rag {
    class EtcdClient;
    class ConfigManager;

    class RagCommandManager {
    public:
        RagCommandManager(std::shared_ptr<EtcdClient> etcd_client,
                         std::shared_ptr<ConfigManager> config_manager);
        ~RagCommandManager();

        // ✅ MÉTODOS PRINCIPALES - Interface limpia
        void showConfig();
        void showCapabilities();
        void updateSetting(const std::string& input);

        // ✅ MÉTODO PARA PROCESAR COMANDOS DIRECTAMENTE
        void processCommand(const std::string& command);

    private:
        std::shared_ptr<EtcdClient> etcd_client_;
        std::shared_ptr<ConfigManager> config_manager_;
    };
}