#include "rag/rag_command_manager.hpp"
#include <iostream>

namespace Rag {

RagCommandManager::RagCommandManager(
    std::shared_ptr<EtcdClient> etcd_client,
    std::shared_ptr<ConfigManager> config_manager)
    : etcd_client_(etcd_client), config_manager_(config_manager) {
    std::cout << "[RAG-CMD] ✅ RagCommandManager inicializado (modo pasivo)" << std::endl;
}

RagCommandManager::~RagCommandManager() = default;

// Métodos mínimos (vacíos o con stubs)
bool RagCommandManager::showConfig() {
    std::cout << "[RAG-CMD] 📋 showConfig() — pendiente de implementación" << std::endl;
    return true;
}

bool RagCommandManager::showCapabilities() {
    std::cout << "[RAG-CMD] 🎯 showCapabilities() — pendiente" << std::endl;
    return true;
}

bool RagCommandManager::updateSetting(const std::string& setting, const std::string& value) {
    std::cout << "[RAG-CMD] 🔧 updateSetting(" << setting << ", " << value << ") — pendiente" << std::endl;
    return true;
}

bool RagCommandManager::updateSecurityLevel(int level) {
    std::cout << "[RAG-CMD] 🔐 updateSecurityLevel(" << level << ") — pendiente" << std::endl;
    return true;
}

bool RagCommandManager::toggleLLM(bool enable) {
    std::cout << "[RAG-CMD] 🤖 toggleLLM(" << (enable ? "ON" : "OFF") << ") — pendiente" << std::endl;
    return true;
}

bool RagCommandManager::changeLogLevel(const std::string& level) {
    std::cout << "[RAG-CMD] 📝 changeLogLevel(" << level << ") — pendiente" << std::endl;
    return true;
}

std::vector<std::string> RagCommandManager::getAvailableCommands() const {
    return {"showConfig", "showCapabilities", "updateSetting", "help"};
}

bool RagCommandManager::isValidSetting(const std::string& setting) const {
    // Stub: permitir algunos conocidos
    return setting == "model_name" || setting == "host" || setting == "port";
}

bool RagCommandManager::sendConfigUpdateToEtcd() {
    std::cout << "[RAG-CMD] 📤 sendConfigUpdateToEtcd() — pendiente" << std::endl;
    return true;
}

} // namespace Rag