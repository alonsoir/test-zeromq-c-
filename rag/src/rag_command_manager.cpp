#include "rag/rag_command_manager.hpp"
#include "rag/etcd_client.hpp"
#include "rag/config_manager.hpp"
#include <iostream>
#include <algorithm>

namespace Rag {

RagCommandManager::RagCommandManager(std::shared_ptr<EtcdClient> etcd_client,
                                     std::shared_ptr<ConfigManager> config_manager)
    : etcd_client_(etcd_client), config_manager_(config_manager) {
    std::cout << "🔧 RagCommandManager inicializado" << std::endl;
}

RagCommandManager::~RagCommandManager() {
    std::cout << "🔧 RagCommandManager finalizado" << std::endl;
}

void RagCommandManager::showConfig() {
    std::cout << "\n🔧 CONFIGURACIÓN RAG - MOSTRANDO..." << std::endl;

    try {
        // Obtener configuración RAG
        auto rag_config = config_manager_->getRagConfig();
        auto etcd_config = config_manager_->getEtcdConfig();

        std::cout << "📋 Configuración RAG:" << std::endl;
        std::cout << "   - Host: " << rag_config.host << std::endl;
        std::cout << "   - Port: " << rag_config.port << std::endl;
        std::cout << "   - Model: " << rag_config.model_name << std::endl;
        std::cout << "   - Embedding Dimension: " << rag_config.embedding_dimension << std::endl;

        std::cout << "📋 Configuración Etcd:" << std::endl;
        std::cout << "   - Host: " << etcd_config.host << std::endl;
        std::cout << "   - Port: " << etcd_config.port << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error al mostrar configuración: " << e.what() << std::endl;
    }
}

void RagCommandManager::showCapabilities() {
    std::cout << "\n🚀 CAPACIDADES DEL SISTEMA RAG" << std::endl;
    std::cout << "===============================" << std::endl;
    std::cout << "✅ Registro automático en etcd-server" << std::endl;
    std::cout << "✅ Configuración persistente en JSON" << std::endl;
    std::cout << "✅ Comandos de configuración en tiempo real" << std::endl;
    std::cout << "✅ Manejo robusto de señales (Ctrl+C)" << std::endl;
    std::cout << "✅ Arquitectura WhiteListManager" << std::endl;
    std::cout << "🌐 Conectado a etcd-server: " << (etcd_client_->is_connected() ? "Sí" : "No") << std::endl;
    std::cout << "===============================" << std::endl;
}

void RagCommandManager::updateSetting(const std::string& input) {
    // Parsear input: "clave valor"
    size_t space_pos = input.find(' ');
    if (space_pos == std::string::npos) {
        std::cout << "❌ Formato incorrecto. Use: update_setting <clave> <valor>" << std::endl;
        std::cout << "   Ejemplo: rag update_setting model_path /nuevo/path/al/modelo" << std::endl;
        return;
    }

    std::string key = input.substr(0, space_pos);
    std::string value = input.substr(space_pos + 1);

    // Eliminar espacios en blanco sobrantes
    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);
    value.erase(0, value.find_first_not_of(" \t"));
    value.erase(value.find_last_not_of(" \t") + 1);

    if (key.empty() || value.empty()) {
        std::cout << "❌ Clave o valor no pueden estar vacíos" << std::endl;
        return;
    }

    std::cout << "🔄 Actualizando configuración RAG..." << std::endl;
    std::cout << "   Clave: " << key << std::endl;
    std::cout << "   Valor: " << value << std::endl;

    // TODO: Implementar actualización real cuando esté disponible en ConfigManager
    std::cout << "📋 (Actualización de configuración llamada correctamente)" << std::endl;
    std::cout << "💡 Nota: La persistencia real se implementará próximamente" << std::endl;
}

void RagCommandManager::processCommand(const std::string& command) {
    if (command == "show_config") {
        showConfig();
    } else if (command == "show_capabilities") {
        showCapabilities();
    } else if (command.find("update_setting") == 0) {
        std::string params = command.substr(14); // "update_setting" tiene 14 caracteres
        params.erase(0, params.find_first_not_of(" \t"));
        updateSetting(params);
    } else {
        std::cout << "❌ Comando RAG no reconocido: " << command << std::endl;
        std::cout << "💡 Comandos disponibles: show_config, update_setting, show_capabilities" << std::endl;
    }
}

} // namespace Rag