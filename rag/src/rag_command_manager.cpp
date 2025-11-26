#include "rag/rag_command_manager.hpp"
#include "rag/config_manager.hpp"
#include "rag/llama_integration.hpp"
#include <iostream>
#include <algorithm>

// Declaración externa de la instancia global de LlamaIntegration (namespace global)
extern std::unique_ptr<LlamaIntegration> llama_integration;

namespace Rag {

RagCommandManager::RagCommandManager() : validator_() {
    std::cout << "🔧 RagCommandManager inicializado" << std::endl;
}

RagCommandManager::~RagCommandManager() {
    std::cout << "🔧 RagCommandManager finalizado" << std::endl;
}

void RagCommandManager::processCommand(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cout << "❌ Comando RAG no especificado" << std::endl;
        return;
    }

    const std::string& command = args[0];

    if (command == "show_config") {
        showConfig(args);
    } else if (command == "show_capabilities") {
        showCapabilities(args);
    } else if (command == "update_setting") {
        updateSetting(args);
    } else if (command == "ask_llm") {
        askLLM(args);
    } else {
        std::cout << "❌ Comando RAG no reconocido: " << command << std::endl;
        std::cout << "💡 Comandos disponibles: show_config, update_setting, show_capabilities, ask_llm" << std::endl;
    }
}

void RagCommandManager::showConfig(const std::vector<std::string>& args) {
    std::cout << "\n🔧 CONFIGURACIÓN RAG - MOSTRANDO..." << std::endl;

    try {
        auto& config_manager = ConfigManager::getInstance();
        auto rag_config = config_manager.getRagConfig();
        auto etcd_config = config_manager.getEtcdConfig();

        std::cout << "📋 Configuración RAG:" << std::endl;
        std::cout << "   - Host: " << rag_config.host << std::endl;
        std::cout << "   - Port: " << rag_config.port << std::endl;
        std::cout << "   - Model: " << rag_config.model_name << std::endl;
        std::cout << "   - Embedding Dimension: " << rag_config.embedding_dimension << std::endl;

        std::cout << "📋 Configuración Etcd:" << std::endl;
        std::cout << "   - Host: " << etcd_config.host << std::endl;
        std::cout << "   - Port: " << etcd_config.port << std::endl;

        // Mostrar estado del LLM
        std::cout << "🤖 Estado LLM: " << (llama_integration ? "CARGADO" : "NO DISPONIBLE") << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error al mostrar configuración: " << e.what() << std::endl;
    }
}

void RagCommandManager::showCapabilities(const std::vector<std::string>& args) {
    std::cout << "\n🚀 CAPACIDADES DEL SISTEMA RAG" << std::endl;
    std::cout << "===============================" << std::endl;
    std::cout << "✅ Configuración persistente en JSON" << std::endl;
    std::cout << "✅ Comandos de configuración en tiempo real" << std::endl;
    std::cout << "✅ Validación robusta de configuraciones" << std::endl;
    std::cout << "✅ Integración con etcd (via WhiteListManager)" << std::endl;
    std::cout << "🤖 LLAMA Integration: " << (llama_integration ? "ACTIVA" : "INACTIVA") << std::endl;
    std::cout << "📊 Base Vectorial: PRÓXIMAMENTE (con logs del pipeline)" << std::endl;
    std::cout << "===============================" << std::endl;
}

void RagCommandManager::updateSetting(const std::vector<std::string>& args) {
    if (args.size() != 3) {
        std::cout << "❌ Error: Uso: rag update_setting <clave> <valor>" << std::endl;
        return;
    }

    const std::string& key = args[1];
    const std::string& value = args[2];

    // Usar el validador específico de RAG
    if (!validator_.validate(key, value)) {
        return;
    }

    auto& config_manager = ConfigManager::getInstance();
    std::string path = "rag." + key;

    if (config_manager.updateSetting(path, value)) {
        std::cout << "✅ Configuración actualizada: " << key << " = " << value << std::endl;
    } else {
        std::cout << "❌ Error al actualizar la configuración" << std::endl;
    }
}

void RagCommandManager::askLLM(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cout << "❌ Error: Uso: rag ask_llm <pregunta>" << std::endl;
        return;
    }

    // Reconstruir la pregunta completa
    std::string question;
    for (size_t i = 1; i < args.size(); ++i) {
        if (i > 1) question += " ";
        question += args[i];
    }

    std::cout << "🤖 Consultando LLM: \"" << question << "\"" << std::endl;

    // Verificar si LLAMA está disponible
    if (!llama_integration) {
        std::cout << "❌ LLAMA Integration no disponible" << std::endl;
        std::cout << "💡 Asegúrate de que el modelo TinyLlama esté en /vagrant/rag/models/" << std::endl;
        return;
    }

    try {
        // Generar respuesta usando LLAMA REAL
        std::string response = llama_integration->generateResponse(question);
        std::cout << "\n🤖 Respuesta: " << response << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error en LLAMA: " << e.what() << std::endl;
        std::cout << "⚠️  Fallo en la generación de respuesta" << std::endl;
    }
}

} // namespace Rag