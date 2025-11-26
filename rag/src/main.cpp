#include "rag/whitelist_manager.hpp"
#include "rag/rag_command_manager.hpp"
#include "rag/config_manager.hpp"
#include "rag/llama_integration.hpp"
#include <iostream>
#include <csignal>
#include <memory>

// Declarar la instancia global de LlamaIntegration (está en namespace global)
std::unique_ptr<LlamaIntegration> llama_integration;
std::unique_ptr<Rag::WhiteListManager> whitelist_manager;

void signalHandler(int signal) {
    std::cout << "\n🛑 Señal " << signal << " recibida. Cerrando..." << std::endl;
    if (whitelist_manager) {
        whitelist_manager.reset();
    }
    if (llama_integration) {
        llama_integration.reset();
    }
    exit(0);
}

int main() {
    std::cout << "🚀 Iniciando RAG Security System - Arquitectura Centralizada" << std::endl;
    std::cout << "============================================================" << std::endl;

    // Configurar manejador de señales
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    try {
        // Cargar configuración
        auto& config_manager = Rag::ConfigManager::getInstance();
        if (!config_manager.loadFromFile("../config/rag-config.json")) {
            std::cerr << "❌ Error crítico: No se pudo cargar la configuración" << std::endl;
            return 1;
        }

        // Inicializar LLAMA Integration con el modelo real
        llama_integration = std::make_unique<LlamaIntegration>();
        auto rag_config = config_manager.getRagConfig();

        std::cout << "🤖 Intentando cargar modelo LLM: " << rag_config.model_name << std::endl;
        if (llama_integration->loadModel("../models/" + rag_config.model_name)) {
            std::cout << "✅ Modelo LLM cargado exitosamente" << std::endl;
        } else {
            std::cout << "❌ No se pudo cargar el modelo LLM" << std::endl;
            // No salimos, continuamos sin LLAMA
        }

        // Inicializar WhiteListManager (único con acceso a etcd)
        whitelist_manager = std::make_unique<Rag::WhiteListManager>();

        // Registrar RagCommandManager (sin acceso directo a etcd)
        auto rag_manager = std::make_shared<Rag::RagCommandManager>();
        whitelist_manager->registerCommandManager("rag", rag_manager);

        // Inicializar sistema
        if (!whitelist_manager->initialize()) {
            std::cerr << "❌ Error inicializando WhiteListManager" << std::endl;
            return 1;
        }

        std::cout << "\n✅ Sistema listo. Escribe 'help' para ver comandos disponibles." << std::endl;

        // Bucle principal de comandos
        std::string input;
        while (true) {
            std::cout << "\nSECURITY_SYSTEM> ";
            std::getline(std::cin, input);

            if (input.empty()) continue;

            whitelist_manager->processCommand(input);
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ Error crítico: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}