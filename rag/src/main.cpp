#include "rag/etcd_client.hpp"
#include "rag/config_manager.hpp"
#include "rag/rag_command_manager.hpp"
#include "rag/whitelist_manager.hpp"
#include <iostream>
#include <memory>

int main(int argc, char* argv[]) {
    std::cout << "🤖 RAG Security System - Iniciando..." << std::endl;

    // Inicializar componentes con shared_ptr
    auto config_manager = std::make_shared<Rag::ConfigManager>();
    auto etcd_client = std::make_shared<Rag::EtcdClient>("http://localhost:2379", "rag");
    auto rag_commands = std::make_shared<Rag::RagCommandManager>(etcd_client, config_manager);
    auto whitelist = std::make_shared<rag::WhitelistManager>();

    // Cargar configuración real - FAIL FAST
    if (!config_manager->loadFromFile("../config/rag-config.json")) {
        std::cerr << "❌ CRITICAL: Cannot load RAG configuration" << std::endl;
        std::cerr << "❌ SYSTEM EXITING - Configuration file is required" << std::endl;
        return 1; // Exit inmediato
    }

    // Inicializar etcd-client con configuración real - FAIL FAST
    if (!etcd_client->initialize()) {
        std::cerr << "❌ CRITICAL: Error initializing etcd-client" << std::endl;
        std::cerr << "❌ SYSTEM EXITING - etcd connection is required" << std::endl;
        return 1; // Exit inmediato
    }

    // Loop interactivo usando rag_commands
    std::string user_input;
    while (true) {
        std::cout << "\n👤 Usuario: ";
        std::getline(std::cin, user_input);

        // Procesar comando usando RagCommandManager
        processUserInput(user_input, rag_commands, whitelist);
    }

    return 0;
}