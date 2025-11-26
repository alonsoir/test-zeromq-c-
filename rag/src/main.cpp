#include "rag/etcd_client.hpp"
#include "rag/config_manager.hpp"
#include "rag/rag_command_manager.hpp"
#include "rag/whitelist_manager.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>

// Variables globales para manejo de señales
volatile sig_atomic_t running = 1;
std::shared_ptr<Rag::EtcdClient> global_etcd_client = nullptr;

void signalHandler(int signum) {
    std::cout << "\n🛑 Señal " << signum << " recibida. Cerrando..." << std::endl;
    running = 0;
}

void unregisterService() {
    if (global_etcd_client) {
        std::cout << "🌐 Desregistrando servicio..." << std::endl;
        if (global_etcd_client->unregisterService()) {
            std::cout << "✅ Servicio desregistrado correctamente" << std::endl;
        } else {
            std::cerr << "❌ Error al desregistrar servicio" << std::endl;
        }
    }
}

void processUserInput(const std::string& input,
                     std::shared_ptr<rag::WhitelistManager> whitelist_manager) {
    whitelist_manager->processCommand(input);
}

int main() {
    std::cout << "🚀 Iniciando RAG Security System - Arquitectura KISS" << std::endl;
    std::cout << "====================================================" << std::endl;

    // Configurar señales
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // Registrar cleanup
    std::atexit(unregisterService);

    // Inicializar ConfigManager (Singleton)
    auto& config_singleton = Rag::ConfigManager::getInstance();
    if (!config_singleton.loadFromFile("../config/rag-config.json")) {
        std::cerr << "❌ Error crítico: No se pudo cargar configuración RAG" << std::endl;
        return 1;
    }

    // ✅ CORREGIDO: Crear shared_ptr para ConfigManager
    auto config_manager = std::shared_ptr<Rag::ConfigManager>(&config_singleton, [](auto*) {
        // No delete porque es singleton
    });

    // ✅ CORREGIDO: Constructor con parámetros correctos
    auto etcd_endpoint = config_singleton.getEtcdEndpoint();
    auto etcd_client = std::make_shared<Rag::EtcdClient>(etcd_endpoint, "rag-security-system");
    global_etcd_client = etcd_client;

    // ✅ CORREGIDO: Ahora RagCommandManager recibe shared_ptr correctamente
    auto rag_command_manager = std::make_shared<Rag::RagCommandManager>(etcd_client, config_manager);

    // ✅ WHITELIST MANAGER como orquestador principal
    auto whitelist_manager = std::make_shared<rag::WhitelistManager>();
    whitelist_manager->registerManager("rag", rag_command_manager);

    // Inicializar etcd client
    if (!etcd_client->initialize()) {
        std::cerr << "❌ Error: No se pudo inicializar etcd client" << std::endl;
        return 1;
    }

    // Registrar componente en etcd
    std::cout << "🌐 Registrando componente en etcd-server..." << std::endl;
    if (!etcd_client->registerService()) {
        std::cerr << "❌ Error: No se pudo registrar en etcd-server" << std::endl;
        return 1;
    }

    std::cout << "\n✅ Sistema listo. Escribe 'help' para ver comandos disponibles." << std::endl;

    // Bucle principal de comandos
    std::string input;
    while (running) {
        std::cout << "\nSECURITY_SYSTEM> ";
        std::getline(std::cin, input);

        if (input.empty()) {
            continue;
        }

        processUserInput(input, whitelist_manager);

        // Pequeña pausa para evitar uso intensivo de CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Desregistrar componente
    unregisterService();

    std::cout << "👋 RAG Security System finalizado correctamente" << std::endl;

    return 0;
}