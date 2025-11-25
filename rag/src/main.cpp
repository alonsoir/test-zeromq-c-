#include "rag/etcd_client.hpp"
#include "rag/config_manager.hpp"
#include "rag/rag_command_manager.hpp"
#include "rag/whitelist_manager.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <csignal>
#include <atomic>

// Variables globales para manejo de shutdown
std::atomic<bool> shutdown_requested{false};
std::atomic<int> signal_count{0};
std::shared_ptr<Rag::EtcdClient> global_etcd_client = nullptr;
std::shared_ptr<Rag::RagCommandManager> global_rag_commands = nullptr;

// Manejador de señales para Ctrl+C
void signalHandler(int signal) {
    int current_count = ++signal_count;

    if (current_count == 1) {
        std::cout << "\n🛑 Señal " << signal << " recibida. Cerrando limpiamente..." << std::endl;
        shutdown_requested = true;
    } else if (current_count >= 3) {
        std::cout << "\n🚨 Cierre forzado..." << std::endl;
        exit(1);
    } else {
        std::cout << "\n🔄 Cierre en progreso... (presiona Ctrl+C nuevamente para forzar)" << std::endl;
    }
}

// Función para desregistrar el servicio de etcd - IMPLEMENTACIÓN REAL
void unregisterService() {
    if (global_etcd_client) {
        std::cout << "📝 Desregistrando servicio de etcd..." << std::endl;
        if (global_etcd_client->unregisterService()) {
            std::cout << "✅ Servicio desregistrado correctamente" << std::endl;
        } else {
            std::cerr << "⚠️  No se pudo desregistrar el servicio correctamente" << std::endl;
        }
    }
}

// Función para procesar input del usuario
void processUserInput(const std::string& input,
                     std::shared_ptr<Rag::RagCommandManager> rag_commands,
                     std::shared_ptr<rag::WhitelistManager> whitelist) {

    if (input == "help" || input == "?") {
        std::cout << "📚 Comandos disponibles:" << std::endl;
        std::cout << "  show_config         - Mostrar configuración RAG" << std::endl;
        std::cout << "  show_capabilities   - Mostrar capacidades" << std::endl;
        std::cout << "  update_setting <key> <value> - Actualizar configuración" << std::endl;
        std::cout << "  security_level <1-5> - Cambiar nivel de seguridad" << std::endl;
        std::cout << "  toggle_llm <on/off> - Activar/desactivar LLM" << std::endl;
        std::cout << "  log_level <level>   - Cambiar nivel de log" << std::endl;
        std::cout << "  available_commands  - Comandos disponibles" << std::endl;
        std::cout << "  status              - Estado del sistema" << std::endl;
        std::cout << "  exit/quit           - Salir" << std::endl;
        return;
    }

    if (input == "show_config") {
        if (rag_commands) {
            rag_commands->showConfig();
        }
    }
    else if (input == "show_capabilities") {
        if (rag_commands) {
            rag_commands->showCapabilities();
        }
    }
    else if (input.find("update_setting") == 0) {
        if (rag_commands) {
            size_t pos1 = input.find(' ');
            size_t pos2 = input.find(' ', pos1 + 1);
            if (pos1 != std::string::npos && pos2 != std::string::npos) {
                std::string setting = input.substr(pos1 + 1, pos2 - pos1 - 1);
                std::string value = input.substr(pos2 + 1);
                rag_commands->updateSetting(setting, value);
            } else {
                std::cout << "❌ Uso: update_setting <clave> <valor>" << std::endl;
            }
        }
    }
    else if (input.find("security_level") == 0) {
        if (rag_commands) {
            try {
                int level = std::stoi(input.substr(14));
                rag_commands->updateSecurityLevel(level);
            } catch (...) {
                std::cout << "❌ Uso: security_level <1-5>" << std::endl;
            }
        }
    }
    else if (input.find("toggle_llm") == 0) {
        if (rag_commands) {
            std::string arg = input.substr(11);
            if (arg == "on" || arg == "true" || arg == "1") {
                rag_commands->toggleLLM(true);
            } else if (arg == "off" || arg == "false" || arg == "0") {
                rag_commands->toggleLLM(false);
            } else {
                std::cout << "❌ Uso: toggle_llm <on/off>" << std::endl;
            }
        }
    }
    else if (input.find("log_level") == 0) {
        if (rag_commands) {
            std::string level = input.substr(10);
            rag_commands->changeLogLevel(level);
        }
    }
    else if (input == "available_commands") {
        if (rag_commands) {
            auto commands = rag_commands->getAvailableCommands();
            std::cout << "🛠️ Comandos disponibles:" << std::endl;
            for (const auto& cmd : commands) {
                std::cout << "  - " << cmd << std::endl;
            }
        }
    }
    else if (input == "status") {
        std::cout << "📊 Estado del sistema RAG:" << std::endl;
        std::cout << "  - ✅ Configuración cargada" << std::endl;
        std::cout << "  - " << (global_etcd_client && global_etcd_client->is_connected() ? "✅" : "❌")
                  << " Conexión etcd" << std::endl;
        std::cout << "  - ✅ Comandos activos" << std::endl;
        std::cout << "  - 🟢 Sistema operativo" << std::endl;
    }
    else if (input == "exit" || input == "quit") {
        std::cout << "👋 Iniciando cierre del sistema..." << std::endl;
        shutdown_requested = true;
    }
    else {
        std::cout << "❌ Comando no reconocido: " << input << std::endl;
        std::cout << "💡 Escribe 'help' para ver comandos disponibles" << std::endl;
    }

    // Usar whitelist para evitar warning
    if (whitelist) {
        // En el futuro: verificar comando en whitelist
        // whitelist->isAllowed(input);
    }
}

int main(int argc, char* argv[]) {
    // Registrar manejadores de señales para shutdown limpio
    std::signal(SIGINT, signalHandler);   // Ctrl+C
    std::signal(SIGTERM, signalHandler);  // kill command

    // Usar parámetros para evitar warnings
    if (argc > 1) {
        std::cout << "Argumentos: " << argv[1] << std::endl;
    }

    std::cout << "🤖 RAG Security System - Iniciando..." << std::endl;

    try {
        // ✅ FAIL-FAST: Cargar configuración PRIMERO
        std::cout << "📁 Cargando configuración desde rag-config.json..." << std::endl;

        auto& config_manager = Rag::ConfigManager::getInstance();

        if (!config_manager.loadFromFile("../config/rag-config.json")) {
            throw std::runtime_error("Cannot load RAG configuration");
        }

        std::cout << "✅ Configuración cargada exitosamente" << std::endl;

        // Mostrar información de configuración
        auto rag_config = config_manager.getRagConfig();
        auto etcd_config = config_manager.getEtcdConfig();

        std::cout << "🔧 Configuración RAG:" << std::endl;
        std::cout << "   - Modelo: " << rag_config.model_name << std::endl;
        std::cout << "   - Host: " << rag_config.host << ":" << rag_config.port << std::endl;
        std::cout << "🔗 Configuración etcd:" << std::endl;
        std::cout << "   - Host: " << etcd_config.host << ":" << etcd_config.port << std::endl;

        // ✅ FAIL-FAST: Inicializar etcd-client
        std::cout << "🔗 Inicializando conexión etcd..." << std::endl;

        std::string etcd_url = "http://" + etcd_config.host + ":" + std::to_string(etcd_config.port);
        auto etcd_client = std::make_shared<Rag::EtcdClient>(etcd_url, "rag");

        if (!etcd_client->initialize()) {
            throw std::runtime_error("Error initializing etcd-client");
        }

        std::cout << "✅ Conexión etcd establecida" << std::endl;

        // ✅ Registrar servicio en etcd
        std::cout << "📝 Registrando servicio en etcd..." << std::endl;
        if (!etcd_client->registerService()) {
            std::cerr << "⚠️  Advertencia: No se pudo registrar servicio en etcd" << std::endl;
        } else {
            std::cout << "✅ Servicio registrado en etcd" << std::endl;
        }

        // ✅ Guardar referencia global para el desregistro
        global_etcd_client = etcd_client;

        // ✅ Crear shared_ptr para ConfigManager
        auto config_manager_ptr = std::shared_ptr<Rag::ConfigManager>(
            &config_manager, [](Rag::ConfigManager*) {
                // No delete - es Singleton
            });

        // ✅ Inicializar RagCommandManager con shared_ptr
        auto rag_commands = std::make_shared<Rag::RagCommandManager>(
            etcd_client, config_manager_ptr);

        // ✅ Guardar referencia global
        global_rag_commands = rag_commands;

        auto whitelist = std::make_shared<rag::WhitelistManager>();

        std::cout << "🎯 Sistema RAG inicializado correctamente" << std::endl;
        std::cout << "💡 Escribe 'help' para ver comandos disponibles" << std::endl;
        std::cout << "💡 Usa Ctrl+C para salir limpiamente" << std::endl;

        // Loop interactivo con verificación de shutdown
        std::string user_input;
        while (!shutdown_requested) {
            std::cout << "\n👤 Usuario: ";

            if (!std::getline(std::cin, user_input)) {
                if (std::cin.eof()) {
                    std::cout << "\n📋 EOF detectado (Ctrl+D). Saliendo..." << std::endl;
                    break;
                }
                continue;
            }

            processUserInput(user_input, rag_commands, whitelist);
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ ERROR CRÍTICO: " << e.what() << std::endl;
        std::cerr << "🚨 Saliendo con error..." << std::endl;
        return 1;
    }

    // ✅ Desregistrar servicio antes de salir
    std::cout << "\n🔴 Iniciando secuencia de cierre..." << std::endl;
    unregisterService();

    // Limpiar recursos globales
    global_rag_commands = nullptr;
    global_etcd_client = nullptr;

    std::cout << "👋 Sistema RAG cerrado correctamente" << std::endl;
    return 0;
}