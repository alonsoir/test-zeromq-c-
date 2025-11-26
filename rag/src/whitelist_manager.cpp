// rag/src/whitelist_manager.cpp - CORRECTO ASÍ
#include "rag/whitelist_manager.hpp"
#include "rag/rag_command_manager.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>

namespace rag {

WhitelistManager::WhitelistManager() {
    std::cout << "🔧 Inicializando WhiteListManager..." << std::endl;
}

WhitelistManager::~WhitelistManager() {
    std::cout << "🔧 Finalizando WhiteListManager..." << std::endl;
}

void WhitelistManager::registerManager(const std::string& component,
                                      std::shared_ptr<Rag::RagCommandManager> manager) {
    managers_[component] = manager;
    std::cout << "✅ Registrado manager para componente: " << component << std::endl;
}

void WhitelistManager::processCommand(const std::string& command) {
    if (command.empty()) return;

    // Comandos globales del sistema
    if (command == "help" || command == "?") {
        showHelp();
        return;
    }

    if (command == "exit" || command == "quit") {
        std::cout << "👋 Saliendo del sistema..." << std::endl;
        exit(0);
    }

    // Parsear comando específico de componente
    auto parsed = parseCommand(command);

    if (!parsed.valid) {
        std::cout << "❌ Comando no válido: " << command << std::endl;
        showHelp();
        return;
    }

    // Enrutar al manager correspondiente
    auto it = managers_.find(parsed.component);
    if (it == managers_.end()) {
        std::cout << "❌ Componente no encontrado: " << parsed.component << std::endl;
        std::cout << "   Componentes disponibles: ";
        for (const auto& [comp, _] : managers_) {
            std::cout << comp << " ";
        }
        std::cout << std::endl;
        return;
    }

    // Enrutamiento básico - esto se puede refinar después
    if (parsed.component == "rag") {
        if (parsed.action == "show_config") {
            it->second->showConfig();
        } else if (parsed.action == "show_capabilities") {
            it->second->showCapabilities();
        } else if (parsed.action == "update_setting") {
            it->second->updateSetting(parsed.parameters);
        } else {
            std::cout << "❌ Acción no reconocida para 'rag': " << parsed.action << std::endl;
        }
    }
}

WhitelistManager::ParsedCommand WhitelistManager::parseCommand(const std::string& command) const {
    ParsedCommand result;
    std::istringstream iss(command);
    std::vector<std::string> tokens;
    std::string token;

    // Tokenizar
    while (std::getline(iss, token, ' ')) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }

    if (tokens.size() < 2) {
        // Comando demasiado corto
        return result;
    }

    result.component = tokens[0];
    result.action = tokens[1];

    // Reconstruir parámetros (si hay más de 2 tokens)
    if (tokens.size() > 2) {
        for (size_t i = 2; i < tokens.size(); ++i) {
            if (i > 2) result.parameters += " ";
            result.parameters += tokens[i];
        }
    }

    result.valid = true;
    return result;
}

void WhitelistManager::showHelp() const {
    std::cout << "\n🎯 SISTEMA DE SEGURIDAD - COMANDOS DISPONIBLES" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "rag show_config           - Mostrar configuración RAG" << std::endl;
    std::cout << "rag update_setting <k> <v> - Actualizar setting RAG" << std::endl;
    std::cout << "rag show_capabilities     - Mostrar capacidades RAG" << std::endl;
    std::cout << "help                      - Mostrar esta ayuda" << std::endl;
    std::cout << "exit                      - Salir del sistema" << std::endl;
    std::cout << "==============================================" << std::endl;

    if (!managers_.empty()) {
        std::cout << "Componentes activos: ";
        for (const auto& [comp, _] : managers_) {
            std::cout << comp << " ";
        }
        std::cout << std::endl;
    }
}

} // namespace rag