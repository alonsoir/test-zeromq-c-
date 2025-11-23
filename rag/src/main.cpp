#include <iostream>
#include <string>
#include <sstream>
#include "rag/llama_integration.hpp"

// Función para chat interactivo simple
void interactive_chat() {
    std::cout << "🚀 Iniciando Chat RAG Security System" << std::endl;
    std::cout << "=====================================" << std::endl;

    // Inicializar integración LLAMA
    LlamaIntegration llama;

    std::cout << "🔄 Cargando modelo..." << std::endl;
    if (!llama.loadModel("../models/default.gguf")) {
        std::cout << "⚠️  No se pudo cargar el modelo. Usando modo simulación." << std::endl;
    } else {
        std::cout << "✅ Modelo cargado correctamente" << std::endl;
    }

    std::cout << "\n💬 Modo Chat Activo" << std::endl;
    std::cout << "Escribe 'quit' o 'exit' para salir." << std::endl;
    std::cout << "=====================================" << std::endl;

    std::string input;
    while (true) {
        std::cout << "\n👤 Tú: ";
        std::getline(std::cin, input);

        // Salir si el usuario escribe quit/exit
        if (input == "quit" || input == "exit" || input == "salir") {
            std::cout << "👋 ¡Hasta luego!" << std::endl;
            break;
        }

        // Procesar vacío
        if (input.empty()) {
            continue;
        }

        // Procesar con LLAMA
        std::cout << "🤖 Procesando...";
        std::string response = llama.generateResponse(input);
        std::cout << "\r🤖 Asistente: " << response << std::endl;
    }
}

// Función para validar un comando específico
void validate_command(const std::string& command) {
    std::cout << "🔒 Validando comando: " << command << std::endl;

    LlamaIntegration llama;
    llama.loadModel("../models/default.gguf");

    // Crear prompt de seguridad
    std::string security_prompt = "Analiza la seguridad del siguiente comando de Linux: '" + command +
                                  "'. Responde solo con 'SEGURO', 'SOSPECHOSO' o 'PELIGROSO' y una breve explicación.";

    std::string analysis = llama.generateResponse(security_prompt);
    std::cout << "📊 Análisis de seguridad: " << analysis << std::endl;
}

// Función para procesar consulta específica
void process_query(const std::string& query) {
    std::cout << "🔍 Procesando consulta: " << query << std::endl;

    LlamaIntegration llama;
    llama.loadModel("../models/default.gguf");

    std::string response = llama.generateResponse(query);
    std::cout << "🤖 Respuesta: " << response << std::endl;
}

// Mostrar ayuda
void show_help() {
    std::cout << "📖 RAG Security System - Ayuda" << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << "Uso: ./rag-security [OPCIÓN] [ARGUMENTO]" << std::endl;
    std::cout << std::endl;
    std::cout << "Opciones:" << std::endl;
    std::cout << "  --chat                 Modo chat interactivo" << std::endl;
    std::cout << "  --query \"consulta\"     Procesar una consulta específica" << std::endl;
    std::cout << "  --validate \"comando\"  Validar seguridad de un comando" << std::endl;
    std::cout << "  --help                 Mostrar esta ayuda" << std::endl;
    std::cout << std::endl;
    std::cout << "Ejemplos:" << std::endl;
    std::cout << "  ./rag-security --chat" << std::endl;
    std::cout << "  ./rag-security --query \"¿Cómo configurar un firewall?\"" << std::endl;
    std::cout << "  ./rag-security --validate \"rm -rf /\"" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "🔐 RAG Security System v1.0" << std::endl;
    std::cout << "============================" << std::endl;

    // Modo por defecto: chat interactivo
    if (argc == 1) {
        interactive_chat();
        return 0;
    }

    // Procesar argumentos de línea de comandos
    std::string option = argv[1];

    if (option == "--chat") {
        interactive_chat();
    }
    else if (option == "--query" && argc > 2) {
        process_query(argv[2]);
    }
    else if (option == "--validate" && argc > 2) {
        validate_command(argv[2]);
    }
    else if (option == "--help") {
        show_help();
    }
    else {
        std::cout << "❌ Opción no válida. Usa --help para ver las opciones disponibles." << std::endl;
        return 1;
    }

    return 0;
}