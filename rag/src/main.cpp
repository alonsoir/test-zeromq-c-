// src/main.cpp
#include "rag/rag_command_system.hpp"
#include "rag/security_context.hpp"
#include "rag/query_validator.hpp"
#include "rag/response_generator.hpp"
#include <iostream>

int main() {
    std::cout << "🚀 RAG Security System Starting..." << std::endl;
    
    // Inicializar componentes
    rag::SecurityContext security;
    rag::QueryValidator validator;
    rag::ResponseGenerator generator;
    
    security.initialize("config/rag_config.json");
    security.setSecurityLevel(5);
    
    // Procesar ejemplo
    std::string testQuery = "SELECT * FROM users";
    if (validator.validate(testQuery)) {
        auto response = security.processSecurityRequest(testQuery);
        std::cout << "Response: " << response << std::endl;
    }
    
    std::cout << "✅ RAG System Ready!" << std::endl;
    return 0;
}