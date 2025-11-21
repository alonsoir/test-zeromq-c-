// src/security_context.cpp
#include "rag/security_context.hpp"
#include <iostream>

namespace rag {

    SecurityContext::SecurityContext() : security_level_(3) {
        std::cout << "SecurityContext: Created" << std::endl;
    }

    SecurityContext::~SecurityContext() = default;

    bool SecurityContext::initialize(const std::string& config_path) {
        std::cout << "SecurityContext: Initializing with config " << config_path << std::endl;
        return true;
    }

    std::string SecurityContext::processSecurityRequest(const std::string& query) {
        std::cout << "SecurityContext: Processing request: " << query << std::endl;
        return "Processed: " + query;
    }

    void SecurityContext::setSecurityLevel(int level) {
        security_level_ = level;
        std::cout << "SecurityContext: Security level set to " << level << std::endl;
    }

    int SecurityContext::getSecurityLevel() const {
        return security_level_;
    }

} // namespace rag