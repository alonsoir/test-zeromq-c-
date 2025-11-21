// include/rag/security_context.hpp
#pragma once
#include <string>
#include <memory>

#include "whitelist_parser.hpp"
#include "etcd_client.hpp"
#include "llama_integration.hpp"
#include "config_manager.hpp"
#include "whitelist_manager.hpp"

namespace rag {

    class SecurityContext {
    public:
        SecurityContext();
        ~SecurityContext();

        bool initialize(const std::string& config_path);
        std::string processSecurityRequest(const std::string& query);
        void setSecurityLevel(int level);
        int getSecurityLevel() const;

    private:
        int security_level_;
    };

} // namespace rag