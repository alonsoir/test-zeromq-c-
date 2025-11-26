// rag/include/rag/whitelist_manager.hpp
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Rag {
    class RagCommandManager;  // ✅ Forward declaration - suficiente para el header
}

namespace rag {

    class WhitelistManager {
    public:
        WhitelistManager();
        ~WhitelistManager();

        // Registrar managers especializados
        void registerManager(const std::string& component,
                            std::shared_ptr<Rag::RagCommandManager> manager);

        void processCommand(const std::string& command);
        void showHelp() const;

    private:
        std::unordered_map<std::string, std::shared_ptr<Rag::RagCommandManager>> managers_;

        struct ParsedCommand {
            std::string component;
            std::string action;
            std::string parameters;
            bool valid = false;
        };

        ParsedCommand parseCommand(const std::string& command) const;
    };

} // namespace rag