#pragma once
#include "rag/command_manager.hpp"
#include "rag/rag_validator.hpp"
#include <memory>

namespace Rag {

class RagCommandManager : public CommandManager {
public:
    RagCommandManager();
    ~RagCommandManager();

    void processCommand(const std::vector<std::string>& args) override;
    void showConfig(const std::vector<std::string>& args) override;
    void updateSetting(const std::vector<std::string>& args) override;
    void showCapabilities(const std::vector<std::string>& args) override;
    void askLLM(const std::vector<std::string>& args);

private:
    RagValidator validator_;
};

} // namespace Rag