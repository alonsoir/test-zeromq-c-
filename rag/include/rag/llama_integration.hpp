// include/rag/llama_integration.hpp
#pragma once
#include <string>
#include <vector>
#include <memory>

namespace rag {

    class LlamaIntegration {
    public:
        LlamaIntegration();
        ~LlamaIntegration();

        bool initialize(const std::string& model_path, size_t context_size = 4096);

        std::string processQuery(const std::string& query,
                                const std::vector<std::string>& context = {});

        bool validateCommandIntent(const std::string& command, const std::string& context);

        std::string generateSafeResponse(const std::string& query);

    private:
        class Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace rag