#pragma once
#include "rag/command_manager.hpp"
#include "rag/rag_validator.hpp"
#include "metadata_reader.hpp"
#include <faiss/IndexFlat.h>
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
    void showHelp() const;
    // ========== NUEVOS MÉTODOS FAISS (Day 41) ==========
    void setFAISSIndices(
        faiss::IndexFlatL2* chronos,
        faiss::IndexFlatL2* sbert,
        faiss::IndexFlatL2* attack
    );

    void setMetadataReader(ml_defender::MetadataReader* metadata);

private:
    RagValidator validator_;

    // ========== NUEVOS MÉTODOS PRIVADOS FAISS ==========
    void handleQuerySimilar(const std::vector<std::string>& args);
    void handleListEvents(const std::vector<std::string>& args);
    void handleStats(const std::vector<std::string>& args);
    void handleInfo(const std::vector<std::string>& args);
    void handleRecent(const std::vector<std::string>& args);
    void handleSearch(const std::vector<std::string>& args);

    // ========== REFERENCIAS FAISS (Day 41) ==========
    faiss::IndexFlatL2* chronos_index_ = nullptr;
    faiss::IndexFlatL2* sbert_index_ = nullptr;
    faiss::IndexFlatL2* attack_index_ = nullptr;
    ml_defender::MetadataReader* metadata_reader_ = nullptr;
};

} // namespace Rag