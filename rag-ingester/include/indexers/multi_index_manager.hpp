#pragma once

#include <memory>
#include <vector>

namespace faiss {
    class Index;
}

namespace rag_ingester {

class MultiIndexManager {
public:
    MultiIndexManager();
    ~MultiIndexManager();
    
    void add_chronos(const std::vector<float>& embeddings);
    void add_sbert(const std::vector<float>& embeddings);
    void add_entity_benign(const std::vector<float>& embeddings);
    void add_entity_malicious(const std::vector<float>& embeddings);
    
    void save_all(const std::string& path);
    void load_all(const std::string& path);
    
private:
    std::unique_ptr<faiss::Index> chronos_index_;
    std::unique_ptr<faiss::Index> sbert_index_;
    std::unique_ptr<faiss::Index> entity_benign_index_;
    std::unique_ptr<faiss::Index> entity_malicious_index_;
};

} // namespace rag_ingester
