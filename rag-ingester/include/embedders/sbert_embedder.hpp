#pragma once

#include <vector>
#include <string>

namespace rag_ingester {

struct Event;

class SBERTEmbedder {
public:
    explicit SBERTEmbedder(const std::string& onnx_path);
    ~SBERTEmbedder();
    
    std::vector<float> embed(const Event& event);
    std::vector<std::vector<float>> embed_batch(const std::vector<Event>& events);
    
private:
    std::string onnx_path_;
    // TODO: ONNX Runtime session
};

} // namespace rag_ingester
