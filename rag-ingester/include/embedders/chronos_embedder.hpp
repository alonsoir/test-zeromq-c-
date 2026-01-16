#pragma once

#include <vector>
#include <string>

namespace rag_ingester {

struct Event;

class ChronosEmbedder {
public:
    static constexpr size_t INPUT_DIM = 103;  // 101 core + 2 meta (Day 38)
    static constexpr size_t OUTPUT_DIM = 512; // Chronos embedding dimension
    
    explicit ChronosEmbedder(const std::string& onnx_path);
    ~ChronosEmbedder();
    
    std::vector<float> embed(const Event& event);
    std::vector<std::vector<float>> embed_batch(const std::vector<Event>& events);
    
private:
    std::string onnx_path_;
    // TODO: ONNX Runtime session
};

} // namespace rag_ingester
