#pragma once

#include <vector>
#include <string>

namespace rag_ingester {

struct Event;

class AttackEmbedder {
public:
    explicit AttackEmbedder(const std::string& onnx_path, float benign_sample_rate);
    ~AttackEmbedder();
    
    std::vector<float> embed(const Event& event);
    std::vector<std::vector<float>> embed_batch(const std::vector<Event>& events);
    
    bool should_embed(const Event& event) const;
    
private:
    std::string onnx_path_;
    float benign_sample_rate_;
    // TODO: ONNX Runtime session
};

} // namespace rag_ingester
