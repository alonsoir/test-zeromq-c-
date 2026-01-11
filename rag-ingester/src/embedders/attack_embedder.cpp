#include "embedders/attack_embedder.hpp"
#include "event_loader.hpp"
#include <spdlog/spdlog.h>

namespace rag_ingester {

AttackEmbedder::AttackEmbedder(const std::string& onnx_path, float benign_sample_rate)
    : onnx_path_(onnx_path), benign_sample_rate_(benign_sample_rate) {
    spdlog::info("AttackEmbedder initialized: {} (benign_sample_rate={})", 
                onnx_path, benign_sample_rate);
    // TODO: Load ONNX Runtime session
}

AttackEmbedder::~AttackEmbedder() {
    spdlog::info("AttackEmbedder destroyed");
}

std::vector<float> AttackEmbedder::embed(const Event& event) {
    spdlog::debug("TODO: AttackEmbedder::embed()");
    return std::vector<float>(256, 0.0f);  // Stub: 256-d zero vector
}

std::vector<std::vector<float>> AttackEmbedder::embed_batch(const std::vector<Event>& events) {
    spdlog::debug("TODO: AttackEmbedder::embed_batch() - {} events", events.size());
    return std::vector<std::vector<float>>(events.size(), std::vector<float>(256, 0.0f));
}

bool AttackEmbedder::should_embed(const Event& event) const {
    // TODO: Implement sampling logic
    return true;
}

} // namespace rag_ingester
