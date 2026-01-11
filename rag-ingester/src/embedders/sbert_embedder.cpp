#include "embedders/sbert_embedder.hpp"
#include "event_loader.hpp"
#include <spdlog/spdlog.h>

namespace rag_ingester {

SBERTEmbedder::SBERTEmbedder(const std::string& onnx_path)
    : onnx_path_(onnx_path) {
    spdlog::info("SBERTEmbedder initialized: {}", onnx_path);
    // TODO: Load ONNX Runtime session
}

SBERTEmbedder::~SBERTEmbedder() {
    spdlog::info("SBERTEmbedder destroyed");
}

std::vector<float> SBERTEmbedder::embed(const Event& event) {
    spdlog::debug("TODO: SBERTEmbedder::embed()");
    return std::vector<float>(384, 0.0f);  // Stub: 384-d zero vector
}

std::vector<std::vector<float>> SBERTEmbedder::embed_batch(const std::vector<Event>& events) {
    spdlog::debug("TODO: SBERTEmbedder::embed_batch() - {} events", events.size());
    return std::vector<std::vector<float>>(events.size(), std::vector<float>(384, 0.0f));
}

} // namespace rag_ingester
