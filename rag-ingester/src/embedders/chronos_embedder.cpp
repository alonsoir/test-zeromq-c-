#include "embedders/chronos_embedder.hpp"
#include "event_loader.hpp"
#include <spdlog/spdlog.h>

namespace rag_ingester {

ChronosEmbedder::ChronosEmbedder(const std::string& onnx_path)
    : onnx_path_(onnx_path) {
    spdlog::info("ChronosEmbedder initialized: {}", onnx_path);
    // TODO: Load ONNX Runtime session
}

ChronosEmbedder::~ChronosEmbedder() {
    spdlog::info("ChronosEmbedder destroyed");
}

std::vector<float> ChronosEmbedder::embed(const Event& event) {
    spdlog::debug("TODO: ChronosEmbedder::embed()");
    return std::vector<float>(512, 0.0f);  // Stub: 512-d zero vector
}

std::vector<std::vector<float>> ChronosEmbedder::embed_batch(const std::vector<Event>& events) {
    spdlog::debug("TODO: ChronosEmbedder::embed_batch() - {} events", events.size());
    return std::vector<std::vector<float>>(events.size(), std::vector<float>(512, 0.0f));
}

} // namespace rag_ingester
