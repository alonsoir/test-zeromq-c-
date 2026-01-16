#include "embedders/chronos_embedder.hpp"
#include "event_loader.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace rag_ingester {

ChronosEmbedder::ChronosEmbedder(const std::string& onnx_path)
    : onnx_path_(onnx_path) {
    spdlog::info("ChronosEmbedder initialized: {} (INPUT_DIM={})", onnx_path, INPUT_DIM);
    // TODO: Load ONNX Runtime session
}

ChronosEmbedder::~ChronosEmbedder() {
    spdlog::info("ChronosEmbedder destroyed");
}

std::vector<float> ChronosEmbedder::embed(const Event& event) {
    // Day 38: Construct 103-dimensional input (101 core + 2 meta)
    std::vector<float> input;
    input.reserve(INPUT_DIM);
    
    // 101 core features from event.features
    input.insert(input.end(), event.features.begin(), event.features.end());
    
    // Meta feature #1: discrepancy_score (ADR-002)
    input.push_back(event.discrepancy_score);
    
    // Meta feature #2: verdict count (ADR-002)
    input.push_back(static_cast<float>(event.verdicts.size()));
    
    // Validate input size (defensive programming)
    if (input.size() != INPUT_DIM) {
        throw std::runtime_error(
            "ChronosEmbedder: Invalid input size " + std::to_string(input.size()) +
            " (expected " + std::to_string(INPUT_DIM) + ")"
        );
    }
    
    spdlog::debug("ChronosEmbedder::embed() - input validated ({} features)", input.size());
    
    // TODO: Pass input to ONNX Runtime model
    // For now, return stub (512-d zero vector)
    return std::vector<float>(OUTPUT_DIM, 0.0f);
}

std::vector<std::vector<float>> ChronosEmbedder::embed_batch(const std::vector<Event>& events) {
    spdlog::debug("ChronosEmbedder::embed_batch() - {} events", events.size());
    
    // TODO: Implement batched ONNX inference
    // For now, call embed() individually
    std::vector<std::vector<float>> results;
    results.reserve(events.size());
    
    for (const auto& event : events) {
        results.push_back(embed(event));
    }
    
    return results;
}

} // namespace rag_ingester
