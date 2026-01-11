#include "event_loader.hpp"
#include <spdlog/spdlog.h>

namespace rag_ingester {

EventLoader::EventLoader(bool encrypted, bool compressed)
    : encrypted_(encrypted), compressed_(compressed) {
    spdlog::info("EventLoader initialized (encrypted={}, compressed={})", 
                encrypted, compressed);
}

std::vector<Event> EventLoader::load(const std::string& filepath) {
    spdlog::info("TODO: EventLoader::load() - {}", filepath);
    // TODO: crypto-transport integration
    return {};
}

} // namespace rag_ingester
