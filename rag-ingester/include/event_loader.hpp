#pragma once

#include <string>
#include <vector>

namespace rag_ingester {

struct Event {
    uint64_t id;
    std::vector<float> features;  // 83 features
    std::string classification;
    // TODO: Full protobuf structure
};

class EventLoader {
public:
    EventLoader(bool encrypted, bool compressed);
    
    std::vector<Event> load(const std::string& filepath);
    
private:
    bool encrypted_;
    bool compressed_;
};

} // namespace rag_ingester
