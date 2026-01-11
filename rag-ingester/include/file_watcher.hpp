#pragma once

#include <string>
#include <functional>

namespace rag_ingester {

class FileWatcher {
public:
    using Callback = std::function<void(const std::string& filepath)>;
    
    FileWatcher(const std::string& directory, const std::string& pattern);
    ~FileWatcher();
    
    void start(Callback callback);
    void stop();
    
private:
    std::string directory_;
    std::string pattern_;
    bool running_{false};
};

} // namespace rag_ingester
