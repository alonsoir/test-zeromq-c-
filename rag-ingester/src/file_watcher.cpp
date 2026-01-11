#include "file_watcher.hpp"
#include <spdlog/spdlog.h>

namespace rag_ingester {

FileWatcher::FileWatcher(const std::string& directory, const std::string& pattern)
    : directory_(directory), pattern_(pattern) {
    spdlog::info("FileWatcher created for: {}", directory);
}

FileWatcher::~FileWatcher() {
    stop();
}

void FileWatcher::start(Callback callback) {
    spdlog::info("TODO: FileWatcher::start() - inotify implementation");
    running_ = true;
}

void FileWatcher::stop() {
    spdlog::info("FileWatcher stopped");
    running_ = false;
}

} // namespace rag_ingester
