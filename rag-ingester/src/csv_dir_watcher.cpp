// csv_dir_watcher.cpp
// Day 69
//
// AUTHORS: Alonso Isidoro Roman + Claude (Anthropic)

#include "csv_dir_watcher.hpp"

#include <sys/inotify.h>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>

namespace rag_ingester {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

CsvDirWatcher::CsvDirWatcher(const std::string& dir_path, LineCallback callback)
    : dir_path_(dir_path)
    , callback_(std::move(callback))
{}

CsvDirWatcher::~CsvDirWatcher() {
    if (running_) stop();
}

// ---------------------------------------------------------------------------
// today_filename / today_filepath
// ---------------------------------------------------------------------------

std::string CsvDirWatcher::today_filename() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d") << ".csv";
    return ss.str();
}

std::string CsvDirWatcher::today_filepath() const {
    return dir_path_ + "/" + today_filename();
}

// ---------------------------------------------------------------------------
// start / stop
// ---------------------------------------------------------------------------

void CsvDirWatcher::start() {
    if (!fs::is_directory(dir_path_)) {
        throw std::runtime_error("CsvDirWatcher: directory not found: " + dir_path_);
    }
    running_ = true;
    thread_  = std::thread(&CsvDirWatcher::run, this);
}

void CsvDirWatcher::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

// ---------------------------------------------------------------------------
// drain_new_lines
// ---------------------------------------------------------------------------

off_t CsvDirWatcher::drain_new_lines(const std::string& filepath, off_t offset) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) return offset;

    f.seekg(offset);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) {
            if (line.back() == '\r') line.pop_back();
            callback_(line);
            ++lines_detected_;
        }
    }
    f.clear();
    return static_cast<off_t>(f.tellg());
}

// ---------------------------------------------------------------------------
// run — background inotify loop
// ---------------------------------------------------------------------------

void CsvDirWatcher::run() {
    int ifd = inotify_init1(IN_NONBLOCK);
    if (ifd < 0) return;

    int dir_wd = inotify_add_watch(ifd, dir_path_.c_str(),
                                   IN_CREATE | IN_MOVED_TO | IN_MODIFY);
    if (dir_wd < 0) { close(ifd); return; }

    // Seed with today's file — seek to current EOF
    std::string active_file   = today_filepath();
    off_t       active_offset = 0;
    if (fs::exists(active_file)) {
        active_offset = static_cast<off_t>(fs::file_size(active_file));
    }

    constexpr size_t BUF_SIZE = 4096;
    char buf[BUF_SIZE] __attribute__((aligned(__alignof__(struct inotify_event))));

    while (running_) {
        ssize_t n = read(ifd, buf, BUF_SIZE);
        if (n < 0) {
            if (errno == EAGAIN) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));

                // Check for day rotation
                std::string new_today = today_filepath();
                if (new_today != active_file && fs::exists(new_today)) {
                    active_file   = new_today;
                    active_offset = 0;
                    ++files_rotated_;
                }
                continue;
            }
            break;
        }

        const char* ptr = buf;
        while (ptr < buf + n) {
            const auto* ev = reinterpret_cast<const struct inotify_event*>(ptr);
            ptr += sizeof(struct inotify_event) + ev->len;

            if (ev->len == 0) continue;
            std::string evname(ev->name);

            if ((ev->mask & (IN_CREATE | IN_MOVED_TO)) && evname.size() > 4
                && evname.substr(evname.size() - 4) == ".csv") {
                std::string new_path = dir_path_ + "/" + evname;
                if (new_path != active_file) {
                    active_file   = new_path;
                    active_offset = 0;
                    ++files_rotated_;
                }
            }

            if ((ev->mask & IN_MODIFY) && evname.size() > 4
                && evname.substr(evname.size() - 4) == ".csv") {
                std::string modified_path = dir_path_ + "/" + evname;
                if (modified_path == active_file) {
                    active_offset = drain_new_lines(active_file, active_offset);
                }
            }
        }
    }

    inotify_rm_watch(ifd, dir_wd);
    close(ifd);
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

uint64_t CsvDirWatcher::lines_detected() const { return lines_detected_.load(); }
uint64_t CsvDirWatcher::files_rotated()  const { return files_rotated_.load();  }

} // namespace rag_ingester