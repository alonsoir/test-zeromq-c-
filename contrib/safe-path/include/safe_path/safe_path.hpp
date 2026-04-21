#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <fcntl.h>

namespace argus::safe_path {

[[nodiscard]] inline std::string resolve(
    const std::string& path,
    const std::string& allowed_prefix)
{
    namespace fs = std::filesystem;

    if (path.empty()) {
        throw std::runtime_error("[safe_path] Empty path rejected");
    }

    const auto canonical = fs::weakly_canonical(fs::path(path)).string();

    std::string prefix = allowed_prefix;
    if (!prefix.empty() && prefix.back() != '/') {
        prefix += '/';
    }

    if (canonical.rfind(prefix, 0) != 0) {
        throw std::runtime_error(
            "[safe_path] SECURITY VIOLATION — path traversal rejected\n"
            "  requested : '" + path + "'\n"
            "  resolved  : '" + canonical + "'\n"
            "  allowed   : '" + prefix + "'\n"
            "  ACTION    : Pipeline halt. Administrator notified.");
    }
    return canonical;
}

[[nodiscard]] inline std::string resolve_writable(
    const std::string& path,
    const std::string& allowed_prefix)
{
    const auto resolved = resolve(path, allowed_prefix);
    namespace fs = std::filesystem;
    const auto parent = fs::path(resolved).parent_path();
    if (!fs::is_directory(parent)) {
        throw std::runtime_error(
            "[safe_path] Parent directory does not exist: " + parent.string());
    }
    return resolved;
}

[[nodiscard]] inline int resolve_seed(
    const std::string& path,
    const std::string& allowed_prefix = "/etc/ml-defender/keys/")
{
    namespace fs = std::filesystem;
    const auto resolved = resolve(path, allowed_prefix);

    if (fs::is_symlink(fs::path(resolved))) {
        throw std::runtime_error(
            "[safe_path] SECURITY VIOLATION — symlink rejected for seed material: "
            + resolved);
    }

    struct stat st{};
    if (stat(resolved.c_str(), &st) != 0 || (st.st_mode & 0777) != 0400) {
        throw std::runtime_error(
            "[safe_path] SECURITY VIOLATION — seed file permissions must be 0400: "
            + resolved);
    }

    const int fd = open(resolved.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(
            "[safe_path] Cannot open seed file: " + resolved);
    }
    return fd;
}

} // namespace argus::safe_path
