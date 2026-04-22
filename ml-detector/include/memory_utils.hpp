#pragma once
// memory_utils.hpp — Pure memory calculation helpers (ADR-037 F17)
// Header-only, zero external dependencies — testable without zmq/fmt/onnx.
#include <cstdint>

namespace ml_detector {

// Use double arithmetic directly — avoids int64_t overflow for extreme values.
// int64_t cast is insufficient: LONG_MAX/4096 * 8192 still overflows int64_t.
// double has 53-bit mantissa, sufficient for any realistic memory value.
[[nodiscard]] inline double compute_memory_mb(long pages, long page_size) noexcept {
    return (static_cast<double>(pages) * static_cast<double>(page_size)) / (1024.0 * 1024.0);
}

} // namespace ml_detector
