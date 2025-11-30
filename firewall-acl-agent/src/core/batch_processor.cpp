//===----------------------------------------------------------------------===//
// ML Defender - Firewall ACL Agent
// batch_processor.cpp - Detection Batch Processing Implementation
//
// Performance Design:
//   - std::unordered_set for O(1) deduplication
//   - Single mutex (not lock-free yet - premature optimization)
//   - Batch flush via ipset restore (single syscall)
//   - Metrics collection with atomics (no lock contention)
//
// Via Appia Quality: Measure first, optimize later
//===----------------------------------------------------------------------===//

#include "firewall/batch_processor.hpp"

#include <algorithm>
#include <sstream>

namespace mldefender::firewall {

//===----------------------------------------------------------------------===//
// Constructor / Destructor
//===----------------------------------------------------------------------===//

BatchProcessor::BatchProcessor(
    IPSetWrapper& ipset,
    const BatchProcessorConfig& config
)
    : ipset_(ipset)
    , config_(config)
    , last_flush_(std::chrono::steady_clock::now())
{
    // Reserve space for expected batch size to minimize rehashing
    pending_ips_.reserve(config_.batch_size_threshold);
}

BatchProcessor::~BatchProcessor() {
    // Flush any remaining IPs before destruction
    try {
        flush();
    } catch (...) {
        // Ignore exceptions in destructor
    }
}

//===----------------------------------------------------------------------===//
// Core API - Detection Processing
//===----------------------------------------------------------------------===//

void BatchProcessor::add_detection(const protobuf::Detection& detection) {
    // Update metrics (atomic, no lock needed)
    metrics_.detections_received.fetch_add(1, std::memory_order_relaxed);

    // Check if detection should be blocked
    if (!should_block(detection)) {
        return;
    }

    // Extract IP from detection
    std::string ip = extract_ip(detection);
    if (ip.empty()) {
        return;
    }

    // Add to batch (lock required)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Check if IP already in pending set (deduplication)
        auto [iter, inserted] = pending_ips_.insert(ip);

        if (!inserted) {
            // IP was duplicate - update metrics
            metrics_.ips_deduplicated.fetch_add(1, std::memory_order_relaxed);
        }

        // Update queue depth metrics
        size_t current_size = pending_ips_.size();
        metrics_.pending_ips_current.store(current_size, std::memory_order_relaxed);

        // Update max queue depth
        size_t current_max = metrics_.pending_ips_max.load(std::memory_order_relaxed);
        if (current_size > current_max) {
            metrics_.pending_ips_max.store(current_size, std::memory_order_relaxed);
        }

        // Check for backpressure
        check_backpressure();
    }

    // Check if flush needed (after releasing lock)
    if (should_flush()) {
        flush();
    }
}

void BatchProcessor::add_detections(
    const std::vector<protobuf::Detection>& detections
) {
    // Batch processing - minimize lock acquisitions

    // First pass: filter and extract IPs (no lock)
    std::vector<std::string> ips_to_add;
    ips_to_add.reserve(detections.size());

    for (const auto& detection : detections) {
        metrics_.detections_received.fetch_add(1, std::memory_order_relaxed);

        if (!should_block(detection)) {
            continue;
        }

        std::string ip = extract_ip(detection);
        if (!ip.empty()) {
            ips_to_add.push_back(std::move(ip));
        }
    }

    if (ips_to_add.empty()) {
        return;
    }

    // Second pass: add to pending set (single lock)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        size_t dedup_count = 0;
        for (auto& ip : ips_to_add) {
            auto [iter, inserted] = pending_ips_.insert(std::move(ip));
            if (!inserted) {
                ++dedup_count;
            }
        }

        metrics_.ips_deduplicated.fetch_add(dedup_count, std::memory_order_relaxed);

        // Update metrics
        size_t current_size = pending_ips_.size();
        metrics_.pending_ips_current.store(current_size, std::memory_order_relaxed);

        size_t current_max = metrics_.pending_ips_max.load(std::memory_order_relaxed);
        if (current_size > current_max) {
            metrics_.pending_ips_max.store(current_size, std::memory_order_relaxed);
        }

        check_backpressure();
    }

    // Check if flush needed
    if (should_flush()) {
        flush();
    }
}

void BatchProcessor::add_ip(const std::string& ip, float confidence) {
    // Simple bypass for raw IP addition
    if (confidence < config_.confidence_threshold && !config_.block_low_confidence) {
        return;
    }

    metrics_.detections_received.fetch_add(1, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto [iter, inserted] = pending_ips_.insert(ip);
        if (!inserted) {
            metrics_.ips_deduplicated.fetch_add(1, std::memory_order_relaxed);
        }

        size_t current_size = pending_ips_.size();
        metrics_.pending_ips_current.store(current_size, std::memory_order_relaxed);

        size_t current_max = metrics_.pending_ips_max.load(std::memory_order_relaxed);
        if (current_size > current_max) {
            metrics_.pending_ips_max.store(current_size, std::memory_order_relaxed);
        }

        check_backpressure();
    }

    if (should_flush()) {
        flush();
    }
}

//===----------------------------------------------------------------------===//
// Flushing Logic
//===----------------------------------------------------------------------===//

size_t BatchProcessor::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    return flush_internal();
}

size_t BatchProcessor::flush_internal() {
    // Assumes lock is already held

    if (pending_ips_.empty()) {
        return 0;
    }

    auto flush_start = std::chrono::steady_clock::now();

    // Convert to vector of IPSetEntry
    std::vector<IPSetEntry> entries;
    entries.reserve(pending_ips_.size());

    for (const auto& ip : pending_ips_) {
        entries.push_back(IPSetEntry{ip});
    }

    size_t batch_size = entries.size();

    // Flush to ipset (releases lock during syscall? No - we hold it)
    // This is OK because flush is fast (10-20ms for 1K IPs)
    auto result = ipset_.add_batch(config_.blacklist_ipset, entries);

    auto flush_end = std::chrono::steady_clock::now();
    auto flush_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        flush_end - flush_start
    );

    // Update metrics
    if (result) {
        metrics_.ips_blocked.fetch_add(batch_size, std::memory_order_relaxed);
        metrics_.batches_flushed.fetch_add(1, std::memory_order_relaxed);
        metrics_.last_batch_size.store(batch_size, std::memory_order_relaxed);
        metrics_.total_batch_size.fetch_add(batch_size, std::memory_order_relaxed);

        // Clear pending set
        pending_ips_.clear();

        // Update last flush time
        last_flush_ = flush_end;
    } else {
        // Flush failed - keep IPs in pending set
        metrics_.flush_errors.fetch_add(1, std::memory_order_relaxed);
        metrics_.ipset_errors.fetch_add(1, std::memory_order_relaxed);

        // TODO: Log error
        // For now, we keep the IPs and will retry on next flush
    }

    // Update latency metrics
    metrics_.last_flush_latency_us.store(
        flush_duration.count(),
        std::memory_order_relaxed
    );
    metrics_.total_flush_time_us.fetch_add(
        flush_duration.count(),
        std::memory_order_relaxed
    );

    // Update current queue size
    metrics_.pending_ips_current.store(
        pending_ips_.size(),
        std::memory_order_relaxed
    );

    // Trigger callback if set
    if (flush_callback_ && result) {
        try {
            flush_callback_(batch_size);
        } catch (...) {
            // Ignore callback exceptions
        }
    }

    return result ? batch_size : 0;
}

bool BatchProcessor::should_flush() const {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check size threshold
    if (pending_ips_.size() >= config_.batch_size_threshold) {
        return true;
    }

    // Check time threshold
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_flush_
    );

    if (elapsed >= config_.batch_time_threshold && !pending_ips_.empty()) {
        return true;
    }

    return false;
}

//===----------------------------------------------------------------------===//
// Configuration Updates
//===----------------------------------------------------------------------===//

void BatchProcessor::set_batch_size_threshold(size_t threshold) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.batch_size_threshold = threshold;
}

void BatchProcessor::set_batch_time_threshold(std::chrono::milliseconds threshold) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.batch_time_threshold = threshold;
}

void BatchProcessor::set_confidence_threshold(float threshold) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.confidence_threshold = threshold;
}

//===----------------------------------------------------------------------===//
// Metrics and Monitoring
//===----------------------------------------------------------------------===//

size_t BatchProcessor::get_pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_ips_.size();
}

std::chrono::milliseconds BatchProcessor::get_time_since_last_flush() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - last_flush_);
}

bool BatchProcessor::is_queue_backing_up() const {
    size_t current = metrics_.pending_ips_current.load(std::memory_order_relaxed);
    return current > (config_.max_pending_ips * 80 / 100);  // >80% of max
}

//===----------------------------------------------------------------------===//
// Callbacks
//===----------------------------------------------------------------------===//

void BatchProcessor::set_flush_callback(std::function<void(size_t)> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    flush_callback_ = std::move(callback);
}

void BatchProcessor::set_backpressure_callback(std::function<void(size_t)> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    backpressure_callback_ = std::move(callback);
}

//===----------------------------------------------------------------------===//
// Internal Helper Methods
//===----------------------------------------------------------------------===//

bool BatchProcessor::should_block(const protobuf::Detection& detection) const {
    // Check detection type
    if (detection.type() == protobuf::DetectionType::DETECTION_UNKNOWN) {
        return false;
    }

    // Extract confidence score
    float confidence = detection.confidence();

    // Check confidence threshold
    if (confidence < config_.confidence_threshold) {
        if (!config_.block_low_confidence) {
            return false;
        }
    }

    return true;
}

std::string BatchProcessor::extract_ip(const protobuf::Detection& detection) const {
    // Extract source IP from detection
    if (!detection.src_ip().empty()) {
        return detection.src_ip();
    }

    return "";
}

void BatchProcessor::check_backpressure() {
    // Assumes lock is already held

    if (backpressure_callback_ && is_queue_backing_up()) {
        size_t queue_size = pending_ips_.size();
        try {
            backpressure_callback_(queue_size);
        } catch (...) {
            // Ignore callback exceptions
        }
    }
}

} // namespace mldefender::firewall