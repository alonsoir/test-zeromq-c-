// sniffer/include/flow/sharded_flow_manager.hpp
// Day 43 - ShardedFlowManager Implementation (unique_ptr version)

#pragma once

#include "flow_tracker.hpp"  // For FlowKey
#include "flow_manager.hpp"   // For FlowStatistics
#include <mutex>

// Forward declaration
struct SimpleEvent;

#include <unordered_map>
#include <shared_mutex>
#include <atomic>
#include <vector>
#include <list>
#include <chrono>
#include <thread>
#include <memory>

namespace sniffer {
namespace flow {

class ShardedFlowManager {
public:
    struct Config {
        size_t max_flows_per_shard = 10000;
        uint64_t flow_timeout_ns = 120000000000ULL;
        bool enable_statistics = true;
        bool auto_export_on_tcp_close = false;
        size_t shard_count = 0;
    };

    struct ShardStats {
        std::atomic<uint64_t> flows_created{0};
        std::atomic<uint64_t> flows_expired{0};
        std::atomic<uint64_t> packets_processed{0};
        std::atomic<uint64_t> current_flows{0};
        std::atomic<uint64_t> lock_contentions{0};
        std::atomic<uint64_t> cleanup_skipped{0};
    };

    struct GlobalStats {
        uint64_t total_flows_created = 0;
        uint64_t total_flows_expired = 0;
        uint64_t total_packets_processed = 0;
        uint64_t total_active_flows = 0;
        uint64_t total_lock_contentions = 0;
        uint64_t total_cleanup_skipped = 0;
        size_t shard_count = 0;
    };

    static ShardedFlowManager& instance();
    void initialize(const Config& config);
    void add_packet(const FlowKey& key, const SimpleEvent& event);
    const FlowStatistics* get_flow_stats(const FlowKey& key) const;
    FlowStatistics* get_flow_stats_mut(const FlowKey& key);
    size_t cleanup_expired(std::chrono::seconds ttl);
    size_t cleanup_all();
    GlobalStats get_stats() const;
    void print_stats() const;
    void reset_stats();
    size_t get_shard_count() const { return shards_.size(); }

private:
    struct Shard {
        std::unique_ptr<std::unordered_map<FlowKey, FlowStatistics, FlowKey::Hash>> flows;
        std::unique_ptr<std::list<FlowKey>> lru_queue;
        std::unique_ptr<std::shared_mutex> mtx;
        std::atomic<uint64_t> last_seen_ns{0};
        ShardStats stats;

        Shard()
            : flows(std::make_unique<std::unordered_map<FlowKey, FlowStatistics, FlowKey::Hash>>()),
              lru_queue(std::make_unique<std::list<FlowKey>>()),
              mtx(std::make_unique<std::shared_mutex>()),
              last_seen_ns(0) {}
    };

    ShardedFlowManager() = default;
    ShardedFlowManager(const ShardedFlowManager&) = delete;
    ShardedFlowManager& operator=(const ShardedFlowManager&) = delete;

    size_t get_shard_id(const FlowKey& key) const {
        return FlowKey::Hash{}(key) % shards_.size();
    }

    size_t cleanup_shard_partial(Shard& shard, size_t max_remove = 100);

    static uint64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    }

    Config config_;
    bool initialized_{false};
    std::vector<std::unique_ptr<Shard>> shards_;
};

} // namespace flow
} // namespace sniffer