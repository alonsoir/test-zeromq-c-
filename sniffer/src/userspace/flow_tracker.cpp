// sniffer/src/userspace/flow_tracker.cpp
// Flow tracking implementation for ransomware detection

#include "flow_tracker.hpp"
#include <algorithm>
#include <iostream>

namespace sniffer {

FlowTracker::FlowTracker(size_t max_flows, uint64_t timeout_seconds)
    : max_flows_(max_flows)
    , timeout_ns_(timeout_seconds * 1'000'000'000ULL)  // Convert to nanoseconds
{
    flows_.reserve(max_flows);
}

bool FlowTracker::update_flow(const PacketInfo& packet) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Create flow key
    FlowKey key{
        packet.src_ip,
        packet.dst_ip,
        packet.src_port,
        packet.dst_port,
        packet.protocol
    };
    
    // Check if flow exists
    auto it = flows_.find(key);
    bool is_new_flow = (it == flows_.end());
    
    if (is_new_flow) {
        // Check if we need to evict old flows
        if (flows_.size() >= max_flows_) {
            evict_oldest_flow();
        }
        
        // Create new flow
        FlowStats stats;
        stats.start_time_ns = packet.timestamp_ns;
        stats.last_seen_ns = packet.timestamp_ns;
        
        if (packet.is_forward) {
            stats.bytes_sent = packet.length;
            stats.packets_sent = 1;
        } else {
            stats.bytes_received = packet.length;
            stats.packets_received = 1;
        }
        
        // Update TCP flags if TCP packet
        if (packet.protocol == 6) {  // TCP
            update_tcp_flags(stats, packet.tcp_flags);
        }
        
        flows_[key] = stats;
        return false;  // New flow created
    } else {
        // Update existing flow
        FlowStats& stats = it->second;
        
        stats.last_seen_ns = packet.timestamp_ns;
        
        if (packet.is_forward) {
            stats.bytes_sent += packet.length;
            stats.packets_sent++;
        } else {
            stats.bytes_received += packet.length;
            stats.packets_received++;
        }
        
        // Update TCP flags if TCP packet
        if (packet.protocol == 6) {  // TCP
            update_tcp_flags(stats, packet.tcp_flags);
        }
        
        return true;  // Existing flow updated
    }
}

const FlowStats* FlowTracker::get_flow(const FlowKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = flows_.find(key);
    if (it == flows_.end()) {
        return nullptr;
    }
    
    return &it->second;
}

FlowStats* FlowTracker::get_flow_mut(const FlowKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = flows_.find(key);
    if (it == flows_.end()) {
        return nullptr;
    }
    
    return &it->second;
}

size_t FlowTracker::cleanup_expired_flows(uint64_t current_time_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t removed = 0;
    
    // Iterate and remove expired flows
    for (auto it = flows_.begin(); it != flows_.end();) {
        const FlowStats& stats = it->second;
        
        // Check if flow has expired
        uint64_t age_ns = current_time_ns - stats.last_seen_ns;
        if (age_ns > timeout_ns_) {
            it = flows_.erase(it);
            removed++;
        } else {
            ++it;
        }
    }
    
    return removed;
}

size_t FlowTracker::cleanup_expired_flows() {
    return cleanup_expired_flows(get_current_time_ns());
}

std::vector<FlowKey> FlowTracker::get_flows_in_window(
    uint64_t window_start_ns, 
    uint64_t window_end_ns) const 
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<FlowKey> result;
    result.reserve(flows_.size() / 10);  // Estimate ~10% of flows in window
    
    for (const auto& [key, stats] : flows_) {
        // Check if flow was active during window
        bool active_in_window = 
            (stats.start_time_ns <= window_end_ns) &&
            (stats.last_seen_ns >= window_start_ns);
        
        if (active_in_window) {
            result.push_back(key);
        }
    }
    
    return result;
}

void FlowTracker::evict_oldest_flow() {
    // Find flow with oldest last_seen time
    auto oldest = flows_.begin();
    uint64_t oldest_time = oldest->second.last_seen_ns;
    
    for (auto it = flows_.begin(); it != flows_.end(); ++it) {
        if (it->second.last_seen_ns < oldest_time) {
            oldest = it;
            oldest_time = it->second.last_seen_ns;
        }
    }
    
    flows_.erase(oldest);
}

void FlowTracker::update_tcp_flags(FlowStats& stats, uint8_t flags) {
    // Update bitmask of all flags seen
    stats.tcp_flags |= flags;
    
    // Count specific flags
    if (flags & 0x02) {  // SYN
        stats.syn_count++;
        
        // Check if this is SYN-ACK (connection established)
        if (flags & 0x10) {  // ACK
            stats.connection_established = true;
        }
    }
    
    if (flags & 0x10) {  // ACK
        stats.ack_count++;
    }
    
    if (flags & 0x04) {  // RST
        stats.rst_count++;
        stats.connection_closed = true;
    }
    
    if (flags & 0x01) {  // FIN
        stats.fin_count++;
        stats.connection_closed = true;
    }
}

} // namespace sniffer
