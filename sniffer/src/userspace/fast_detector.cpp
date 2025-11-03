// sniffer/src/userspace/fast_detector.cpp
#include "fast_detector.hpp"
#include "protocol_numbers.hpp"
#include "time_window_aggregator.hpp"  // For is_external_ip()
#include <cstring>

namespace sniffer {

FastDetector::FastDetector()
    : window_start_ns_(0),
      smb_conn_count_(0),
      rst_count_(0),
      total_tcp_(0) {
    // Pre-reserve to avoid rehashing during hot path
    recent_external_ips_.reserve(32);
    recent_ports_.reserve(32);
}

void FastDetector::ingest(const SimpleEvent& evt) {
    // Initialize window on first event
    if (window_start_ns_ == 0) {
        window_start_ns_ = evt.timestamp;
    }

    // Check if window expired
    uint64_t elapsed = evt.timestamp - window_start_ns_;
    if (elapsed > WINDOW_NS) {
        // Window expired - reset all counters
        recent_external_ips_.clear();
        recent_ports_.clear();
        smb_conn_count_ = 0;
        rst_count_ = 0;
        total_tcp_ = 0;
        window_start_ns_ = evt.timestamp;
    }

    // === Heuristic Analysis ===

    // 1. External IPs (C&C scanning behavior)
    if (is_external_ip(evt.dst_ip)) {
        if (is_new_external_ip(evt.dst_ip)) {
            recent_external_ips_.insert(evt.dst_ip);
        }
    }

    // 2. SMB connections (lateral movement)
    if (is_smb_connection(evt.dst_port)) {
        smb_conn_count_++;
    }

    // 3. TCP RST tracking (failed connections)
    if (evt.protocol == 6) {  // TCP
        total_tcp_++;
        if (evt.tcp_flags & 0x04) {  // RST flag
            rst_count_++;
        }
    }

    // 4. Port diversity (port scanning)
    if (evt.dst_port > 0) {
        recent_ports_.insert(evt.dst_port);
    }
}

bool FastDetector::is_external_ip(uint32_t ip) const noexcept {
    // Use existing TimeWindowAggregator logic
    return TimeWindowAggregator::is_external_ip(ip);
}

bool FastDetector::is_new_external_ip(uint32_t ip) {
    return recent_external_ips_.find(ip) == recent_external_ips_.end();
}

bool FastDetector::is_smb_connection(uint16_t port) const noexcept {
    return (port == 445 || port == 139);
}

bool FastDetector::detect_rapid_port_scan() const noexcept {
    return recent_ports_.size() > THRESHOLD_PORT_SCAN;
}

double FastDetector::rst_ratio() const noexcept {
    return total_tcp_ > 0 
        ? static_cast<double>(rst_count_) / static_cast<double>(total_tcp_)
        : 0.0;
}

bool FastDetector::is_suspicious() const {
    // Heuristic 1: Too many external IPs (C&C scanning)
    if (recent_external_ips_.size() > THRESHOLD_EXTERNAL_IPS) {
        return true;
    }

    // Heuristic 2: Multiple SMB connections (lateral movement)
    if (smb_conn_count_ > THRESHOLD_SMB_CONNS) {
        return true;
    }

    // Heuristic 3: Rapid port scanning (reconnaissance)
    if (detect_rapid_port_scan()) {
        return true;
    }

    // Heuristic 4: High RST ratio (connection failures)
    if (rst_ratio() > THRESHOLD_RST_RATIO) {
        return true;
    }

    return false;
}

void FastDetector::reset_if_stale(uint64_t now_ns) {
    if (now_ns - window_start_ns_ > WINDOW_NS) {
        recent_external_ips_.clear();
        recent_ports_.clear();
        smb_conn_count_ = 0;
        rst_count_ = 0;
        total_tcp_ = 0;
        window_start_ns_ = now_ns;
    }
}

FastDetector::Snapshot FastDetector::snapshot() const {
    Snapshot s;
    s.external_ips_10s = recent_external_ips_.size();
    s.smb_conns = smb_conn_count_;
    s.resets = rst_count_;
    s.total_tcp = total_tcp_;
    s.unique_ports = recent_ports_.size();
    s.window_start_ns = window_start_ns_;
    return s;
}

} // namespace sniffer
