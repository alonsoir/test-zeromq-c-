// sniffer/include/time_window_aggregator.hpp
// Time-windowed feature aggregation for ransomware detection
// Aggregates features from FlowTracker, IPWhitelist, DNSAnalyzer

#pragma once

#include <cstdint>
#include <vector>
#include <deque>
#include <mutex>
#include <memory>
#include <chrono>

// Forward declarations
namespace sniffer {
class FlowTracker;
class IPWhitelist;
class DNSAnalyzer;
struct FlowKey;
}

namespace sniffer {

/**
 * Event tracked in time window
 * Generic event structure for temporal aggregation
 */
struct TimeWindowEvent {
    uint64_t timestamp_ns;      // Event timestamp
    uint32_t src_ip;            // Source IP
    uint32_t dst_ip;            // Destination IP
    uint16_t src_port;          // Source port
    uint16_t dst_port;          // Destination port
    uint8_t protocol;           // Protocol (6=TCP, 17=UDP)
    uint32_t bytes;             // Bytes in event

    TimeWindowEvent(uint64_t ts, uint32_t sip, uint32_t dip,
                   uint16_t sp, uint16_t dp, uint8_t proto, uint32_t b)
        : timestamp_ns(ts), src_ip(sip), dst_ip(dip)
        , src_port(sp), dst_port(dp), protocol(proto), bytes(b) {}
};

/**
 * Window statistics for a time period
 */
struct WindowStats {
    uint64_t window_start_ns;
    uint64_t window_end_ns;
    size_t event_count;
    size_t unique_ips_count;
    size_t unique_ports_count;
    uint64_t total_bytes;

    WindowStats()
        : window_start_ns(0), window_end_ns(0)
        , event_count(0), unique_ips_count(0)
        , unique_ports_count(0), total_bytes(0) {}
};

/**
 * Aggregated ransomware features (Phase 1A: 3 features)
 * Complete structure for all 20 features (to be expanded)
 */
struct RansomwareWindowFeatures {
    // === PHASE 1A: Critical Features (3) ===
    float dns_query_entropy;                // Feature 1
    int32_t new_external_ips_30s;          // Feature 2
    int32_t smb_connection_diversity;      // Feature 3

    // === PHASE 1B: High Priority Features (7) ===
    float dns_query_rate_per_min;          // Feature 3 (original)
    float failed_dns_queries_ratio;        // Feature 4
    int32_t new_internal_connections_30s;  // Feature 9
    float upload_download_ratio_30s;       // Feature 11
    int32_t unique_destinations_30s;       // Feature 13
    float protocol_diversity_score;        // Feature 17

    // === PHASE 1C: Medium Priority Features (10) ===
    int32_t tls_self_signed_cert_count;    // Feature 5
    int32_t non_standard_port_http_count;  // Feature 6
    int32_t rdp_failed_auth_count;         // Feature 8
    float port_scan_pattern_score;         // Feature 10
    int32_t burst_connections_count;       // Feature 12
    int32_t large_upload_sessions_count;   // Feature 14
    bool nocturnal_activity_flag;          // Feature 15
    float connection_rate_stddev;          // Feature 16
    float avg_flow_duration_seconds;       // Feature 18
    float tcp_rst_ratio;                   // Feature 19
    float syn_without_ack_ratio;           // Feature 20

    // Metadata
    uint64_t window_start_ns;
    uint64_t window_end_ns;
    bool features_valid;

    RansomwareWindowFeatures()
        : dns_query_entropy(0.0f)
        , new_external_ips_30s(0)
        , smb_connection_diversity(0)
        , dns_query_rate_per_min(0.0f)
        , failed_dns_queries_ratio(0.0f)
        , new_internal_connections_30s(0)
        , upload_download_ratio_30s(0.0f)
        , unique_destinations_30s(0)
        , protocol_diversity_score(0.0f)
        , tls_self_signed_cert_count(0)
        , non_standard_port_http_count(0)
        , rdp_failed_auth_count(0)
        , port_scan_pattern_score(0.0f)
        , burst_connections_count(0)
        , large_upload_sessions_count(0)
        , nocturnal_activity_flag(false)
        , connection_rate_stddev(0.0f)
        , avg_flow_duration_seconds(0.0f)
        , tcp_rst_ratio(0.0f)
        , syn_without_ack_ratio(0.0f)
        , window_start_ns(0)
        , window_end_ns(0)
        , features_valid(false) {}
};

/**
 * TimeWindowAggregator: Aggregates features across time windows
 *
 * Collects features from multiple sources (FlowTracker, IPWhitelist, DNSAnalyzer)
 * and aggregates them into time windows for ransomware detection.
 *
 * Key capabilities:
 * - Track events in configurable time windows (30s, 60s, 300s, etc.)
 * - Aggregate features from multiple sources
 * - Thread-safe operation
 * - Memory-efficient ring buffer
 * - Support for sliding windows
 *
 * Usage:
 *   TimeWindowAggregator agg(flow_tracker, ip_whitelist, dns_analyzer);
 *   agg.add_event(event);
 *   auto features = agg.extract_ransomware_features_phase1a();
 */
class TimeWindowAggregator {
public:
    /**
     * Constructor
     * @param flow_tracker Reference to flow tracker
     * @param ip_whitelist Reference to IP whitelist
     * @param dns_analyzer Reference to DNS analyzer
     * @param max_events Maximum events in ring buffer
     */
    TimeWindowAggregator(FlowTracker& flow_tracker,
                        IPWhitelist& ip_whitelist,
                        DNSAnalyzer& dns_analyzer,
                        size_t max_events = 10000);

    /**
     * Add event to time window tracker
     * @param event Event to add
     */
    void add_event(const TimeWindowEvent& event);

    /**
     * Get window statistics for time period
     * @param window_start_ns Window start (nanoseconds)
     * @param window_end_ns Window end (nanoseconds)
     * @return Window statistics
     */
    WindowStats get_window_stats(uint64_t window_start_ns,
                                 uint64_t window_end_ns) const;

    /**
     * Count events in time window
     * @param window_start_ns Window start (nanoseconds)
     * @param window_end_ns Window end (nanoseconds)
     * @return Number of events in window
     */
    size_t count_events_in_window(uint64_t window_start_ns,
                                  uint64_t window_end_ns) const;

    // ========== PHASE 1A: Extract Critical Features (3) ==========

    /**
     * Extract Phase 1A ransomware features (3 critical features)
     * - dns_query_entropy
     * - new_external_ips_30s
     * - smb_connection_diversity
     *
     * @param window_start_ns Optional window start (default: 30s ago)
     * @param window_end_ns Optional window end (default: now)
     * @return Ransomware features (only Phase 1A populated)
     */
    RansomwareWindowFeatures extract_ransomware_features_phase1a(
        uint64_t window_start_ns = 0,
        uint64_t window_end_ns = 0);

    /**
     * Extract DNS query entropy (Feature 1)
     * Uses DNSAnalyzer to calculate Shannon entropy
     * @return Entropy value (0.0-8.0)
     */
    float extract_dns_query_entropy() const;

    /**
     * Extract new external IPs in 30s window (Feature 2)
     * Uses IPWhitelist to count new IPs
     * @param window_start_ns Window start (default: 30s ago)
     * @param window_end_ns Window end (default: now)
     * @return Count of new external IPs
     */
    int32_t extract_new_external_ips_30s(
        uint64_t window_start_ns = 0,
        uint64_t window_end_ns = 0) const;

    /**
     * Extract SMB connection diversity (Feature 7)
     * Count unique destination IPs contacted via SMB (port 445)
     * @param window_start_ns Window start (default: 60s ago)
     * @param window_end_ns Window end (default: now)
     * @return Count of unique SMB destinations
     */
    int32_t extract_smb_connection_diversity(
        uint64_t window_start_ns = 0,
        uint64_t window_end_ns = 0) const;

    // ========== FUTURE PHASES: Additional Features ==========

    /**
     * Extract Phase 1B features (7 high priority features)
     * To be implemented after Phase 1A validation
     */
    RansomwareWindowFeatures extract_ransomware_features_phase1b(
        uint64_t window_start_ns = 0,
        uint64_t window_end_ns = 0);

    /**
     * Extract all 20 ransomware features (complete)
     * To be implemented after Phase 1A + 1B validation
     */
    RansomwareWindowFeatures extract_ransomware_features_complete(
        uint64_t window_start_ns = 0,
        uint64_t window_end_ns = 0);

    // ========== Utility Methods ==========

    /**
     * Cleanup old events outside retention window
     * @param retention_ns Retention time in nanoseconds
     * @return Number of events removed
     */
    size_t cleanup_old_events(uint64_t retention_ns);

    /**
     * Get event count in buffer
     */
    size_t get_event_count() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_.size();
    }

    /**
     * Clear all events (for testing)
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.clear();
    }

    /**
     * Get current timestamp in nanoseconds
     */
    static uint64_t get_current_time_ns() noexcept;

    /**
     * Check if IP is external (not in private ranges)
     * @param ip IP address (host byte order)
     * @return true if external, false if internal
     */
    static bool is_external_ip(uint32_t ip) noexcept;

private:
    // References to external components
    FlowTracker& flow_tracker_;
    IPWhitelist& ip_whitelist_;
    DNSAnalyzer& dns_analyzer_;

    // Event ring buffer (most recent events)
    std::deque<TimeWindowEvent> events_;

    // Thread safety
    mutable std::mutex mutex_;

    // Configuration
    size_t max_events_;

    // ========== Helper Methods ==========

    /**
     * Get events in time window
     * @param window_start_ns Window start
     * @param window_end_ns Window end
     * @return Vector of events in window
     */
    std::vector<TimeWindowEvent> get_events_in_window(
        uint64_t window_start_ns,
        uint64_t window_end_ns) const;

    /**
     * Count unique IPs in event list
     * @param events Events to analyze
     * @param check_destination true=count dst_ip, false=count src_ip
     * @return Count of unique IPs
     */
    static size_t count_unique_ips(
        const std::vector<TimeWindowEvent>& events,
        bool check_destination = true);

    /**
     * Count unique ports in event list
     * @param events Events to analyze
     * @param check_destination true=count dst_port, false=count src_port
     * @return Count of unique ports
     */
    static size_t count_unique_ports(
        const std::vector<TimeWindowEvent>& events,
        bool check_destination = true);

    /**
     * Get default window start (30s ago)
     */
    static uint64_t get_default_window_start_30s() noexcept;

    /**
     * Get default window start (60s ago)
     */
    static uint64_t get_default_window_start_60s() noexcept;
};

} // namespace sniffer