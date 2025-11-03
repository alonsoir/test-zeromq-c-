// sniffer/src/userspace/ransomware_feature_extractor.cpp
// Ransomware-specific feature extraction implementation

#include "ransomware_feature_extractor.hpp"

namespace sniffer {

RansomwareFeatureExtractor::RansomwareFeatureExtractor(
    FlowTracker& flow_tracker,
    DNSAnalyzer& dns_analyzer,
    IPWhitelist& ip_whitelist,
    size_t max_events)
    : aggregator_(flow_tracker, ip_whitelist, dns_analyzer, max_events)
    , flow_tracker_(flow_tracker)
    , dns_analyzer_(dns_analyzer)
    , ip_whitelist_(ip_whitelist)
{
}

// ========== EVENT TRACKING ==========

void RansomwareFeatureExtractor::add_event(const TimeWindowEvent& event) {
    aggregator_.add_event(event);
}

size_t RansomwareFeatureExtractor::cleanup_old_events(uint64_t retention_ns) {
    return aggregator_.cleanup_old_events(retention_ns);
}

// ========== PHASE 1A: Critical Features (3) ==========

RansomwareWindowFeatures RansomwareFeatureExtractor::extract_features_phase1a(
    uint64_t window_start_ns,
    uint64_t window_end_ns)
{
    // Delegate to TimeWindowAggregator
    return aggregator_.extract_ransomware_features_phase1a(
        window_start_ns, window_end_ns);
}

// ========== FUTURE PHASES ==========

RansomwareWindowFeatures RansomwareFeatureExtractor::extract_features_phase1b(
    uint64_t window_start_ns,
    uint64_t window_end_ns)
{
    // Delegate to TimeWindowAggregator
    return aggregator_.extract_ransomware_features_phase1b(
        window_start_ns, window_end_ns);
}

RansomwareWindowFeatures RansomwareFeatureExtractor::extract_features_complete(
    uint64_t window_start_ns,
    uint64_t window_end_ns)
{
    // Delegate to TimeWindowAggregator
    return aggregator_.extract_ransomware_features_complete(
        window_start_ns, window_end_ns);
}

// ========== INDIVIDUAL FEATURE EXTRACTORS ==========

float RansomwareFeatureExtractor::extract_dns_query_entropy() const {
    return aggregator_.extract_dns_query_entropy();
}

int32_t RansomwareFeatureExtractor::extract_new_external_ips_30s(
    uint64_t window_start_ns,
    uint64_t window_end_ns) const
{
    return aggregator_.extract_new_external_ips_30s(
        window_start_ns, window_end_ns);
}

int32_t RansomwareFeatureExtractor::extract_smb_connection_diversity(
    uint64_t window_start_ns,
    uint64_t window_end_ns) const
{
    return aggregator_.extract_smb_connection_diversity(
        window_start_ns, window_end_ns);
}

// ========== UTILITY ==========

size_t RansomwareFeatureExtractor::get_event_count() const noexcept {
    return aggregator_.get_event_count();
}

void RansomwareFeatureExtractor::clear() {
    aggregator_.clear();
}

WindowStats RansomwareFeatureExtractor::get_window_stats(
    uint64_t window_start_ns,
    uint64_t window_end_ns) const
{
    return aggregator_.get_window_stats(window_start_ns, window_end_ns);
}

} // namespace sniffer
