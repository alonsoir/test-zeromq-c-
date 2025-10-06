// sniffer/src/userspace/feature_extractor.cpp
#include "feature_extractor.hpp"
#include <numeric>
#include <cmath>
#include <iostream>

namespace sniffer {

std::array<double, FeatureExtractor::FEATURE_COUNT>
FeatureExtractor::extract_features(const FlowStatistics& flow) const {

    std::array<double, FEATURE_COUNT> features;
    features.fill(0.0);  // Initialize all to 0

    // Get duration in microseconds
    uint64_t duration_us = flow.get_duration_us();
    double duration_sec = duration_us / 1'000'000.0;

    // 1. duration (in microseconds to match Python training)
    features[DURATION] = static_cast<double>(duration_us);

    // 2. spkts (forward packets)
    features[SPKTS] = static_cast<double>(flow.spkts);

    // 3. dpkts (backward packets)
    features[DPKTS] = static_cast<double>(flow.dpkts);

    // 4. sbytes (forward bytes)
    features[SBYTES] = static_cast<double>(flow.sbytes);

    // 5. dbytes (backward bytes)
    features[DBYTES] = static_cast<double>(flow.dbytes);

    // 6. sload (forward bits per second)
    features[SLOAD] = extract_sload(flow);

    // 7. smean (forward packet length mean)
    features[SMEAN] = extract_smean(flow);

    // 8. dmean (backward packet length mean)
    features[DMEAN] = extract_dmean(flow);

    // 9. flow_iat_mean (inter-arrival time mean)
    features[FLOW_IAT_MEAN] = extract_flow_iat_mean(flow);

    // 10. flow_iat_std (inter-arrival time std dev)
    features[FLOW_IAT_STD] = extract_flow_iat_std(flow);

    // 11-14. Directional TCP flags
    features[FWD_PSH_FLAGS] = static_cast<double>(flow.fwd_psh_flags);
    features[BWD_PSH_FLAGS] = static_cast<double>(flow.bwd_psh_flags);
    features[FWD_URG_FLAGS] = static_cast<double>(flow.fwd_urg_flags);
    features[BWD_URG_FLAGS] = static_cast<double>(flow.bwd_urg_flags);

    // 15-17. Packet length statistics (all packets)
    features[PACKET_LEN_MEAN] = extract_packet_len_mean(flow);
    features[PACKET_LEN_STD] = extract_packet_len_std(flow);
    features[PACKET_LEN_VAR] = extract_packet_len_var(flow);

    // 18-23. TCP flag counters
    features[FIN_FLAG_COUNT] = static_cast<double>(flow.fin_count);
    features[SYN_FLAG_COUNT] = static_cast<double>(flow.syn_count);
    features[RST_FLAG_COUNT] = static_cast<double>(flow.rst_count);
    features[PSH_FLAG_COUNT] = static_cast<double>(flow.psh_count);
    features[ACK_FLAG_COUNT] = static_cast<double>(flow.ack_count);
    features[URG_FLAG_COUNT] = static_cast<double>(flow.urg_count);

    return features;
}

// Individual feature extractors

double FeatureExtractor::extract_duration(const FlowStatistics& flow) const {
    return static_cast<double>(flow.get_duration_us());
}

double FeatureExtractor::extract_sload(const FlowStatistics& flow) const {
    uint64_t duration_us = flow.get_duration_us();
    if (duration_us == 0) return 0.0;

    double duration_sec = duration_us / 1'000'000.0;

    // Forward bits per second
    return (flow.sbytes * 8.0) / duration_sec;
}

double FeatureExtractor::extract_smean(const FlowStatistics& flow) const {
    return calculate_mean(flow.fwd_lengths);
}

double FeatureExtractor::extract_dmean(const FlowStatistics& flow) const {
    return calculate_mean(flow.bwd_lengths);
}

double FeatureExtractor::extract_flow_iat_mean(const FlowStatistics& flow) const {
    return calculate_iat_mean(flow.packet_timestamps);
}

double FeatureExtractor::extract_flow_iat_std(const FlowStatistics& flow) const {
    return calculate_iat_std(flow.packet_timestamps);
}

double FeatureExtractor::extract_packet_len_mean(const FlowStatistics& flow) const {
    return calculate_mean(flow.all_lengths);
}

double FeatureExtractor::extract_packet_len_std(const FlowStatistics& flow) const {
    return calculate_std(flow.all_lengths);
}

double FeatureExtractor::extract_packet_len_var(const FlowStatistics& flow) const {
    return calculate_variance(flow.all_lengths);
}

// Statistical helpers

double FeatureExtractor::calculate_mean(const std::vector<uint32_t>& values) const {
    if (values.empty()) return 0.0;

    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

double FeatureExtractor::calculate_std(const std::vector<uint32_t>& values) const {
    if (values.size() < 2) return 0.0;

    double mean = calculate_mean(values);

    double sum_sq_diff = 0.0;
    for (uint32_t val : values) {
        double diff = static_cast<double>(val) - mean;
        sum_sq_diff += diff * diff;
    }

    double variance = sum_sq_diff / static_cast<double>(values.size());
    return std::sqrt(variance);
}

double FeatureExtractor::calculate_variance(const std::vector<uint32_t>& values) const {
    if (values.size() < 2) return 0.0;

    double mean = calculate_mean(values);

    double sum_sq_diff = 0.0;
    for (uint32_t val : values) {
        double diff = static_cast<double>(val) - mean;
        sum_sq_diff += diff * diff;
    }

    return sum_sq_diff / static_cast<double>(values.size());
}

// Inter-arrival time helpers

std::vector<uint64_t> FeatureExtractor::compute_inter_arrival_times(
    const std::vector<uint64_t>& timestamps) const {

    std::vector<uint64_t> iats;
    if (timestamps.size() < 2) return iats;

    iats.reserve(timestamps.size() - 1);

    for (size_t i = 1; i < timestamps.size(); ++i) {
        if (timestamps[i] >= timestamps[i-1]) {
            uint64_t iat = timestamps[i] - timestamps[i-1];
            iats.push_back(iat);
        }
    }

    return iats;
}

double FeatureExtractor::calculate_iat_mean(const std::vector<uint64_t>& timestamps) const {
    auto iats = compute_inter_arrival_times(timestamps);
    if (iats.empty()) return 0.0;

    double sum = std::accumulate(iats.begin(), iats.end(), 0.0);
    return sum / static_cast<double>(iats.size());
}

double FeatureExtractor::calculate_iat_std(const std::vector<uint64_t>& timestamps) const {
    auto iats = compute_inter_arrival_times(timestamps);
    if (iats.size() < 2) return 0.0;

    // Calculate mean
    double sum = std::accumulate(iats.begin(), iats.end(), 0.0);
    double mean = sum / static_cast<double>(iats.size());

    // Calculate std dev
    double sum_sq_diff = 0.0;
    for (uint64_t iat : iats) {
        double diff = static_cast<double>(iat) - mean;
        sum_sq_diff += diff * diff;
    }

    double variance = sum_sq_diff / static_cast<double>(iats.size());
    return std::sqrt(variance);
}

// Feature name lookup

const char* FeatureExtractor::get_feature_name(size_t index) {
    static const char* names[] = {
        "duration",
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "sload",
        "smean",
        "dmean",
        "flow_iat_mean",
        "flow_iat_std",
        "fwd_psh_flags",
        "bwd_psh_flags",
        "fwd_urg_flags",
        "bwd_urg_flags",
        "packet_len_mean",
        "packet_len_std",
        "packet_len_var",
        "fin_flag_count",
        "syn_flag_count",
        "rst_flag_count",
        "psh_flag_count",
        "ack_flag_count",
        "urg_flag_count"
    };

    if (index >= FEATURE_COUNT) {
        return "UNKNOWN";
    }

    return names[index];
}

} // namespace sniffer