// sniffer/src/userspace/feature_extractor.cpp
#include "feature_extractor.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace sniffer {

std::array<double, FeatureExtractor::FEATURE_COUNT>
FeatureExtractor::extract_features(const FlowStatistics& flow) const {

    std::array<double, FEATURE_COUNT> features;
    features.fill(0.0);  // Initialize all to 0

    // === ORIGINALES 23 FEATURES ===
    features[DURATION] = static_cast<double>(flow.get_duration_us());
    features[SPKTS] = static_cast<double>(flow.spkts);
    features[DPKTS] = static_cast<double>(flow.dpkts);
    features[SBYTES] = static_cast<double>(flow.sbytes);
    features[DBYTES] = static_cast<double>(flow.dbytes);
    features[SLOAD] = extract_sload(flow);
    features[SMEAN] = extract_smean(flow);
    features[DMEAN] = extract_dmean(flow);
    features[FLOW_IAT_MEAN] = extract_flow_iat_mean(flow);
    features[FLOW_IAT_STD] = extract_flow_iat_std(flow);
    features[FWD_PSH_FLAGS] = static_cast<double>(flow.fwd_psh_flags);
    features[BWD_PSH_FLAGS] = static_cast<double>(flow.bwd_psh_flags);
    features[FWD_URG_FLAGS] = static_cast<double>(flow.fwd_urg_flags);
    features[BWD_URG_FLAGS] = static_cast<double>(flow.bwd_urg_flags);
    features[PACKET_LEN_MEAN] = extract_packet_len_mean(flow);
    features[PACKET_LEN_STD] = extract_packet_len_std(flow);
    features[PACKET_LEN_VAR] = extract_packet_len_var(flow);
    features[FIN_FLAG_COUNT] = static_cast<double>(flow.fin_count);
    features[SYN_FLAG_COUNT] = static_cast<double>(flow.syn_count);
    features[RST_FLAG_COUNT] = static_cast<double>(flow.rst_count);
    features[PSH_FLAG_COUNT] = static_cast<double>(flow.psh_count);
    features[ACK_FLAG_COUNT] = static_cast<double>(flow.ack_count);
    features[URG_FLAG_COUNT] = static_cast<double>(flow.urg_count);

    // === FASE 1: NUEVAS 20 FEATURES ===
    features[DLOAD] = extract_dload(flow);
    features[RATE] = extract_rate(flow);
    features[SRATE] = extract_srate(flow);
    features[DRATE] = extract_drate(flow);
    features[SPKTS_RATIO] = extract_spkts_ratio(flow);
    features[SBYTES_RATIO] = extract_sbytes_ratio(flow);
    features[FLOW_IAT_MAX] = extract_flow_iat_max(flow);
    features[FLOW_IAT_MIN] = extract_flow_iat_min(flow);
    features[PACKET_LEN_MAX] = extract_packet_len_max(flow);
    features[PACKET_LEN_MIN] = extract_packet_len_min(flow);
    features[FWD_LEN_MAX] = extract_fwd_len_max(flow);
    features[FWD_LEN_MIN] = extract_fwd_len_min(flow);
    features[FWD_LEN_TOT] = extract_fwd_len_tot(flow);
    features[BWD_LEN_MAX] = extract_bwd_len_max(flow);
    features[BWD_LEN_MIN] = extract_bwd_len_min(flow);
    features[BWD_LEN_TOT] = extract_bwd_len_tot(flow);
    features[ECE_FLAG_COUNT] = static_cast<double>(flow.ece_count);
    features[CWR_FLAG_COUNT] = static_cast<double>(flow.cwr_count);
    features[FWD_HEADER_LEN_MEAN] = extract_fwd_header_len_mean(flow);
    features[BWD_HEADER_LEN_MEAN] = extract_bwd_header_len_mean(flow);

    return features;
}

// ============================================================================
// ORIGINAL EXTRACTORS
// ============================================================================

double FeatureExtractor::extract_duration(const FlowStatistics& flow) const {
    return static_cast<double>(flow.get_duration_us());
}

double FeatureExtractor::extract_sload(const FlowStatistics& flow) const {
    uint64_t duration_us = flow.get_duration_us();
    if (duration_us == 0) return 0.0;

    double duration_sec = duration_us / 1'000'000.0;
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

// ============================================================================
// FASE 1: NUEVOS EXTRACTORS
// ============================================================================

double FeatureExtractor::extract_dload(const FlowStatistics& flow) const {
    uint64_t duration_us = flow.get_duration_us();
    if (duration_us == 0) return 0.0;

    double duration_sec = duration_us / 1'000'000.0;
    return (flow.dbytes * 8.0) / duration_sec;  // Backward bits per second
}

double FeatureExtractor::extract_rate(const FlowStatistics& flow) const {
    uint64_t duration_us = flow.get_duration_us();
    if (duration_us == 0) return 0.0;

    double duration_sec = duration_us / 1'000'000.0;
    uint64_t total_packets = flow.spkts + flow.dpkts;
    return static_cast<double>(total_packets) / duration_sec;
}

double FeatureExtractor::extract_srate(const FlowStatistics& flow) const {
    uint64_t duration_us = flow.get_duration_us();
    if (duration_us == 0) return 0.0;

    double duration_sec = duration_us / 1'000'000.0;
    return static_cast<double>(flow.spkts) / duration_sec;
}

double FeatureExtractor::extract_drate(const FlowStatistics& flow) const {
    uint64_t duration_us = flow.get_duration_us();
    if (duration_us == 0) return 0.0;

    double duration_sec = duration_us / 1'000'000.0;
    return static_cast<double>(flow.dpkts) / duration_sec;
}

double FeatureExtractor::extract_spkts_ratio(const FlowStatistics& flow) const {
    uint64_t total_packets = flow.spkts + flow.dpkts;
    if (total_packets == 0) return 0.0;

    return static_cast<double>(flow.spkts) / static_cast<double>(total_packets);
}

double FeatureExtractor::extract_sbytes_ratio(const FlowStatistics& flow) const {
    uint64_t total_bytes = flow.sbytes + flow.dbytes;
    if (total_bytes == 0) return 0.0;

    return static_cast<double>(flow.sbytes) / static_cast<double>(total_bytes);
}

double FeatureExtractor::extract_flow_iat_max(const FlowStatistics& flow) const {
    return calculate_iat_max(flow.packet_timestamps);
}

double FeatureExtractor::extract_flow_iat_min(const FlowStatistics& flow) const {
    return calculate_iat_min(flow.packet_timestamps);
}

double FeatureExtractor::extract_packet_len_max(const FlowStatistics& flow) const {
    return calculate_max(flow.all_lengths);
}

double FeatureExtractor::extract_packet_len_min(const FlowStatistics& flow) const {
    return calculate_min(flow.all_lengths);
}

double FeatureExtractor::extract_fwd_len_max(const FlowStatistics& flow) const {
    return calculate_max(flow.fwd_lengths);
}

double FeatureExtractor::extract_fwd_len_min(const FlowStatistics& flow) const {
    return calculate_min(flow.fwd_lengths);
}

double FeatureExtractor::extract_fwd_len_tot(const FlowStatistics& flow) const {
    return calculate_sum(flow.fwd_lengths);
}

double FeatureExtractor::extract_bwd_len_max(const FlowStatistics& flow) const {
    return calculate_max(flow.bwd_lengths);
}

double FeatureExtractor::extract_bwd_len_min(const FlowStatistics& flow) const {
    return calculate_min(flow.bwd_lengths);
}

double FeatureExtractor::extract_bwd_len_tot(const FlowStatistics& flow) const {
    return calculate_sum(flow.bwd_lengths);
}

double FeatureExtractor::extract_fwd_header_len_mean(const FlowStatistics& flow) const {
    return calculate_mean_u16(flow.fwd_header_lengths);
}

double FeatureExtractor::extract_bwd_header_len_mean(const FlowStatistics& flow) const {
    return calculate_mean_u16(flow.bwd_header_lengths);
}

// ============================================================================
// STATISTICAL HELPERS
// ============================================================================

double FeatureExtractor::calculate_mean(const std::vector<uint32_t>& values) const {
    if (values.empty()) return 0.0;

    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

double FeatureExtractor::calculate_mean_u16(const std::vector<uint16_t>& values) const {
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

double FeatureExtractor::calculate_max(const std::vector<uint32_t>& values) const {
    if (values.empty()) return 0.0;

    auto max_it = std::max_element(values.begin(), values.end());
    return static_cast<double>(*max_it);
}

double FeatureExtractor::calculate_min(const std::vector<uint32_t>& values) const {
    if (values.empty()) return 0.0;

    auto min_it = std::min_element(values.begin(), values.end());
    return static_cast<double>(*min_it);
}

double FeatureExtractor::calculate_sum(const std::vector<uint32_t>& values) const {
    if (values.empty()) return 0.0;

    return static_cast<double>(std::accumulate(values.begin(), values.end(), 0ULL));
}

// ============================================================================
// IAT HELPERS
// ============================================================================

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

    double sum = std::accumulate(iats.begin(), iats.end(), 0.0);
    double mean = sum / static_cast<double>(iats.size());

    double sum_sq_diff = 0.0;
    for (uint64_t iat : iats) {
        double diff = static_cast<double>(iat) - mean;
        sum_sq_diff += diff * diff;
    }

    double variance = sum_sq_diff / static_cast<double>(iats.size());
    return std::sqrt(variance);
}

double FeatureExtractor::calculate_iat_max(const std::vector<uint64_t>& timestamps) const {
    auto iats = compute_inter_arrival_times(timestamps);
    if (iats.empty()) return 0.0;

    auto max_it = std::max_element(iats.begin(), iats.end());
    return static_cast<double>(*max_it);
}

double FeatureExtractor::calculate_iat_min(const std::vector<uint64_t>& timestamps) const {
    auto iats = compute_inter_arrival_times(timestamps);
    if (iats.empty()) return 0.0;

    auto min_it = std::min_element(iats.begin(), iats.end());
    return static_cast<double>(*min_it);
}

// ============================================================================
// FEATURE NAMES
// ============================================================================

const char* FeatureExtractor::get_feature_name(size_t index) {
    static const char* names[] = {
        // Originales 23
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
        "urg_flag_count",
        // Fase 1: Nuevas 20
        "dload",
        "rate",
        "srate",
        "drate",
        "spkts_ratio",
        "sbytes_ratio",
        "flow_iat_max",
        "flow_iat_min",
        "packet_len_max",
        "packet_len_min",
        "fwd_len_max",
        "fwd_len_min",
        "fwd_len_tot",
        "bwd_len_max",
        "bwd_len_min",
        "bwd_len_tot",
        "ece_flag_count",
        "cwr_flag_count",
        "fwd_header_len_mean",
        "bwd_header_len_mean"
    };

    if (index >= FEATURE_COUNT) {
        return "UNKNOWN";
    }

    return names[index];
}

} // namespace sniffer