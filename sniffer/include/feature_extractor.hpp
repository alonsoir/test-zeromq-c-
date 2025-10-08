// sniffer/include/feature_extractor.hpp
#pragma once

#include "flow_manager.hpp"
#include <array>
#include <vector>

namespace sniffer {

class FeatureExtractor {
public:
    // ⭐ ACTUALIZADO: 23 -> 43 features (Fase 1)
    static constexpr size_t FEATURE_COUNT = 43;

    // Feature indices for the array
    enum FeatureIndex {
        // === BÁSICAS (23 originales) ===
        DURATION = 0,
        SPKTS,
        DPKTS,
        SBYTES,
        DBYTES,
        SLOAD,
        SMEAN,
        DMEAN,
        FLOW_IAT_MEAN,
        FLOW_IAT_STD,
        FWD_PSH_FLAGS,
        BWD_PSH_FLAGS,
        FWD_URG_FLAGS,
        BWD_URG_FLAGS,
        PACKET_LEN_MEAN,
        PACKET_LEN_STD,
        PACKET_LEN_VAR,
        FIN_FLAG_COUNT,
        SYN_FLAG_COUNT,
        RST_FLAG_COUNT,
        PSH_FLAG_COUNT,
        ACK_FLAG_COUNT,
        URG_FLAG_COUNT,

        // === FASE 1: NUEVAS 20 FEATURES ===
        // Rates
        DLOAD,              // 23: backward bits/sec
        RATE,               // 24: total packets/sec
        SRATE,              // 25: forward packets/sec
        DRATE,              // 26: backward packets/sec

        // Ratios
        SPKTS_RATIO,        // 27: spkts / (spkts + dpkts)
        SBYTES_RATIO,       // 28: sbytes / (sbytes + dbytes)

        // Flow IAT extended
        FLOW_IAT_MAX,       // 29
        FLOW_IAT_MIN,       // 30

        // Packet length extended
        PACKET_LEN_MAX,     // 31
        PACKET_LEN_MIN,     // 32

        // Forward length stats
        FWD_LEN_MAX,        // 33
        FWD_LEN_MIN,        // 34
        FWD_LEN_TOT,        // 35

        // Backward length stats
        BWD_LEN_MAX,        // 36
        BWD_LEN_MIN,        // 37
        BWD_LEN_TOT,        // 38

        // TCP flags extended
        ECE_FLAG_COUNT,     // 39
        CWR_FLAG_COUNT,     // 40

        // Header lengths
        FWD_HEADER_LEN_MEAN, // 41
        BWD_HEADER_LEN_MEAN  // 42
    };

    FeatureExtractor() = default;

    // Extract all features from a flow
    std::array<double, FEATURE_COUNT> extract_features(const FlowStatistics& flow) const;

    // Get feature name by index
    static const char* get_feature_name(size_t index);

private:
    // === ORIGINAL EXTRACTORS ===
    double extract_duration(const FlowStatistics& flow) const;
    double extract_sload(const FlowStatistics& flow) const;
    double extract_smean(const FlowStatistics& flow) const;
    double extract_dmean(const FlowStatistics& flow) const;
    double extract_flow_iat_mean(const FlowStatistics& flow) const;
    double extract_flow_iat_std(const FlowStatistics& flow) const;
    double extract_packet_len_mean(const FlowStatistics& flow) const;
    double extract_packet_len_std(const FlowStatistics& flow) const;
    double extract_packet_len_var(const FlowStatistics& flow) const;

    // === FASE 1: NUEVOS EXTRACTORS ===
    double extract_dload(const FlowStatistics& flow) const;
    double extract_rate(const FlowStatistics& flow) const;
    double extract_srate(const FlowStatistics& flow) const;
    double extract_drate(const FlowStatistics& flow) const;
    double extract_spkts_ratio(const FlowStatistics& flow) const;
    double extract_sbytes_ratio(const FlowStatistics& flow) const;
    double extract_flow_iat_max(const FlowStatistics& flow) const;
    double extract_flow_iat_min(const FlowStatistics& flow) const;
    double extract_packet_len_max(const FlowStatistics& flow) const;
    double extract_packet_len_min(const FlowStatistics& flow) const;
    double extract_fwd_len_max(const FlowStatistics& flow) const;
    double extract_fwd_len_min(const FlowStatistics& flow) const;
    double extract_fwd_len_tot(const FlowStatistics& flow) const;
    double extract_bwd_len_max(const FlowStatistics& flow) const;
    double extract_bwd_len_min(const FlowStatistics& flow) const;
    double extract_bwd_len_tot(const FlowStatistics& flow) const;
    double extract_fwd_header_len_mean(const FlowStatistics& flow) const;
    double extract_bwd_header_len_mean(const FlowStatistics& flow) const;

    // === STATISTICAL HELPERS ===
    double calculate_mean(const std::vector<uint32_t>& values) const;
    double calculate_mean_u16(const std::vector<uint16_t>& values) const;
    double calculate_std(const std::vector<uint32_t>& values) const;
    double calculate_variance(const std::vector<uint32_t>& values) const;
    double calculate_max(const std::vector<uint32_t>& values) const;
    double calculate_min(const std::vector<uint32_t>& values) const;
    double calculate_sum(const std::vector<uint32_t>& values) const;

    // === IAT HELPERS ===
    std::vector<uint64_t> compute_inter_arrival_times(const std::vector<uint64_t>& timestamps) const;
    double calculate_iat_mean(const std::vector<uint64_t>& timestamps) const;
    double calculate_iat_std(const std::vector<uint64_t>& timestamps) const;
    double calculate_iat_max(const std::vector<uint64_t>& timestamps) const;
    double calculate_iat_min(const std::vector<uint64_t>& timestamps) const;
};

} // namespace sniffer