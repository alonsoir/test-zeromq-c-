// sniffer/include/feature_extractor.hpp
#pragma once

#include "flow_manager.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>

namespace sniffer {

// Feature Extractor - Computes the 23 ML features from flow statistics
// CRITICAL: Feature order MUST match sniffer_feature_mapping.txt exactly
class FeatureExtractor {
public:
    // Feature indices for clarity
    enum FeatureIndex {
        DURATION = 0,           // 1. duration
        SPKTS = 1,              // 2. spkts
        DPKTS = 2,              // 3. dpkts
        SBYTES = 3,             // 4. sbytes
        DBYTES = 4,             // 5. dbytes
        SLOAD = 5,              // 6. sload
        SMEAN = 6,              // 7. smean
        DMEAN = 7,              // 8. dmean
        FLOW_IAT_MEAN = 8,      // 9. flow_iat_mean
        FLOW_IAT_STD = 9,       // 10. flow_iat_std
        FWD_PSH_FLAGS = 10,     // 11. fwd_psh_flags
        BWD_PSH_FLAGS = 11,     // 12. bwd_psh_flags
        FWD_URG_FLAGS = 12,     // 13. fwd_urg_flags
        BWD_URG_FLAGS = 13,     // 14. bwd_urg_flags
        PACKET_LEN_MEAN = 14,   // 15. packet_len_mean
        PACKET_LEN_STD = 15,    // 16. packet_len_std
        PACKET_LEN_VAR = 16,    // 17. packet_len_var
        FIN_FLAG_COUNT = 17,    // 18. fin_flag_count
        SYN_FLAG_COUNT = 18,    // 19. syn_flag_count
        RST_FLAG_COUNT = 19,    // 20. rst_flag_count
        PSH_FLAG_COUNT = 20,    // 21. psh_flag_count
        ACK_FLAG_COUNT = 21,    // 22. ack_flag_count
        URG_FLAG_COUNT = 22,    // 23. urg_flag_count

        FEATURE_COUNT = 23
    };

    FeatureExtractor() = default;
    ~FeatureExtractor() = default;

    // Extract all 23 features from flow statistics
    // Returns features in EXACT order required by ML model
    std::array<double, FEATURE_COUNT> extract_features(const FlowStatistics& flow) const;

    // Individual feature extraction (for debugging/testing)
    double extract_duration(const FlowStatistics& flow) const;
    double extract_sload(const FlowStatistics& flow) const;
    double extract_smean(const FlowStatistics& flow) const;
    double extract_dmean(const FlowStatistics& flow) const;
    double extract_flow_iat_mean(const FlowStatistics& flow) const;
    double extract_flow_iat_std(const FlowStatistics& flow) const;
    double extract_packet_len_mean(const FlowStatistics& flow) const;
    double extract_packet_len_std(const FlowStatistics& flow) const;
    double extract_packet_len_var(const FlowStatistics& flow) const;

    // Feature names for logging/debugging
    static const char* get_feature_name(size_t index);

private:
    // Statistical helpers
    double calculate_mean(const std::vector<uint32_t>& values) const;
    double calculate_std(const std::vector<uint32_t>& values) const;
    double calculate_variance(const std::vector<uint32_t>& values) const;

    // Inter-arrival time helpers
    double calculate_iat_mean(const std::vector<uint64_t>& timestamps) const;
    double calculate_iat_std(const std::vector<uint64_t>& timestamps) const;
    std::vector<uint64_t> compute_inter_arrival_times(const std::vector<uint64_t>& timestamps) const;
};

} // namespace sniffer