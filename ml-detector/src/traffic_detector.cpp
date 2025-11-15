// traffic_detector.cpp
#include "ml_defender/traffic_detector.hpp"
#include "ml_defender/traffic_trees_inline.hpp"

namespace ml_defender {

TrafficDetector::TrafficDetector() noexcept {
    // Model validated at compile time via constexpr
}

TrafficDetector::Prediction TrafficDetector::predict(const Features& features) const noexcept {
    Prediction result;

    // Convert features to std::array for traffic_predict()
    const std::array<float, TRAFFIC_NUM_FEATURES> feature_array = {
        features.packet_rate,                    // [0]
        features.connection_rate,                // [1]
        features.tcp_udp_ratio,                  // [2]
        features.avg_packet_size,                // [3]
        features.port_entropy,                   // [4]
        features.flow_duration_std,              // [5]
        features.src_ip_entropy,                 // [6]
        features.dst_ip_concentration,           // [7]
        features.protocol_variety,               // [8]
        features.temporal_consistency            // [9]
    };

    // Use the inline prediction function from traffic_trees_inline.hpp
    // Returns P(internal) - probability of internal traffic
    result.internal_prob = traffic_predict(feature_array);
    result.internet_prob = 1.0f - result.internal_prob;

    // Determine predicted class and probability of predicted class
    if (result.internal_prob >= 0.5f) {
        result.class_id = 1;  // Internal
        result.probability = result.internal_prob;
    } else {
        result.class_id = 0;  // Internet
        result.probability = result.internet_prob;
    }

    return result;
}

std::vector<TrafficDetector::Prediction> TrafficDetector::predict_batch(
    const std::vector<Features>& features_batch
) const {
    std::vector<Prediction> results;
    results.reserve(features_batch.size());

    for (const auto& features : features_batch) {
        results.push_back(predict(features));
    }

    return results;
}

} // namespace ml_defender