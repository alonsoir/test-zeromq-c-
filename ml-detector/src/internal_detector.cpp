// internal_detector.cpp
#include "ml_defender/internal_detector.hpp"
#include "ml_defender/internal_trees_inline.hpp"

namespace ml_defender {

InternalDetector::InternalDetector() noexcept {
    // Model validated at compile time via constexpr
}

InternalDetector::Prediction InternalDetector::predict(const Features& features) const noexcept {
    Prediction result;

    // Convert features to std::array for internal_traffic_predict()
    const std::array<float, INTERNAL_NUM_FEATURES> feature_array = {
        features.internal_connection_rate,       // [0]
        features.service_port_consistency,       // [1]
        features.protocol_regularity,            // [2]
        features.packet_size_consistency,        // [3]
        features.connection_duration_std,        // [4]
        features.lateral_movement_score,         // [5]
        features.service_discovery_patterns,     // [6]
        features.data_exfiltration_indicators,   // [7]
        features.temporal_anomaly_score,         // [8]
        features.access_pattern_entropy          // [9]
    };

    // Use the inline prediction function from internal_trees_inline.hpp
    // Returns P(suspicious) - probability of suspicious traffic
    result.suspicious_prob = internal_traffic_predict(feature_array);
    result.benign_prob = 1.0f - result.suspicious_prob;

    // Determine predicted class and probability of predicted class
    if (result.suspicious_prob >= 0.5f) {
        result.class_id = 1;  // Suspicious
        result.probability = result.suspicious_prob;
    } else {
        result.class_id = 0;  // Benign
        result.probability = result.benign_prob;
    }

    return result;
}

std::vector<InternalDetector::Prediction> InternalDetector::predict_batch(
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