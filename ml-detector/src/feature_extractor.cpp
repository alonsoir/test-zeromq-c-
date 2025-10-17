#include "feature_extractor.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

namespace ml_detector {

// Nombres de las 23 features (del metadata)
static const std::vector<std::string> LEVEL1_FEATURE_NAMES = {
    "Packet Length Std",
    "Subflow Fwd Bytes",
    "Fwd Packet Length Max",
    "Avg Fwd Segment Size",
    "ACK Flag Count",
    "Packet Length Variance",
    "PSH Flag Count",
    "Bwd Packet Length Max",
    "act_data_pkt_fwd",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Std",
    "Fwd Packets/s",
    "Subflow Bwd Bytes",
    "Destination Port",
    "Init_Win_bytes_forward",
    "Subflow Fwd Packets",
    "Fwd IAT Min",
    "Packet Length Mean",
    "Total Length of Bwd Packets",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Min",
    "Flow Duration",
    "Flow Packets/s"
};

FeatureExtractor::FeatureExtractor()
    : logger_(spdlog::get("ml-detector")) {
    if (!logger_) {
        logger_ = spdlog::stdout_color_mt("feature-extractor");
    }
}

const std::vector<std::string>& FeatureExtractor::get_feature_names() {
    return LEVEL1_FEATURE_NAMES;
}

float FeatureExtractor::safe_divide(float numerator, float denominator, float default_value) {
    if (denominator == 0.0f || std::isnan(denominator) || std::isinf(denominator)) {
        return default_value;
    }
    float result = numerator / denominator;
    if (std::isnan(result) || std::isinf(result)) {
        return default_value;
    }
    return result;
}

float FeatureExtractor::calculate_std_dev(const std::vector<float>& values, float mean) {
    if (values.empty()) return 0.0f;
    
    float sum_sq_diff = 0.0f;
    for (float val : values) {
        float diff = val - mean;
        sum_sq_diff += diff * diff;
    }
    
    return std::sqrt(sum_sq_diff / static_cast<float>(values.size()));
}

float FeatureExtractor::calculate_variance(const std::vector<float>& values, float mean) {
    if (values.empty()) return 0.0f;
    
    float sum_sq_diff = 0.0f;
    for (float val : values) {
        float diff = val - mean;
        sum_sq_diff += diff * diff;
    }
    
    return sum_sq_diff / static_cast<float>(values.size());
}

std::vector<float> FeatureExtractor::extract_level1_features(const protobuf::NetworkSecurityEvent& event) {
    std::vector<float> features(23, 0.0f);
    
    // Verificar que tenemos NetworkFeatures
    if (!event.has_network_features()) {
        logger_->error("NetworkSecurityEvent missing network_features");
        throw std::runtime_error("Missing network_features in event");
    }
    
    const auto& nf = event.network_features();
    
    // Duración del flow (microsegundos → segundos)
    float flow_duration_sec = static_cast<float>(nf.flow_duration_microseconds()) / 1000000.0f;
    
    // Feature 0: Packet Length Std
    features[0] = static_cast<float>(nf.packet_length_std());
    
    // Feature 1: Subflow Fwd Bytes
    features[1] = static_cast<float>(nf.total_forward_bytes());
    
    // Feature 2: Fwd Packet Length Max
    features[2] = static_cast<float>(nf.forward_packet_length_max());
    
    // Feature 3: Avg Fwd Segment Size
    features[3] = static_cast<float>(nf.average_forward_segment_size());
    
    // Feature 4: ACK Flag Count
    features[4] = static_cast<float>(nf.ack_flag_count());
    
    // Feature 5: Packet Length Variance
    features[5] = static_cast<float>(nf.packet_length_variance());
    
    // Feature 6: PSH Flag Count
    features[6] = static_cast<float>(nf.psh_flag_count());
    
    // Feature 7: Bwd Packet Length Max
    features[7] = static_cast<float>(nf.backward_packet_length_max());
    
    // Feature 8: act_data_pkt_fwd
    // Packets con payload - asumimos que total_forward_packets incluye data packets
    features[8] = static_cast<float>(nf.total_forward_packets());
    
    // Feature 9: Total Length of Fwd Packets
    features[9] = static_cast<float>(nf.total_forward_bytes());
    
    // Feature 10: Fwd Packet Length Std
    features[10] = static_cast<float>(nf.forward_packet_length_std());
    
    // Feature 11: Fwd Packets/s
    features[11] = static_cast<float>(nf.forward_packets_per_second());
    
    // Feature 12: Subflow Bwd Bytes
    features[12] = static_cast<float>(nf.total_backward_bytes());
    
    // Feature 13: Destination Port
    features[13] = static_cast<float>(nf.destination_port());
    
    // Feature 14: Init_Win_bytes_forward
    // Si no está en el protobuf, podemos dejarlo en 0 o calcular aproximación
    features[14] = 0.0f;  // TODO: Añadir campo al protobuf si es crítico
    
    // Feature 15: Subflow Fwd Packets
    features[15] = static_cast<float>(nf.total_forward_packets());
    
    // Feature 16: Fwd IAT Min
    features[16] = static_cast<float>(nf.forward_inter_arrival_time_min());
    
    // Feature 17: Packet Length Mean
    features[17] = static_cast<float>(nf.packet_length_mean());
    
    // Feature 18: Total Length of Bwd Packets
    features[18] = static_cast<float>(nf.total_backward_bytes());
    
    // Feature 19: Bwd Packet Length Mean
    features[19] = static_cast<float>(nf.backward_packet_length_mean());
    
    // Feature 20: Bwd Packet Length Min
    features[20] = static_cast<float>(nf.backward_packet_length_min());
    
    // Feature 21: Flow Duration
    features[21] = flow_duration_sec;
    
    // Feature 22: Flow Packets/s
    features[22] = static_cast<float>(nf.flow_packets_per_second());
    
    // Logging detallado en modo debug
    if (logger_->level() <= spdlog::level::debug) {
        logger_->debug("Feature Extraction:");
        logger_->debug("  Flow: {}:{} -> {}:{} ({})", 
                      nf.source_ip(), nf.source_port(),
                      nf.destination_ip(), nf.destination_port(),
                      nf.protocol_name());
        logger_->debug("  Duration: {:.3f}s", flow_duration_sec);
        logger_->debug("  Packets: {} fwd, {} bwd, {} total", 
                      nf.total_forward_packets(), 
                      nf.total_backward_packets(),
                      nf.total_forward_packets() + nf.total_backward_packets());
        logger_->debug("  Bytes: {} fwd, {} bwd, {} total",
                      nf.total_forward_bytes(), 
                      nf.total_backward_bytes(),
                      nf.total_forward_bytes() + nf.total_backward_bytes());
        
        for (size_t i = 0; i < features.size(); ++i) {
            logger_->debug("  Feature[{:2d}] {:30s}: {:12.6f}", 
                          i, LEVEL1_FEATURE_NAMES[i], features[i]);
        }
    }
    
    return features;
}

bool FeatureExtractor::validate_features(const std::vector<float>& features) {
    if (features.size() != 23) {
        logger_->error("Invalid feature vector size: {} (expected 23)", features.size());
        return false;
    }
    
    for (size_t i = 0; i < features.size(); ++i) {
        if (std::isnan(features[i])) {
            logger_->error("Feature[{}] {} is NaN", i, LEVEL1_FEATURE_NAMES[i]);
            return false;
        }
        
        if (std::isinf(features[i])) {
            logger_->error("Feature[{}] {} is Inf", i, LEVEL1_FEATURE_NAMES[i]);
            return false;
        }
        
        // Rangos razonables (básico)
        if (features[i] < -1e9 || features[i] > 1e9) {
            logger_->warn("Feature[{}] {} has suspicious value: {:.2e}", 
                         i, LEVEL1_FEATURE_NAMES[i], features[i]);
        }
    }
    
    return true;
}

} // namespace ml_detector
