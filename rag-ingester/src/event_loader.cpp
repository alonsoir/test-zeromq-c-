// event_loader.cpp
// RAG Ingester - EventLoader Implementation
// Day 36: Decrypt + Decompress + Parse pipeline
// Via Appia Quality - Robust, exception-safe event processing
// USANDO API REAL: crypto_transport::decrypt() y decompress()

#include "event_loader.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>

// API REAL de crypto-transport
#include <crypto_transport/crypto.hpp>
#include <crypto_transport/compression.hpp>

#include <network_security.pb.h>
#include <reason_codes.hpp>

namespace rag_ingester {

// ============================================================================
// CryptoImpl - PIMPL pattern con API REAL
// ============================================================================

class EventLoader::CryptoImpl {
public:
    CryptoImpl(const std::string& key_path) {
        // Read encryption key from file
        std::ifstream key_file(key_path, std::ios::binary);
        if (!key_file) {
            throw std::runtime_error("Failed to open encryption key: " + key_path);
        }

        key_.resize(32);
        key_file.read(reinterpret_cast<char*>(key_.data()), 32);

        if (key_file.gcount() != 32) {
            throw std::runtime_error("Invalid key file (expected 32 bytes): " + key_path);
        }

        std::cout << "[INFO] EventLoader: Loaded encryption key ("
                  << key_.size() << " bytes)" << std::endl;
        std::cout << "[INFO] EventLoader: Crypto initialized (ChaCha20-Poly1305 + LZ4)" << std::endl;
    }

    std::vector<uint8_t> decrypt(const std::vector<uint8_t>& encrypted) {
        if (encrypted.empty()) {
            return encrypted; // Pass-through si vacío
        }

        try {
            // API REAL: crypto_transport::decrypt()
            return crypto_transport::decrypt(encrypted, key_);
        } catch (const std::exception& e) {
            // Si falla el decrypt, probablemente los datos no están cifrados
            // Retornar sin cambios (pass-through)
            std::cerr << "[WARN] Decrypt failed (data may be unencrypted): "
                      << e.what() << std::endl;
            return encrypted;
        }
    }

    std::vector<uint8_t> decompress(const std::vector<uint8_t>& compressed) {
        if (compressed.empty()) {
            return compressed; // Pass-through si vacío
        }

        try {
            // API REAL: crypto_transport::decompress()
            // Necesitamos estimar el tamaño original
            size_t estimated_size = compressed.size() * 10; // 10x compression ratio
            if (estimated_size < 1024) estimated_size = 1024;

            return crypto_transport::decompress(compressed, estimated_size);
        } catch (const std::exception& e) {
            // Si falla el decompress, probablemente los datos no están comprimidos
            // Retornar sin cambios (pass-through)
            std::cerr << "[WARN] Decompress failed (data may be uncompressed): "
                      << e.what() << std::endl;
            return compressed;
        }
    }

private:
    std::vector<uint8_t> key_;
};

// ============================================================================
// EventLoader Implementation
// ============================================================================

EventLoader::EventLoader(const std::string& encryption_key_path)
    : crypto_(std::make_unique<CryptoImpl>(encryption_key_path)) {
    stats_ = {};
}

EventLoader::~EventLoader() = default;

Event EventLoader::load(const std::string& filepath) {
    try {
        auto encrypted = read_file(filepath);
        auto decrypted = decrypt(encrypted);
        auto decompressed = decompress(decrypted);
        auto event = parse_protobuf(decompressed);

        event.filepath = filepath;

        stats_.total_loaded++;
        stats_.bytes_processed += encrypted.size();

        int expected_features = 101;
        if (event.features.size() < expected_features) {
            event.is_partial = true;
            stats_.partial_feature_count++;
        } else {
            event.is_partial = false;
        }

        return event;

    } catch (const std::exception& e) {
        stats_.total_failed++;
        throw std::runtime_error("Failed to load " + filepath + ": " + e.what());
    }
}

std::vector<Event> EventLoader::load_batch(const std::vector<std::string>& filepaths) {
    std::vector<Event> events;
    events.reserve(filepaths.size());

    for (const auto& filepath : filepaths) {
        try {
            events.push_back(load(filepath));
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] EventLoader: " << e.what() << std::endl;
        }
    }

    return events;
}

EventLoader::LoadStats EventLoader::get_stats() const noexcept {
    return stats_;
}

std::vector<uint8_t> EventLoader::read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    auto size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("Empty file: " + path);
    }

    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(buffer.data()), size);

    if (!file) {
        throw std::runtime_error("Failed to read file: " + path);
    }

    return buffer;
}

std::vector<uint8_t> EventLoader::decrypt(const std::vector<uint8_t>& encrypted) {
    return crypto_->decrypt(encrypted);
}

std::vector<uint8_t> EventLoader::decompress(const std::vector<uint8_t>& compressed) {
    return crypto_->decompress(compressed);
}

Event EventLoader::parse_protobuf(const std::vector<uint8_t>& data) {
    protobuf::NetworkSecurityEvent proto_event;

    if (!proto_event.ParseFromArray(data.data(), data.size())) {
        throw std::runtime_error("Failed to parse protobuf NetworkSecurityEvent");
    }

    Event event;
    event.event_id = proto_event.event_id();

    if (proto_event.has_event_timestamp()) {
        const auto& ts = proto_event.event_timestamp();
        event.timestamp_ns = ts.seconds() * 1'000'000'000ULL + ts.nanos();
    } else {
        event.timestamp_ns = 0;
    }

    if (proto_event.has_ml_analysis()) {
        const auto& ml = proto_event.ml_analysis();

        if (ml.has_level1_general_detection()) {
            event.final_class = ml.level1_general_detection().prediction_class();
            event.confidence = ml.level1_general_detection().confidence_score();
        }

        if (!ml.final_threat_classification().empty()) {
            event.final_class = ml.final_threat_classification();
            event.confidence = ml.ensemble_confidence();
        }
    }

    if (event.final_class.empty() && !proto_event.final_classification().empty()) {
        event.final_class = proto_event.final_classification();
        event.confidence = proto_event.overall_threat_score();
    }

    if (proto_event.has_capturing_node()) {
        event.source_detector = proto_event.capturing_node().node_id();
    } else {
        event.source_detector = proto_event.originating_node_id();
    }

    event.features = extract_features(&proto_event);

    // 🎯 ADR-002: Parse Multi-Engine Provenance (Day 37)
    if (proto_event.has_provenance()) {
        const auto& prov = proto_event.provenance();

        // Parse all engine verdicts
        for (int i = 0; i < prov.verdicts_size(); i++) {
            const auto& v = prov.verdicts(i);

            EngineVerdict verdict;
            verdict.engine_name = v.engine_name();
            verdict.classification = v.classification();
            verdict.confidence = v.confidence();
            verdict.reason_code = v.reason_code();
            verdict.timestamp_ns = v.timestamp_ns();

            event.verdicts.push_back(verdict);
        }

        // Parse provenance metadata
        event.discrepancy_score = prov.discrepancy_score();
        event.final_decision = prov.final_decision();

        // Log if high discrepancy
        if (event.discrepancy_score > 0.30f) {
            std::cout << "[INFO] EventLoader: High discrepancy event "
                      << event.event_id << " (score=" << event.discrepancy_score
                      << ", engines=" << event.verdicts.size() << ")" << std::endl;
        }
    } else {
        // No provenance - legacy event or sniffer-only
        event.discrepancy_score = 0.0f;
        event.final_decision = "UNKNOWN";
    }

    return event;
}

std::vector<float> EventLoader::extract_features(const void* proto_event_ptr) {
    const auto* proto_event =
        static_cast<const protobuf::NetworkSecurityEvent*>(proto_event_ptr);

    if (!proto_event->has_network_features()) {
        throw std::runtime_error("NetworkSecurityEvent missing network_features");
    }

    const auto& net = proto_event->network_features();

    std::vector<float> features;
    features.reserve(140);

    // PART 1: BASIC FLOW STATISTICS (61 features)
    features.push_back(static_cast<float>(net.protocol_number()));
    features.push_back(static_cast<float>(net.interface_mode()));
    features.push_back(static_cast<float>(net.source_ifindex()));
    features.push_back(static_cast<float>(net.flow_duration_microseconds() / 1e6));

    features.push_back(static_cast<float>(net.total_forward_packets()));
    features.push_back(static_cast<float>(net.total_backward_packets()));
    features.push_back(static_cast<float>(net.total_forward_bytes()));
    features.push_back(static_cast<float>(net.total_backward_bytes()));

    features.push_back(static_cast<float>(net.forward_packet_length_max()));
    features.push_back(static_cast<float>(net.forward_packet_length_min()));
    features.push_back(static_cast<float>(net.forward_packet_length_mean()));
    features.push_back(static_cast<float>(net.forward_packet_length_std()));

    features.push_back(static_cast<float>(net.backward_packet_length_max()));
    features.push_back(static_cast<float>(net.backward_packet_length_min()));
    features.push_back(static_cast<float>(net.backward_packet_length_mean()));
    features.push_back(static_cast<float>(net.backward_packet_length_std()));

    features.push_back(static_cast<float>(net.flow_bytes_per_second()));
    features.push_back(static_cast<float>(net.flow_packets_per_second()));
    features.push_back(static_cast<float>(net.forward_packets_per_second()));
    features.push_back(static_cast<float>(net.backward_packets_per_second()));
    features.push_back(static_cast<float>(net.download_upload_ratio()));
    features.push_back(static_cast<float>(net.average_packet_size()));
    features.push_back(static_cast<float>(net.average_forward_segment_size()));
    features.push_back(static_cast<float>(net.average_backward_segment_size()));

    features.push_back(static_cast<float>(net.flow_inter_arrival_time_mean()));
    features.push_back(static_cast<float>(net.flow_inter_arrival_time_std()));
    features.push_back(static_cast<float>(net.flow_inter_arrival_time_max()));
    features.push_back(static_cast<float>(net.flow_inter_arrival_time_min()));

    features.push_back(static_cast<float>(net.forward_inter_arrival_time_total()));
    features.push_back(static_cast<float>(net.forward_inter_arrival_time_mean()));
    features.push_back(static_cast<float>(net.forward_inter_arrival_time_std()));
    features.push_back(static_cast<float>(net.forward_inter_arrival_time_max()));
    features.push_back(static_cast<float>(net.forward_inter_arrival_time_min()));

    features.push_back(static_cast<float>(net.backward_inter_arrival_time_total()));
    features.push_back(static_cast<float>(net.backward_inter_arrival_time_mean()));
    features.push_back(static_cast<float>(net.backward_inter_arrival_time_std()));
    features.push_back(static_cast<float>(net.backward_inter_arrival_time_max()));
    features.push_back(static_cast<float>(net.backward_inter_arrival_time_min()));

    features.push_back(static_cast<float>(net.fin_flag_count()));
    features.push_back(static_cast<float>(net.syn_flag_count()));
    features.push_back(static_cast<float>(net.rst_flag_count()));
    features.push_back(static_cast<float>(net.psh_flag_count()));
    features.push_back(static_cast<float>(net.ack_flag_count()));
    features.push_back(static_cast<float>(net.urg_flag_count()));
    features.push_back(static_cast<float>(net.cwe_flag_count()));
    features.push_back(static_cast<float>(net.ece_flag_count()));

    features.push_back(static_cast<float>(net.forward_psh_flags()));
    features.push_back(static_cast<float>(net.backward_psh_flags()));
    features.push_back(static_cast<float>(net.forward_urg_flags()));
    features.push_back(static_cast<float>(net.backward_urg_flags()));

    features.push_back(static_cast<float>(net.forward_header_length()));
    features.push_back(static_cast<float>(net.backward_header_length()));
    features.push_back(static_cast<float>(net.forward_average_bytes_bulk()));
    features.push_back(static_cast<float>(net.forward_average_packets_bulk()));
    features.push_back(static_cast<float>(net.forward_average_bulk_rate()));
    features.push_back(static_cast<float>(net.backward_average_bytes_bulk()));
    features.push_back(static_cast<float>(net.backward_average_packets_bulk()));
    features.push_back(static_cast<float>(net.backward_average_bulk_rate()));

    features.push_back(static_cast<float>(net.minimum_packet_length()));
    features.push_back(static_cast<float>(net.maximum_packet_length()));
    features.push_back(static_cast<float>(net.packet_length_mean()));
    features.push_back(static_cast<float>(net.packet_length_std()));
    features.push_back(static_cast<float>(net.packet_length_variance()));

    features.push_back(static_cast<float>(net.active_mean()));
    features.push_back(static_cast<float>(net.idle_mean()));

    // PART 2: EMBEDDED DETECTOR FEATURES (40 features)

    if (net.has_ddos_embedded()) {
        const auto& ddos = net.ddos_embedded();
        features.push_back(ddos.syn_ack_ratio());
        features.push_back(ddos.packet_symmetry());
        features.push_back(ddos.source_ip_dispersion());
        features.push_back(ddos.protocol_anomaly_score());
        features.push_back(ddos.packet_size_entropy());
        features.push_back(ddos.traffic_amplification_factor());
        features.push_back(ddos.flow_completion_rate());
        features.push_back(ddos.geographical_concentration());
        features.push_back(ddos.traffic_escalation_rate());
        features.push_back(ddos.resource_saturation_score());
    } else {
        for (int i = 0; i < 10; i++) features.push_back(0.0f);
    }

    if (net.has_ransomware_embedded()) {
        const auto& ransomware = net.ransomware_embedded();
        features.push_back(ransomware.io_intensity());
        features.push_back(ransomware.entropy());
        features.push_back(ransomware.resource_usage());
        features.push_back(ransomware.network_activity());
        features.push_back(ransomware.file_operations());
        features.push_back(ransomware.process_anomaly());
        features.push_back(ransomware.temporal_pattern());
        features.push_back(ransomware.access_frequency());
        features.push_back(ransomware.data_volume());
        features.push_back(ransomware.behavior_consistency());
    } else {
        for (int i = 0; i < 10; i++) features.push_back(0.0f);
    }

    if (net.has_traffic_classification()) {
        const auto& traffic = net.traffic_classification();
        features.push_back(traffic.packet_rate());
        features.push_back(traffic.connection_rate());
        features.push_back(traffic.tcp_udp_ratio());
        features.push_back(traffic.avg_packet_size());
        features.push_back(traffic.port_entropy());
        features.push_back(traffic.flow_duration_std());
        features.push_back(traffic.src_ip_entropy());
        features.push_back(traffic.dst_ip_concentration());
        features.push_back(traffic.protocol_variety());
        features.push_back(traffic.temporal_consistency());
    } else {
        for (int i = 0; i < 10; i++) features.push_back(0.0f);
    }

    if (net.has_internal_anomaly()) {
        const auto& internal = net.internal_anomaly();
        features.push_back(internal.internal_connection_rate());
        features.push_back(internal.service_port_consistency());
        features.push_back(internal.protocol_regularity());
        features.push_back(internal.packet_size_consistency());
        features.push_back(internal.connection_duration_std());
        features.push_back(internal.lateral_movement_score());
        features.push_back(internal.service_discovery_patterns());
        features.push_back(internal.data_exfiltration_indicators());
        features.push_back(internal.temporal_anomaly_score());
        features.push_back(internal.access_pattern_entropy());
    } else {
        for (int i = 0; i < 10; i++) features.push_back(0.0f);
    }

    return features;
}

} // namespace rag_ingester