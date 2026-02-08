// tools/synthetic_sniffer_injector.cpp
// Synthetic Sniffer → ML Detector Event Generator
// Generates NetworkSecurityEvent with 142 features for stress testing
// AUTHORS: Alonso Isidoro Roman + Claude (Anthropic)
// DATE: 2 February 2026 - Day 49

#include <zmq.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <random>
#include <iomanip>
#include <memory>

// Protobuf
#include "network_security.pb.h"

// Crypto-transport
#include <crypto_transport/crypto_manager.hpp>
#include <crypto_transport/utils.hpp>

// etcd-client
#include <etcd_client/etcd_client.hpp>

namespace fs = std::filesystem;

class SyntheticSnifferInjector {
private:
    zmq::context_t zmq_ctx_;
    zmq::socket_t publisher_;
    std::unique_ptr<etcd_client::EtcdClient> etcd_client_;
    std::string crypto_seed_;
    std::mt19937 rng_;

    // Random generators
    float rand_float(float min, float max) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(rng_);
    }

    uint32_t rand_uint(uint32_t min, uint32_t max) {
        std::uniform_int_distribution<uint32_t> dist(min, max);
        return dist(rng_);
    }

    uint64_t rand_uint64(uint64_t min, uint64_t max) {
        std::uniform_int_distribution<uint64_t> dist(min, max);
        return dist(rng_);
    }

    std::string generate_random_ip() {
        return std::to_string(rand_uint(1, 254)) + "." +
               std::to_string(rand_uint(0, 255)) + "." +
               std::to_string(rand_uint(0, 255)) + "." +
               std::to_string(rand_uint(1, 254));
    }

    // Create complete NetworkSecurityEvent with all 142 features
    protobuf::NetworkSecurityEvent create_synthetic_event(uint64_t event_id) {
        protobuf::NetworkSecurityEvent event;

        // Event ID and timestamp
        event.set_event_id("synthetic-" + std::to_string(event_id));
        auto* timestamp = event.mutable_event_timestamp();
        auto now = std::chrono::system_clock::now();
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch());
        timestamp->set_seconds(seconds.count());

        // Network Features (142 features)
        auto* nf = event.mutable_network_features();

        // Basic identification (6 features)
        nf->set_source_ip(generate_random_ip());
        nf->set_destination_ip(generate_random_ip());
        nf->set_source_port(rand_uint(1024, 65535));
        nf->set_destination_port(rand_uint(1, 1024));
        nf->set_protocol_number(rand_uint(1, 255));
        nf->set_protocol_name(rand_uint(0, 1) ? "TCP" : "UDP");

        // Dual-NIC deployment (4 features)
        nf->set_interface_mode(rand_uint(0, 2));
        nf->set_is_wan_facing(rand_uint(0, 1));
        nf->set_source_ifindex(rand_uint(1, 4));
        nf->set_source_interface("eth" + std::to_string(rand_uint(0, 1)));

        // Timing (3 features)
        auto* flow_start = nf->mutable_flow_start_time();
        flow_start->set_seconds(seconds.count());
        auto* flow_dur = nf->mutable_flow_duration();
        flow_dur->set_seconds(rand_uint(0, 300));
        nf->set_flow_duration_microseconds(rand_uint64(1000, 300000000));

        // Packet statistics (4 features)
        nf->set_total_forward_packets(rand_uint64(1, 10000));
        nf->set_total_backward_packets(rand_uint64(1, 10000));
        nf->set_total_forward_bytes(rand_uint64(64, 15000000));
        nf->set_total_backward_bytes(rand_uint64(64, 15000000));

        // Forward packet length stats (4 features)
        nf->set_forward_packet_length_max(rand_uint64(64, 1500));
        nf->set_forward_packet_length_min(rand_uint64(20, 64));
        nf->set_forward_packet_length_mean(rand_float(64, 1500));
        nf->set_forward_packet_length_std(rand_float(0, 500));

        // Backward packet length stats (4 features)
        nf->set_backward_packet_length_max(rand_uint64(64, 1500));
        nf->set_backward_packet_length_min(rand_uint64(20, 64));
        nf->set_backward_packet_length_mean(rand_float(64, 1500));
        nf->set_backward_packet_length_std(rand_float(0, 500));

        // Flow rates and ratios (8 features)
        nf->set_flow_bytes_per_second(rand_float(100, 100000));
        nf->set_flow_packets_per_second(rand_float(1, 10000));
        nf->set_forward_packets_per_second(rand_float(1, 5000));
        nf->set_backward_packets_per_second(rand_float(1, 5000));
        nf->set_download_upload_ratio(rand_float(0.1, 10.0));
        nf->set_average_packet_size(rand_float(64, 1500));
        nf->set_average_forward_segment_size(rand_float(64, 1460));
        nf->set_average_backward_segment_size(rand_float(64, 1460));

        // Flow IAT (4 features)
        nf->set_flow_inter_arrival_time_mean(rand_float(0.001, 1.0));
        nf->set_flow_inter_arrival_time_std(rand_float(0.0, 0.5));
        nf->set_flow_inter_arrival_time_max(rand_uint64(1000, 2000000));
        nf->set_flow_inter_arrival_time_min(rand_uint64(0, 10000));

        // Forward IAT (5 features)
        nf->set_forward_inter_arrival_time_total(rand_float(0.001, 300));
        nf->set_forward_inter_arrival_time_mean(rand_float(0.001, 1.0));
        nf->set_forward_inter_arrival_time_std(rand_float(0.0, 0.5));
        nf->set_forward_inter_arrival_time_max(rand_uint64(1000, 2000000));
        nf->set_forward_inter_arrival_time_min(rand_uint64(0, 10000));

        // Backward IAT (5 features)
        nf->set_backward_inter_arrival_time_total(rand_float(0.001, 300));
        nf->set_backward_inter_arrival_time_mean(rand_float(0.001, 1.0));
        nf->set_backward_inter_arrival_time_std(rand_float(0.0, 0.5));
        nf->set_backward_inter_arrival_time_max(rand_uint64(1000, 2000000));
        nf->set_backward_inter_arrival_time_min(rand_uint64(0, 10000));

        // TCP Flags (8 features)
        nf->set_fin_flag_count(rand_uint(0, 5));
        nf->set_syn_flag_count(rand_uint(0, 10));
        nf->set_rst_flag_count(rand_uint(0, 5));
        nf->set_psh_flag_count(rand_uint(0, 50));
        nf->set_ack_flag_count(rand_uint(0, 100));
        nf->set_urg_flag_count(rand_uint(0, 2));
        nf->set_cwe_flag_count(rand_uint(0, 1));
        nf->set_ece_flag_count(rand_uint(0, 1));

        // TCP Flags directional (4 features)
        nf->set_forward_psh_flags(rand_uint(0, 25));
        nf->set_backward_psh_flags(rand_uint(0, 25));
        nf->set_forward_urg_flags(rand_uint(0, 1));
        nf->set_backward_urg_flags(rand_uint(0, 1));

        // Headers and bulk (8 features)
        nf->set_forward_header_length(rand_float(20, 60));
        nf->set_backward_header_length(rand_float(20, 60));
        nf->set_forward_average_bytes_bulk(rand_float(0, 100000));
        nf->set_forward_average_packets_bulk(rand_float(0, 100));
        nf->set_forward_average_bulk_rate(rand_float(0, 10000));
        nf->set_backward_average_bytes_bulk(rand_float(0, 100000));
        nf->set_backward_average_packets_bulk(rand_float(0, 100));
        nf->set_backward_average_bulk_rate(rand_float(0, 10000));

        // Additional packet stats (5 features)
        nf->set_minimum_packet_length(rand_uint64(20, 64));
        nf->set_maximum_packet_length(rand_uint64(64, 1500));
        nf->set_packet_length_mean(rand_float(64, 1500));
        nf->set_packet_length_std(rand_float(0, 500));
        nf->set_packet_length_variance(rand_float(0, 250000));

        // Active/Idle (2 features)
        nf->set_active_mean(rand_float(0.001, 10.0));
        nf->set_idle_mean(rand_float(0.001, 10.0));

        // DDoS Embedded Features (10 features)
        auto* ddos = nf->mutable_ddos_embedded();
        ddos->set_syn_ack_ratio(rand_float(0.0, 1.0));
        ddos->set_packet_symmetry(rand_float(0.0, 1.0));
        ddos->set_source_ip_dispersion(rand_float(0.0, 1.0));
        ddos->set_protocol_anomaly_score(rand_float(0.0, 1.0));
        ddos->set_packet_size_entropy(rand_float(0.0, 8.0));
        ddos->set_traffic_amplification_factor(rand_float(1.0, 100.0));
        ddos->set_flow_completion_rate(rand_float(0.0, 1.0));
        ddos->set_geographical_concentration(rand_float(0.0, 1.0));
        ddos->set_traffic_escalation_rate(rand_float(0.0, 10.0));
        ddos->set_resource_saturation_score(rand_float(0.0, 1.0));

        // Ransomware Embedded Features (10 features)
        auto* ransomware = nf->mutable_ransomware_embedded();
        ransomware->set_io_intensity(rand_float(0.0, 1.0));
        ransomware->set_entropy(rand_float(0.0, 8.0));
        ransomware->set_resource_usage(rand_float(0.0, 1.0));
        ransomware->set_network_activity(rand_float(0.0, 1.0));
        ransomware->set_file_operations(rand_float(0.0, 1.0));
        ransomware->set_process_anomaly(rand_float(0.0, 1.0));
        ransomware->set_temporal_pattern(rand_float(0.0, 1.0));
        ransomware->set_access_frequency(rand_float(0.0, 1.0));
        ransomware->set_data_volume(rand_float(0.0, 1.0));
        ransomware->set_behavior_consistency(rand_float(0.0, 1.0));

        // Traffic Classification Features (10 features)
        auto* traffic = nf->mutable_traffic_classification();
        traffic->set_packet_rate(rand_float(0.0, 1.0));
        traffic->set_connection_rate(rand_float(0.0, 1.0));
        traffic->set_tcp_udp_ratio(rand_float(0.0, 1.0));
        traffic->set_avg_packet_size(rand_float(0.0, 1.0));
        traffic->set_port_entropy(rand_float(0.0, 8.0));
        traffic->set_flow_duration_std(rand_float(0.0, 1.0));
        traffic->set_src_ip_entropy(rand_float(0.0, 8.0));
        traffic->set_dst_ip_concentration(rand_float(0.0, 1.0));
        traffic->set_protocol_variety(rand_float(0.0, 1.0));
        traffic->set_temporal_consistency(rand_float(0.0, 1.0));

        // Internal Anomaly Features (10 features)
        auto* internal = nf->mutable_internal_anomaly();
        internal->set_internal_connection_rate(rand_float(0.0, 1.0));
        internal->set_service_port_consistency(rand_float(0.0, 1.0));
        internal->set_protocol_regularity(rand_float(0.0, 1.0));
        internal->set_packet_size_consistency(rand_float(0.0, 1.0));
        internal->set_connection_duration_std(rand_float(0.0, 1.0));
        internal->set_lateral_movement_score(rand_float(0.0, 1.0));
        internal->set_service_discovery_patterns(rand_float(0.0, 1.0));
        internal->set_data_exfiltration_indicators(rand_float(0.0, 1.0));
        internal->set_temporal_anomaly_score(rand_float(0.0, 1.0));
        internal->set_access_pattern_entropy(rand_float(0.0, 8.0));

        // Ransomware 20 features (additional)
        auto* ransomware20 = nf->mutable_ransomware();
        ransomware20->set_dns_query_entropy(rand_float(0.0, 8.0));
        ransomware20->set_new_external_ips_30s(rand_uint(0, 100));
        ransomware20->set_dns_query_rate_per_min(rand_float(0.0, 100.0));
        ransomware20->set_failed_dns_queries_ratio(rand_float(0.0, 1.0));
        ransomware20->set_tls_self_signed_cert_count(rand_uint(0, 10));
        ransomware20->set_non_standard_port_http_count(rand_uint(0, 20));
        ransomware20->set_smb_connection_diversity(rand_uint(0, 50));
        ransomware20->set_rdp_failed_auth_count(rand_uint(0, 10));
        ransomware20->set_new_internal_connections_30s(rand_uint(0, 50));
        ransomware20->set_port_scan_pattern_score(rand_float(0.0, 1.0));
        ransomware20->set_upload_download_ratio_30s(rand_float(0.0, 10.0));
        ransomware20->set_burst_connections_count(rand_uint(0, 100));
        ransomware20->set_unique_destinations_30s(rand_uint(0, 100));
        ransomware20->set_large_upload_sessions_count(rand_uint(0, 20));
        ransomware20->set_nocturnal_activity_flag(rand_uint(0, 1));
        ransomware20->set_connection_rate_stddev(rand_float(0.0, 10.0));
        ransomware20->set_protocol_diversity_score(rand_float(0.0, 1.0));
        ransomware20->set_avg_flow_duration_seconds(rand_float(0.1, 300.0));
        ransomware20->set_tcp_rst_ratio(rand_float(0.0, 1.0));
        ransomware20->set_syn_without_ack_ratio(rand_float(0.0, 1.0));

        // General attack features (repeated double - 23 features)
        for (int i = 0; i < 23; ++i) {
            nf->add_general_attack_features(rand_float(0.0, 1.0));
        }

        // Total: 142+ features

        return event;
    }

public:
    public:
    SyntheticSnifferInjector(const std::string& etcd_endpoint = "localhost:2379")
        : zmq_ctx_(1)
        , publisher_(zmq_ctx_, zmq::socket_type::pub)
        , rng_(std::random_device{}())
    {
        // Bind to ml-detector input port
        publisher_.bind("tcp://*:5571");

        // Parse endpoint: "localhost:2379" → host="localhost", port=2379
        size_t colon_pos = etcd_endpoint.find(':');
        if (colon_pos == std::string::npos) {
            throw std::runtime_error("Invalid etcd endpoint format (expected host:port)");
        }

        std::string host = etcd_endpoint.substr(0, colon_pos);
        int port = std::stoi(etcd_endpoint.substr(colon_pos + 1));

        std::cout << "🔗 [etcd] Initializing etcd-client: " << host << ":" << port << "\n";

        // Build etcd-client Config
        etcd_client::Config etcd_config;
        etcd_config.host = host;
        etcd_config.port = port;
        etcd_config.timeout_seconds = 5;
        etcd_config.component_name = "synthetic-sniffer";
        etcd_config.encryption_enabled = true;
        etcd_config.heartbeat_enabled = true;

        // Initialize etcd-client
        etcd_client_ = std::make_unique<etcd_client::EtcdClient>(etcd_config);

        // Connect and register
        if (!etcd_client_->connect()) {
            throw std::runtime_error("Failed to connect to etcd-server");
        }

        if (!etcd_client_->register_component()) {
            throw std::runtime_error("Failed to register component");
        }

        std::cout << "✅ [etcd] Connected and registered\n";

        // Get encryption key
        crypto_seed_ = etcd_client_->get_encryption_key();
        if (crypto_seed_.empty()) {
            throw std::runtime_error("Encryption key is empty after registration");
        }

        std::cout << "✅ [etcd] Retrieved encryption key (" << crypto_seed_.size() << " hex chars)\n";

        // Give ZMQ time to bind
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        std::cout << "✅ Synthetic Sniffer Injector initialized\n";
        std::cout << "   Port: 5571 (ml-detector input)\n";
        std::cout << "   Encryption: ChaCha20-Poly1305 + LZ4\n\n";
    }

    void inject_events(uint64_t total_events, uint64_t events_per_sec) {
        const auto interval = std::chrono::nanoseconds(1'000'000'000 / events_per_sec);

        std::cout << "🚀 Injecting " << total_events << " events @ "
                  << events_per_sec << " events/sec\n";
        std::cout << "   Interval: " << interval.count() << " ns\n\n";

        auto start_time = std::chrono::steady_clock::now();
        uint64_t sent = 0;
        uint64_t last_report = 0;

        for (uint64_t i = 0; i < total_events; ++i) {
            auto event_start = std::chrono::steady_clock::now();

            // Create event
            auto event = create_synthetic_event(i);

            // Serialize protobuf
            std::string serialized;
            if (!event.SerializeToString(&serialized)) {
                std::cerr << "❌ Failed to serialize event " << i << "\n";
                continue;
            }

            // Convert to bytes
            std::vector<uint8_t> data(serialized.begin(), serialized.end());

            // Compress (LZ4 with 4-byte header)
            auto compressed = crypto_transport::compress(data);

            // Encrypt (ChaCha20-Poly1305)
            auto key = crypto_transport::hex_to_bytes(crypto_seed_);
            auto encrypted = crypto_transport::encrypt(compressed, key);

            // Send via ZMQ
            zmq::message_t msg(encrypted.data(), encrypted.size());
            publisher_.send(msg, zmq::send_flags::dontwait);

            sent++;

            // Progress report every second
            if (sent - last_report >= events_per_sec) {
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                auto elapsed_sec = std::chrono::duration<double>(elapsed).count();
                auto actual_rate = sent / elapsed_sec;

                std::cout << "\r📊 Sent: " << sent << "/" << total_events
                          << " (" << std::fixed << std::setprecision(1)
                          << (100.0 * sent / total_events) << "%) "
                          << "@ " << actual_rate << " events/sec    " << std::flush;

                last_report = sent;
            }

            // Rate limiting
            auto event_end = std::chrono::steady_clock::now();
            auto elapsed = event_end - event_start;
            if (elapsed < interval) {
                std::this_thread::sleep_for(interval - elapsed);
            }
        }

        auto total_time = std::chrono::steady_clock::now() - start_time;
        auto total_sec = std::chrono::duration<double>(total_time).count();

        std::cout << "\n\n✅ Injection complete!\n";
        std::cout << "   Total events: " << sent << "\n";
        std::cout << "   Total time: " << total_sec << " sec\n";
        std::cout << "   Actual rate: " << (sent / total_sec) << " events/sec\n\n";
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <total_events> <events_per_second>\n";
        std::cerr << "Example: " << argv[0] << " 10000 1000\n";
        return 1;
    }

    uint64_t total = std::stoull(argv[1]);
    uint64_t rate = std::stoull(argv[2]);

    try {
        SyntheticSnifferInjector injector;
        injector.inject_events(total, rate);
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}