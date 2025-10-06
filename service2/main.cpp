// ============================================================================
// service2/main_simple_fixed.cpp - CORREGIDO con includes correctos
// ============================================================================

#include "service2_main.h"  // No "config_types.h", usar nombre específico
#include "EtcdServiceRegistry.h"  // Sin "../common/" - mismo directorio en Docker
#include <zmq.hpp>
#include <protobuf/network_security.pb.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>
#include <cstdlib>
#include <json/json.h>
#include <iomanip>

std::atomic<bool> g_running(true);

void signalHandler(int signum) {
    std::cout << "\n[Service2] Señal recibida: " << signum << ". Iniciando shutdown..." << std::endl;
    g_running = false;
}

AnalysisResult analyzeNetworkEvent(const protobuf::NetworkSecurityEvent& event) {
    AnalysisResult result;
    result.ddos_probability = 0.0;
    result.anomaly_score = 0.0;
    result.should_block = false;

    const auto& features = event.network_features();
    const auto& geo = event.geo_enrichment();

    double suspicious_score = 0.0;

    // Análisis de tasa de paquetes
    if (features.forward_packets_per_second() > 1000) {
        suspicious_score += 0.3;
        result.suspicious_features.push_back("high_packet_rate");
    }

    // Ratio anómalo de paquetes
    double packet_ratio = features.total_backward_packets() > 0 ?
        static_cast<double>(features.total_forward_packets()) / features.total_backward_packets() :
        features.total_forward_packets();

    if (packet_ratio > 50) {
        suspicious_score += 0.4;
        result.suspicious_features.push_back("abnormal_packet_ratio");
    }

    // Duración de flujo
    if (features.has_flow_duration()) {
        auto duration_seconds = features.flow_duration().seconds() +
                               (features.flow_duration().nanos() / 1000000000.0);
        if (duration_seconds < 0.1) {
            suspicious_score += 0.2;
            result.suspicious_features.push_back("short_flow_duration");
        }
    }

    // Análisis geográfico
    if (geo.has_source_ip_geo()) {
        const auto& source_geo = geo.source_ip_geo();

        if (source_geo.is_known_malicious()) {
            suspicious_score += 0.5;
            result.suspicious_features.push_back("known_malicious_ip");
        }

        if (source_geo.is_tor_exit_node()) {
            suspicious_score += 0.3;
            result.suspicious_features.push_back("tor_exit_node");
        }

        if (source_geo.threat_level() == protobuf::GeoLocationInfo::HIGH ||
            source_geo.threat_level() == protobuf::GeoLocationInfo::CRITICAL) {
            suspicious_score += 0.4;
            result.suspicious_features.push_back("high_threat_country");
        }
    }

    // SYN flood detection
    if (features.syn_flag_count() > 100 && features.ack_flag_count() < 10) {
        suspicious_score += 0.6;
        result.suspicious_features.push_back("syn_flood_pattern");
    }

    result.ddos_probability = std::min(suspicious_score, 1.0);
    result.anomaly_score = result.ddos_probability;

    if (result.ddos_probability > 0.9) {
        result.threat_type = "DDOS_HIGH_CONFIDENCE";
        result.should_block = true;
    } else if (result.ddos_probability > 0.7) {
        result.threat_type = "DDOS_MEDIUM_CONFIDENCE";
    } else if (result.ddos_probability > 0.4) {
        result.threat_type = "SUSPICIOUS_ACTIVITY";
    } else {
        result.threat_type = "NORMAL_TRAFFIC";
    }

    return result;
}

void displayNetworkFeatures(const protobuf::NetworkFeatures& features) {
    std::cout << "    🌐 Network Features:" << std::endl;
    std::cout << "      Flow: " << features.source_ip() << ":" << features.source_port()
              << " → " << features.destination_ip() << ":" << features.destination_port() << std::endl;
    std::cout << "      Protocol: " << features.protocol_name() << " (" << features.protocol_number() << ")" << std::endl;
    std::cout << "      Packets: F=" << features.total_forward_packets()
              << ", B=" << features.total_backward_packets() << std::endl;
    std::cout << "      Rate: " << features.forward_packets_per_second() << " pps" << std::endl;
    std::cout << "      TCP Flags: SYN=" << features.syn_flag_count()
              << ", ACK=" << features.ack_flag_count() << std::endl;
}

void displayGeoEnrichment(const protobuf::GeoEnrichment& geo) {
    std::cout << "    🗺️ Geo Enrichment:" << std::endl;

    if (geo.has_source_ip_geo()) {
        const auto& source_geo = geo.source_ip_geo();
        std::cout << "      Origin: " << source_geo.city_name() << ", " << source_geo.country_name() << std::endl;
    }

    std::cout << "      Distance: " << std::fixed << std::setprecision(1)
              << geo.source_destination_distance_km() << " km" << std::endl;
}

void displayDistributedNode(const protobuf::DistributedNode& node) {
    std::cout << "    🖥️ Node: " << node.node_id() << " (" << node.node_hostname() << ")" << std::endl;
}

int main() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    std::cout << "[Service2] Iniciando Feature Processor con integración etcd..." << std::endl;

    try {
        std::string node_id = std::getenv("NODE_ID") ? std::getenv("NODE_ID") : "service2_node_001";
        std::string service_name = std::getenv("SERVICE_NAME") ? std::getenv("SERVICE_NAME") : "feature_processor";
        std::string etcd_endpoint = std::getenv("ETCD_ENDPOINTS") ? std::getenv("ETCD_ENDPOINTS") : "http://etcd:2379";
        std::string zmq_endpoint = std::getenv("ZMQ_CONNECT_ENDPOINT") ? std::getenv("ZMQ_CONNECT_ENDPOINT") : "tcp://service1:5555";

        std::cout << "[Service2] Conectando a: " << zmq_endpoint << std::endl;

        // Registrarse en etcd
        auto& etcd_registry = EtcdServiceRegistry::getInstance(etcd_endpoint);
        std::string service_config = "{}"; // Config básico por ahora

        if (!etcd_registry.registerService(service_name, node_id, service_config, 30)) {
            std::cerr << "[Service2] ERROR: No se pudo registrar en etcd" << std::endl;
            return 1;
        }

        std::cout << "[Service2] ✅ Registrado en etcd" << std::endl;

        // Esperar a que service1 esté listo
        std::this_thread::sleep_for(std::chrono::seconds(5));

        zmq::context_t context(1);
        zmq::socket_t socket(context, ZMQ_PULL);
        socket.connect(zmq_endpoint);
        socket.set(zmq::sockopt::rcvtimeo, 1000);

        std::cout << "[Service2] ✅ Conectado. Iniciando procesamiento..." << std::endl;

        int events_processed = 0;
        int suspicious_events = 0;
        int high_risk_events = 0;
        auto start_time = std::chrono::steady_clock::now();

        while (g_running) {
            try {
                zmq::message_t message;
                auto result = socket.recv(message, zmq::recv_flags::dontwait);

                if (!result) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }

                protobuf::NetworkSecurityEvent event;
                std::string data(static_cast<char*>(message.data()), message.size());

                if (!event.ParseFromString(data)) {
                    std::cerr << "[Service2] Error deserializando evento" << std::endl;
                    continue;
                }

                events_processed++;
                auto analysis = analyzeNetworkEvent(event);

                if (analysis.ddos_probability > 0.8) {
                    high_risk_events++;
                    suspicious_events++;
                } else if (analysis.ddos_probability > 0.4) {
                    suspicious_events++;
                }

                // Log eventos sospechosos
                if (analysis.ddos_probability > 0.6) {
                    std::cout << "\n[Service2] 🚨 EVENTO SOSPECHOSO:" << std::endl;
                    std::cout << "  📊 DDOS Probability: " << std::fixed << std::setprecision(2)
                             << (analysis.ddos_probability * 100) << "%" << std::endl;
                    std::cout << "  🏷️ Threat Type: " << analysis.threat_type << std::endl;
                    std::cout << "  🚫 Should Block: " << (analysis.should_block ? "YES" : "NO") << std::endl;

                    displayNetworkFeatures(event.network_features());
                    displayGeoEnrichment(event.geo_enrichment());
                    displayDistributedNode(event.capturing_node());
                    std::cout << std::endl;
                }

                // Estadísticas cada 25 eventos
                if (events_processed % 25 == 0) {
                    auto current_time = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
                    double rate = elapsed > 0 ? events_processed / static_cast<double>(elapsed) : 0;

                    std::cout << "[Service2] 📈 Processed: " << events_processed
                             << " | Suspicious: " << suspicious_events
                             << " | High-Risk: " << high_risk_events
                             << " | Rate: " << std::fixed << std::setprecision(1) << rate << " events/sec" << std::endl;
                }

            } catch (const std::exception& e) {
                std::cerr << "[Service2] Error procesando evento: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[Service2] Error crítico: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "[Service2] 🛑 Shutdown completado" << std::endl;
    return 0;
}