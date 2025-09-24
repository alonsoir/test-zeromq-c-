// service1/main_simple_fixed.cpp - CORREGIDO con includes correctos
#include "service1_main.h"  // No "main.h", usar nombre específico
#include "EtcdServiceRegistry.h"  // Sin "../common/" - mismo directorio en Docker
#include <zmq.hpp>
#include <protobuf/network_security.pb.h>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>
#include <cstdlib>
#include <json/json.h>

std::atomic<bool> g_running(true);

void signalHandler(int signum) {
    std::cout << "\n[Service1] Señal recibida: " << signum << ". Iniciando shutdown..." << std::endl;
    g_running = false;
}

std::string createServiceConfig() {
    Json::Value config;

    // Configuración básica del servicio
    config["service_type"] = "packet_sniffer";
    config["version"] = "1.0.0";
    config["capabilities"] = Json::Value(Json::arrayValue);
    config["capabilities"].append("ddos_detection");
    config["capabilities"].append("packet_analysis");
    config["capabilities"].append("feature_extraction");

    // Configuración ZeroMQ
    config["zeromq"]["bind_address"] = "tcp://*:5555";
    config["zeromq"]["socket_type"] = "PUSH";
    config["zeromq"]["hwm"] = 10000;

    // Configuración de rendimiento
    config["performance"]["max_packets_per_second"] = 100000;
    config["performance"]["batch_size"] = 100;
    config["performance"]["thread_pool_size"] = 4;

    // Configuración de detección DDOS
    config["ddos_detection"]["threshold_pps"] = 10000;
    config["ddos_detection"]["suspicious_ratio"] = 0.8;
    config["ddos_detection"]["analysis_window_seconds"] = 60;

    // Features que extrae (83 features como mencionaste)
    config["feature_extraction"]["features_count"] = 83;
    config["feature_extraction"]["include_geo_enrichment"] = true;
    config["feature_extraction"]["include_flow_stats"] = true;

    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, config);
}

// Función para generar datos de prueba
protobuf::NetworkSecurityEvent generateNetworkEvent() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> ip_dist(1, 254);
    static std::uniform_int_distribution<> port_dist(1024, 65535);
    static std::uniform_int_distribution<> packet_size_dist(64, 1500);
    static std::uniform_real_distribution<> suspicious_dist(0.0, 1.0);
    static std::uniform_int_distribution<> flow_dist(1, 10000);

    protobuf::NetworkSecurityEvent event;

    // Identificación única del evento
    event.set_event_id("evt_" + std::to_string(std::time(nullptr)) + "_" + std::to_string(rd()));

    // Timestamp usando protobuf timestamp
    auto* timestamp = event.mutable_event_timestamp();
    auto now = std::chrono::system_clock::now();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch());
    timestamp->set_seconds(seconds.count());

    std::string node_id = std::getenv("NODE_ID") ? std::getenv("NODE_ID") : "service1_node_001";
    event.set_originating_node_id(node_id);

    // Configurar NetworkFeatures
    auto* features = event.mutable_network_features();

    // IPs de origen y destino
    std::string source_ip = std::to_string(ip_dist(gen)) + "." +
                           std::to_string(ip_dist(gen)) + "." +
                           std::to_string(ip_dist(gen)) + "." +
                           std::to_string(ip_dist(gen));
    std::string destination_ip = "192.168.1." + std::to_string(ip_dist(gen));

    features->set_source_ip(source_ip);
    features->set_destination_ip(destination_ip);
    features->set_source_port(port_dist(gen));
    features->set_destination_port(80);
    features->set_protocol_number(6); // TCP
    features->set_protocol_name("TCP");

    // Timing con protobuf Duration
    auto* flow_start = features->mutable_flow_start_time();
    flow_start->set_seconds(seconds.count());

    auto* duration = features->mutable_flow_duration();
    uint64_t duration_ms = static_cast<uint64_t>(suspicious_dist(gen) * 1000);
    duration->set_seconds(duration_ms / 1000);
    duration->set_nanos((duration_ms % 1000) * 1000000);
    features->set_flow_duration_microseconds(duration_ms * 1000);

    // Estadísticas de flujo
    uint64_t forward_packets = flow_dist(gen);
    uint64_t backward_packets = forward_packets * (suspicious_dist(gen) > 0.7 ? 0.1 : 0.8);

    features->set_total_forward_packets(forward_packets);
    features->set_total_backward_packets(backward_packets);
    features->set_total_forward_bytes(forward_packets * packet_size_dist(gen));
    features->set_total_backward_bytes(backward_packets * packet_size_dist(gen));

    // Velocidades y ratios
    features->set_flow_bytes_per_second(suspicious_dist(gen) * 100000);
    features->set_flow_packets_per_second(suspicious_dist(gen) * 1000);
    features->set_forward_packets_per_second(suspicious_dist(gen) > 0.8 ? 10000 : 100);
    features->set_backward_packets_per_second(forward_packets * 0.1);

    // TCP Flags (simulados)
    features->set_syn_flag_count(suspicious_dist(gen) > 0.9 ? 1000 : 1);
    features->set_ack_flag_count(forward_packets);
    features->set_fin_flag_count(suspicious_dist(gen) > 0.5 ? 1 : 0);
    features->set_rst_flag_count(suspicious_dist(gen) > 0.8 ? 10 : 0);

    // 83 features DDOS específicas
    for (int i = 0; i < 83; ++i) {
        features->add_ddos_features(suspicious_dist(gen));
    }

    // Configurar GeoEnrichment
    auto* geo = event.mutable_geo_enrichment();

    // Source IP geo
    auto* source_geo = geo->mutable_source_ip_geo();
    if (suspicious_dist(gen) > 0.9) {
        source_geo->set_country_name("China");
        source_geo->set_country_code("CN");
        source_geo->set_city_name("Beijing");
        source_geo->set_threat_level(protobuf::GeoLocationInfo::HIGH);
    } else {
        source_geo->set_country_name("United States");
        source_geo->set_country_code("US");
        source_geo->set_city_name("New York");
        source_geo->set_threat_level(protobuf::GeoLocationInfo::LOW);
    }

    source_geo->set_is_tor_exit_node(suspicious_dist(gen) > 0.95);
    source_geo->set_is_known_malicious(suspicious_dist(gen) > 0.98);

    // Destination IP geo
    auto* dest_geo = geo->mutable_destination_ip_geo();
    dest_geo->set_country_name("Spain");
    dest_geo->set_country_code("ES");
    dest_geo->set_city_name("Madrid");
    dest_geo->set_threat_level(protobuf::GeoLocationInfo::LOW);

    // Análisis geográfico
    geo->set_source_destination_distance_km(suspicious_dist(gen) > 0.8 ? 8000 : 500);
    geo->set_source_destination_same_country(suspicious_dist(gen) > 0.8 ? false : true);
    geo->set_distance_category(suspicious_dist(gen) > 0.8 ? "international" : "national");
    geo->set_enrichment_complete(true);
    geo->set_source_ip_enriched(true);
    geo->set_destination_ip_enriched(true);

    // Configurar Node info
    auto* capturing_node = event.mutable_capturing_node();
    capturing_node->set_node_id(node_id);
    capturing_node->set_node_hostname("service1-container");
    capturing_node->set_node_role(protobuf::DistributedNode::PACKET_SNIFFER);
    capturing_node->set_node_status(protobuf::DistributedNode::ACTIVE);
    capturing_node->set_agent_version("1.0.0");

    auto* last_heartbeat = capturing_node->mutable_last_heartbeat();
    last_heartbeat->set_seconds(seconds.count());

    // Scoring final
    double threat_score = suspicious_dist(gen);
    event.set_overall_threat_score(threat_score);

    if (threat_score > 0.9) {
        event.set_final_classification("MALICIOUS");
        event.set_threat_category("DDOS");
    } else if (threat_score > 0.6) {
        event.set_final_classification("SUSPICIOUS");
        event.set_threat_category("POTENTIAL_ATTACK");
    } else {
        event.set_final_classification("BENIGN");
        event.set_threat_category("NORMAL");
    }

    event.set_schema_version(31);
    event.set_protobuf_version("3.1.0");

    return event;
}

int main() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    std::cout << "[Service1] Iniciando Packet Sniffer con integración etcd..." << std::endl;

    try {
        std::string node_id = std::getenv("NODE_ID") ? std::getenv("NODE_ID") : "service1_node_001";
        std::string service_name = std::getenv("SERVICE_NAME") ? std::getenv("SERVICE_NAME") : "packet_sniffer";
        std::string etcd_endpoint = std::getenv("ETCD_ENDPOINTS") ? std::getenv("ETCD_ENDPOINTS") : "http://etcd:2379";
        int zmq_port = std::getenv("ZMQ_BIND_PORT") ? std::stoi(std::getenv("ZMQ_BIND_PORT")) : 5555;

        std::cout << "[Service1] Configuración:" << std::endl;
        std::cout << "  - Node ID: " << node_id << std::endl;
        std::cout << "  - Service Name: " << service_name << std::endl;
        std::cout << "  - etcd Endpoint: " << etcd_endpoint << std::endl;
        std::cout << "  - ZMQ Port: " << zmq_port << std::endl;

        // Registrarse en etcd
        auto& etcd_registry = EtcdServiceRegistry::getInstance(etcd_endpoint);

        std::string service_config = createServiceConfig();

        if (!etcd_registry.registerService(service_name, node_id, service_config, 30)) {
            std::cerr << "[Service1] ERROR: No se pudo registrar en etcd" << std::endl;
            return 1;
        }

        std::cout << "[Service1] ✅ Registrado exitosamente en etcd" << std::endl;

        // Configurar ZeroMQ
        zmq::context_t context(1);
        zmq::socket_t socket(context, ZMQ_PUSH);

        std::string bind_address = "tcp://*:" + std::to_string(zmq_port);
        socket.bind(bind_address);

        std::cout << "[Service1] ✅ ZeroMQ socket enlazado en: " << bind_address << std::endl;
        std::cout << "[Service1] 🚀 Iniciando generación de eventos..." << std::endl;

        int events_sent = 0;
        auto start_time = std::chrono::steady_clock::now();

        while (g_running) {
            try {
                auto network_event = generateNetworkEvent();

                std::string serialized_data;
                if (!network_event.SerializeToString(&serialized_data)) {
                    std::cerr << "[Service1] Error serializando evento" << std::endl;
                    continue;
                }

                zmq::message_t message(serialized_data.size());
                memcpy(message.data(), serialized_data.c_str(), serialized_data.size());

                socket.send(message, zmq::send_flags::dontwait);
                events_sent++;

                if (events_sent % 100 == 0) {
                    auto current_time = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
                    double rate = elapsed > 0 ? events_sent / static_cast<double>(elapsed) : 0;

                    std::cout << "[Service1] 📊 Eventos enviados: " << events_sent
                             << " | Rate: " << std::fixed << std::setprecision(1) << rate << " events/sec" << std::endl;

                    // Actualizar estadísticas en etcd
                    Json::Value stats;
                    stats["events_sent"] = events_sent;
                    stats["events_per_second"] = rate;
                    stats["uptime_seconds"] = static_cast<int>(elapsed);
                    stats["last_update"] = std::time(nullptr);

                    Json::StreamWriterBuilder builder;
                    std::string stats_json = Json::writeString(builder, stats);

                    etcd_registry.updateServiceConfig(service_name, stats_json, "stats");
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(20));

            } catch (const std::exception& e) {
                std::cerr << "[Service1] Error en loop principal: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[Service1] Error crítico: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "[Service1] 🛑 Shutdown completado" << std::endl;
    return 0;
}