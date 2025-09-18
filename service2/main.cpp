#include <zmq.hpp>
#include <iostream>
#include <string>
#include <iomanip>
#include "protobuf/network_security.pb.h"
#include "main.h"

// Función para mostrar información de NetworkFeatures
void displayNetworkFeatures(const protobuf::NetworkFeatures& features) {
    std::cout << "\n📊 NETWORK FEATURES ANALYSIS" << std::endl;
    std::cout << "═══════════════════════════════════════════════════" << std::endl;

    // Información básica del flujo
    std::cout << "🔍 Flow Identification:" << std::endl;
    std::cout << "   Source IP:Port      → " << features.source_ip() << ":" << features.source_port() << std::endl;
    std::cout << "   Destination IP:Port → " << features.destination_ip() << ":" << features.destination_port() << std::endl;
    std::cout << "   Protocol           → " << features.protocol_name() << " (" << features.protocol_number() << ")" << std::endl;

    // Estadísticas de paquetes
    std::cout << "\n📦 Packet Statistics:" << std::endl;
    std::cout << "   Forward Packets  → " << features.total_forward_packets() << std::endl;
    std::cout << "   Backward Packets → " << features.total_backward_packets() << std::endl;
    std::cout << "   Forward Bytes    → " << features.total_forward_bytes() << " bytes" << std::endl;
    std::cout << "   Backward Bytes   → " << features.total_backward_bytes() << " bytes" << std::endl;
    std::cout << "   Total Data       → " << (features.total_forward_bytes() + features.total_backward_bytes()) << " bytes" << std::endl;

    // Estadísticas de longitud
    std::cout << "\n📏 Packet Length Statistics:" << std::endl;
    std::cout << "   Forward  → Min:" << features.forward_packet_length_min()
              << " Max:" << features.forward_packet_length_max()
              << " Mean:" << std::fixed << std::setprecision(2) << features.forward_packet_length_mean() << std::endl;
    std::cout << "   Backward → Min:" << features.backward_packet_length_min()
              << " Max:" << features.backward_packet_length_max()
              << " Mean:" << std::fixed << std::setprecision(2) << features.backward_packet_length_mean() << std::endl;

    // TCP Flags
    if (features.protocol_name() == "TCP") {
        std::cout << "\n🏳️  TCP Flags Analysis:" << std::endl;
        std::cout << "   SYN: " << features.syn_flag_count() << std::endl;
        std::cout << "   ACK: " << features.ack_flag_count() << std::endl;
        std::cout << "   FIN: " << features.fin_flag_count() << std::endl;
        std::cout << "   PSH: " << features.psh_flag_count() << std::endl;
        std::cout << "   RST: " << features.rst_flag_count() << std::endl;
    }

    // Velocidades
    std::cout << "\n🚀 Flow Speeds:" << std::endl;
    std::cout << "   Bytes/sec    → " << std::fixed << std::setprecision(2) << features.flow_bytes_per_second() << std::endl;
    std::cout << "   Packets/sec  → " << std::fixed << std::setprecision(2) << features.flow_packets_per_second() << std::endl;

    // Features ML
    if (features.ddos_features_size() > 0) {
        std::cout << "\n🧠 ML Features:" << std::endl;
        std::cout << "   DDOS Features → " << features.ddos_features_size() << " features extracted" << std::endl;
        std::cout << "   Sample values → [";
        for (int i = 0; i < std::min(5, features.ddos_features_size()); i++) {
            std::cout << std::fixed << std::setprecision(3) << features.ddos_features(i);
            if (i < 4 && i < features.ddos_features_size() - 1) std::cout << ", ";
        }
        std::cout << "...]" << std::endl;
    }
}

// Función para mostrar información geográfica
void displayGeoEnrichment(const protobuf::GeoEnrichment& geo) {
    std::cout << "\n🌍 GEOGRAPHICAL ENRICHMENT" << std::endl;
    std::cout << "═══════════════════════════════════════════════════" << std::endl;

    // Source IP Geography
    if (geo.has_source_ip_geo()) {
        const auto& source = geo.source_ip_geo();
        std::cout << "📤 Source Location:" << std::endl;
        std::cout << "   Country → " << source.country_name() << " (" << source.country_code() << ")" << std::endl;
        std::cout << "   City    → " << source.city_name() << ", " << source.region_name() << std::endl;
        std::cout << "   ISP     → " << source.isp_name() << std::endl;
        std::cout << "   Coords  → " << std::fixed << std::setprecision(4)
                  << source.latitude() << ", " << source.longitude() << std::endl;
    }

    // Destination IP Geography
    if (geo.has_destination_ip_geo()) {
        const auto& dest = geo.destination_ip_geo();
        std::cout << "\n📥 Destination Location:" << std::endl;
        std::cout << "   Country → " << dest.country_name() << " (" << dest.country_code() << ")" << std::endl;
        std::cout << "   City    → " << dest.city_name() << ", " << dest.region_name() << std::endl;
        std::cout << "   ISP     → " << dest.isp_name() << std::endl;
        std::cout << "   Coords  → " << std::fixed << std::setprecision(4)
                  << dest.latitude() << ", " << dest.longitude() << std::endl;
    }

    // Geographic Analysis
    std::cout << "\n📏 Geographic Analysis:" << std::endl;
    std::cout << "   Distance        → " << std::fixed << std::setprecision(1) << geo.source_destination_distance_km() << " km" << std::endl;
    std::cout << "   Same Country    → " << (geo.source_destination_same_country() ? "Yes" : "No") << std::endl;
    std::cout << "   Category        → " << geo.distance_category() << std::endl;
    std::cout << "   Enriched        → " << (geo.enrichment_complete() ? "✅ Complete" : "❌ Incomplete") << std::endl;
}

// Función para mostrar información del nodo distribuido
void displayDistributedNode(const protobuf::DistributedNode& node) {
    std::cout << "\n🌐 DISTRIBUTED NODE INFORMATION" << std::endl;
    std::cout << "═══════════════════════════════════════════════════" << std::endl;

    std::cout << "🖥️  Node Details:" << std::endl;
    std::cout << "   Node ID      → " << node.node_id() << std::endl;
    std::cout << "   Hostname     → " << node.node_hostname() << std::endl;
    std::cout << "   IP Address   → " << node.node_ip_address() << std::endl;
    std::cout << "   Location     → " << node.physical_location() << std::endl;

    // Role mapping
    std::string role_name;
    switch (node.node_role()) {
        case protobuf::DistributedNode::PACKET_SNIFFER: role_name = "Packet Sniffer"; break;
        case protobuf::DistributedNode::FEATURE_PROCESSOR: role_name = "Feature Processor"; break;
        case protobuf::DistributedNode::GEOIP_ENRICHER: role_name = "GeoIP Enricher"; break;
        case protobuf::DistributedNode::ML_ANALYZER: role_name = "ML Analyzer"; break;
        case protobuf::DistributedNode::THREAT_DETECTOR: role_name = "Threat Detector"; break;
        default: role_name = "Unknown"; break;
    }

    // Status mapping
    std::string status_name, status_emoji;
    switch (node.node_status()) {
        case protobuf::DistributedNode::ACTIVE:
            status_name = "Active"; status_emoji = "✅"; break;
        case protobuf::DistributedNode::STARTING:
            status_name = "Starting"; status_emoji = "🔄"; break;
        case protobuf::DistributedNode::ERROR:
            status_name = "Error"; status_emoji = "❌"; break;
        default: status_name = "Unknown"; status_emoji = "❓"; break;
    }

    std::cout << "   Role         → " << role_name << std::endl;
    std::cout << "   Status       → " << status_emoji << " " << status_name << std::endl;
}

int main() {
    // Inicializar libprotobuf
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::cout << "🎯 Service2 starting - Protobuf + ZeroMQ Consumer" << std::endl;

    // Configurar ZeroMQ
    zmq::context_t context{1};
    zmq::socket_t socket{context, zmq::socket_type::pull};
    socket.connect("tcp://service1:5555");

    std::cout << "✅ Service2 connected to tcp://service1:5555, waiting for messages..." << std::endl;

    try {
        while (true) {
            // Recibir mensaje
            zmq::message_t message;
            auto result = socket.recv(message, zmq::recv_flags::none);

            if (!result) {
                std::cerr << "❌ Error: Failed to receive message" << std::endl;
                continue;
            }

            std::cout << "\n📥 Received message (" << message.size() << " bytes)" << std::endl;

            // Deserializar el mensaje protobuf
            protobuf::NetworkSecurityEvent event;
            std::string received_data(static_cast<char*>(message.data()), message.size());

            if (!event.ParseFromString(received_data)) {
                std::cerr << "❌ Error: Failed to parse protobuf message" << std::endl;
                continue;
            }

            std::cout << "✅ Successfully parsed NetworkSecurityEvent protobuf message" << std::endl;

            // Mostrar información principal del evento
            std::cout << "\n🎯 MAIN EVENT INFORMATION" << std::endl;
            std::cout << "═══════════════════════════════════════════════════" << std::endl;
            std::cout << "🆔 Event Details:" << std::endl;
            std::cout << "   Event ID         → " << event.event_id() << std::endl;
            std::cout << "   Originating Node → " << event.originating_node_id() << std::endl;
            std::cout << "   Classification   → " << event.final_classification() << std::endl;
            std::cout << "   Threat Category  → " << event.threat_category() << std::endl;
            std::cout << "   Threat Score     → " << std::fixed << std::setprecision(3) << event.overall_threat_score() << std::endl;
            std::cout << "   Schema Version   → v" << (event.schema_version() / 10.0) << std::endl;
            std::cout << "   Protobuf Version → " << event.protobuf_version() << std::endl;

            // Mostrar timestamp del evento
            if (event.has_event_timestamp()) {
                auto timestamp = event.event_timestamp();
                std::time_t time = timestamp.seconds();
                std::cout << "   Timestamp        → " << std::ctime(&time);
            }

            // Mostrar NetworkFeatures si existe
            if (event.has_network_features()) {
                displayNetworkFeatures(event.network_features());
            }

            // Mostrar GeoEnrichment si existe
            if (event.has_geo_enrichment()) {
                displayGeoEnrichment(event.geo_enrichment());
            }

            // Mostrar información del nodo capturador si existe
            if (event.has_capturing_node()) {
                displayDistributedNode(event.capturing_node());
            }

            std::cout << "\n🔍 MESSAGE PROCESSING COMPLETE" << std::endl;
            std::cout << "═══════════════════════════════════════════════════" << std::endl;
            std::cout << "✅ Successfully processed NetworkSecurityEvent" << std::endl;
            std::cout << "📊 Total message size: " << message.size() << " bytes" << std::endl;
            std::cout << "🎯 All protobuf fields parsed and displayed" << std::endl;

            // Para POC, procesamos un mensaje y salimos
            std::cout << "\n🔚 Service2 finished successfully (POC mode - processing one message)" << std::endl;
            break;
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ Exception caught: " << e.what() << std::endl;
        google::protobuf::ShutdownProtobufLibrary();
        return 1;
    }

    // Cleanup
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}