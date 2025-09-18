#include <zmq.hpp>
#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include "protobuf/network_security.pb.h"
#include "main.h"

// Generador de números aleatorios
std::random_device rd;
std::mt19937 gen(rd());

// Función para generar IP aleatoria
std::string generateRandomIP() {
    std::uniform_int_distribution<> dis(1, 254);
    return std::to_string(dis(gen)) + "." +
           std::to_string(dis(gen)) + "." +
           std::to_string(dis(gen)) + "." +
           std::to_string(dis(gen));
}

// Función para generar puerto aleatorio
uint32_t generateRandomPort() {
    std::uniform_int_distribution<> dis(1024, 65535);
    return dis(gen);
}

// Función para generar NetworkFeatures con datos aleatorios coherentes
void populateNetworkFeatures(protobuf::NetworkFeatures* features) {
    // IPs y puertos
    features->set_source_ip(generateRandomIP());
    features->set_destination_ip(generateRandomIP());
    features->set_source_port(generateRandomPort());
    features->set_destination_port(generateRandomPort());

    // Protocolo TCP (6)
    features->set_protocol_number(6);
    features->set_protocol_name("TCP");

    // Timing
    auto now = std::chrono::system_clock::now();
    auto timestamp = google::protobuf::util::TimeUtil::TimeTToTimestamp(
        std::chrono::system_clock::to_time_t(now));
    *features->mutable_flow_start_time() = timestamp;

    auto duration = google::protobuf::util::TimeUtil::MillisecondsToDuration(
        std::uniform_int_distribution<>(100, 5000)(gen));
    *features->mutable_flow_duration() = duration;

    features->set_flow_duration_microseconds(
        std::uniform_int_distribution<uint64_t>(100000, 5000000)(gen));

    // Estadísticas de paquetes (coherentes entre forward/backward)
    uint64_t forward_packets = std::uniform_int_distribution<uint64_t>(10, 1000)(gen);
    uint64_t backward_packets = std::uniform_int_distribution<uint64_t>(5, forward_packets)(gen);

    features->set_total_forward_packets(forward_packets);
    features->set_total_backward_packets(backward_packets);

    // Bytes (coherentes con número de paquetes)
    uint64_t avg_packet_size = std::uniform_int_distribution<uint64_t>(64, 1500)(gen);
    features->set_total_forward_bytes(forward_packets * avg_packet_size);
    features->set_total_backward_bytes(backward_packets * avg_packet_size);

    // Estadísticas de longitud - Forward
    features->set_forward_packet_length_max(1500);
    features->set_forward_packet_length_min(64);
    features->set_forward_packet_length_mean(avg_packet_size);
    features->set_forward_packet_length_std(
        std::uniform_real_distribution<>(50.0, 200.0)(gen));

    // Estadísticas de longitud - Backward
    features->set_backward_packet_length_max(1500);
    features->set_backward_packet_length_min(64);
    features->set_backward_packet_length_mean(avg_packet_size * 0.8); // Algo menor
    features->set_backward_packet_length_std(
        std::uniform_real_distribution<>(40.0, 180.0)(gen));

    // Velocidades y ratios
    double flow_duration_sec = duration.seconds() + (duration.nanos() / 1e9);
    features->set_flow_bytes_per_second(
        (features->total_forward_bytes() + features->total_backward_bytes()) / flow_duration_sec);
    features->set_flow_packets_per_second(
        (forward_packets + backward_packets) / flow_duration_sec);

    // TCP Flags (números realistas)
    features->set_syn_flag_count(std::uniform_int_distribution<>(1, 3)(gen));
    features->set_ack_flag_count(forward_packets + backward_packets); // Mayoría tienen ACK
    features->set_fin_flag_count(std::uniform_int_distribution<>(0, 2)(gen));
    features->set_psh_flag_count(std::uniform_int_distribution<>(1, 10)(gen));
    features->set_rst_flag_count(0); // Normal traffic

    // Features para DDOS (83 features simuladas)
    for (int i = 0; i < 83; i++) {
        features->add_ddos_features(std::uniform_real_distribution<>(0.0, 1.0)(gen));
    }

    std::cout << "📊 Generated NetworkFeatures:" << std::endl;
    std::cout << "   Source: " << features->source_ip() << ":" << features->source_port() << std::endl;
    std::cout << "   Destination: " << features->destination_ip() << ":" << features->destination_port() << std::endl;
    std::cout << "   Protocol: " << features->protocol_name() << std::endl;
    std::cout << "   Forward packets: " << features->total_forward_packets() << std::endl;
    std::cout << "   Backward packets: " << features->total_backward_packets() << std::endl;
    std::cout << "   Forward bytes: " << features->total_forward_bytes() << std::endl;
    std::cout << "   Backward bytes: " << features->total_backward_bytes() << std::endl;
}

// Función para generar información geográfica
void populateGeoEnrichment(protobuf::GeoEnrichment* geo) {
    // Source IP Geo
    auto* source_geo = geo->mutable_source_ip_geo();
    source_geo->set_country_name("Spain");
    source_geo->set_country_code("ES");
    source_geo->set_region_name("Andalusia");
    source_geo->set_city_name("Sevilla");
    source_geo->set_latitude(37.3886);
    source_geo->set_longitude(-5.9823);
    source_geo->set_isp_name("Telefonica");

    // Destination IP Geo
    auto* dest_geo = geo->mutable_destination_ip_geo();
    dest_geo->set_country_name("United States");
    dest_geo->set_country_code("US");
    dest_geo->set_region_name("California");
    dest_geo->set_city_name("San Francisco");
    dest_geo->set_latitude(37.7749);
    dest_geo->set_longitude(-122.4194);
    dest_geo->set_isp_name("Cloudflare");

    // Sniffer Node Geo
    auto* sniffer_geo = geo->mutable_sniffer_node_geo();
    sniffer_geo->set_country_name("Spain");
    sniffer_geo->set_country_code("ES");
    sniffer_geo->set_region_name("Andalusia");
    sniffer_geo->set_city_name("Sevilla");
    sniffer_geo->set_latitude(37.3886);
    sniffer_geo->set_longitude(-5.9823);

    // Análisis geográfico
    geo->set_source_destination_distance_km(9000.5); // Sevilla to SF
    geo->set_source_destination_same_country(false);
    geo->set_distance_category("international");
    geo->set_enrichment_complete(true);

    std::cout << "🌍 Generated GeoEnrichment:" << std::endl;
    std::cout << "   Source: " << source_geo->city_name() << ", " << source_geo->country_name() << std::endl;
    std::cout << "   Destination: " << dest_geo->city_name() << ", " << dest_geo->country_name() << std::endl;
    std::cout << "   Distance: " << geo->source_destination_distance_km() << " km" << std::endl;
}

int main() {
    // Inicializar libprotobuf
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::cout << "🚀 Service1 starting - Protobuf + ZeroMQ Producer" << std::endl;

    // Configurar ZeroMQ
    zmq::context_t context{1};
    zmq::socket_t socket{context, zmq::socket_type::push};
    socket.bind("tcp://*:5555");

    std::cout << "✅ Service1 bound to tcp://*:5555, waiting for consumer..." << std::endl;

    // Dar tiempo para que service2 se conecte
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Crear el mensaje principal
    protobuf::NetworkSecurityEvent event;

    // Generar ID único
    event.set_event_id("evt_" + std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()));

    // Timestamp del evento
    auto now = std::chrono::system_clock::now();
    auto timestamp = google::protobuf::util::TimeUtil::TimeTToTimestamp(
        std::chrono::system_clock::to_time_t(now));
    *event.mutable_event_timestamp() = timestamp;

    event.set_originating_node_id("service1_node");
    event.set_final_classification("BENIGN");
    event.set_threat_category("NORMAL");
    event.set_overall_threat_score(0.05); // Low threat score
    event.set_schema_version(31); // v3.1
    event.set_protobuf_version("3.1.0");

    // Llenar con datos aleatorios coherentes
    populateNetworkFeatures(event.mutable_network_features());
    populateGeoEnrichment(event.mutable_geo_enrichment());

    // Información del nodo capturador
    auto* node = event.mutable_capturing_node();
    node->set_node_id("service1_node");
    node->set_node_hostname("service1_container");
    node->set_node_ip_address("172.18.0.2"); // IP típica de Docker
    node->set_physical_location("Sevilla, Spain");
    node->set_node_role(protobuf::DistributedNode::PACKET_SNIFFER);
    node->set_node_status(protobuf::DistributedNode::ACTIVE);

    std::cout << "\n🎯 Main Event Details:" << std::endl;
    std::cout << "   Event ID: " << event.event_id() << std::endl;
    std::cout << "   Classification: " << event.final_classification() << std::endl;
    std::cout << "   Threat Score: " << event.overall_threat_score() << std::endl;
    std::cout << "   Schema Version: " << event.schema_version() << std::endl;

    // Serializar el mensaje
    std::string serialized_data;
    if (!event.SerializeToString(&serialized_data)) {
        std::cerr << "❌ Error: Failed to serialize protobuf message" << std::endl;
        return 1;
    }

    // Enviar vía ZeroMQ
    zmq::message_t zmq_msg(serialized_data.data(), serialized_data.size());
    auto result = socket.send(zmq_msg, zmq::send_flags::none);

    if (result) {
        std::cout << "\n✅ Successfully sent NetworkSecurityEvent (" << serialized_data.size() << " bytes)" << std::endl;
        std::cout << "📤 Message serialized and sent via ZeroMQ to service2" << std::endl;
    } else {
        std::cerr << "❌ Error: Failed to send message via ZeroMQ" << std::endl;
        return 1;
    }

    // Cleanup
    google::protobuf::ShutdownProtobufLibrary();

    std::cout << "🔚 Service1 finished successfully" << std::endl;
    return 0;
}