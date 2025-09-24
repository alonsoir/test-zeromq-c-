#include "ring_consumer.hpp"
#include <iostream>
#include <cstring>
#include <arpa/inet.h>
#include <google/protobuf/timestamp.pb.h>

namespace sniffer {

RingBufferConsumer::RingBufferConsumer()
    : ring_buf_(nullptr), ring_fd_(-1), running_(false), should_stop_(false),
      events_processed_(0), events_dropped_(0), events_sent_(0),
      zmq_enabled_(false) {
}

RingBufferConsumer::RingBufferConsumer(const NetworkConfig& net_config)
    : ring_buf_(nullptr), ring_fd_(-1), running_(false), should_stop_(false),
      events_processed_(0), events_dropped_(0), events_sent_(0),
      network_config_(net_config), zmq_enabled_(false) {

    // Inicializar ZeroMQ
    initialize_zmq(net_config);
}

RingBufferConsumer::~RingBufferConsumer() {
    stop();

    if (ring_buf_) {
        ring_buffer__free(ring_buf_);
        std::cout << "[INFO] Ring buffer freed" << std::endl;
    }

    // Cleanup ZeroMQ
    if (zmq_socket_) {
        zmq_socket_->close();
    }
}

bool RingBufferConsumer::initialize(int ring_fd) {
    if (ring_fd < 0) {
        std::cerr << "[ERROR] Invalid ring buffer file descriptor" << std::endl;
        return false;
    }

    ring_fd_ = ring_fd;

    // Crear ring buffer consumer
    ring_buf_ = ring_buffer__new(ring_fd_, handle_event, this, nullptr);
    if (!ring_buf_) {
        std::cerr << "[ERROR] Failed to create ring buffer consumer" << std::endl;
        return false;
    }

    std::cout << "[INFO] Ring buffer consumer initialized with FD: " << ring_fd_ << std::endl;
    return true;
}

bool RingBufferConsumer::initialize_zmq(const NetworkConfig& net_config) {
    try {
        network_config_ = net_config;

        // Crear contexto ZeroMQ
        zmq_context_ = std::make_unique<zmq::context_t>(1);

        // Crear socket PUSH
        if (net_config.output_socket.socket_type == "PUSH") {
            zmq_socket_ = std::make_unique<zmq::socket_t>(*zmq_context_, ZMQ_PUSH);
        } else {
            std::cerr << "[ERROR] Unsupported socket type: " << net_config.output_socket.socket_type << std::endl;
            return false;
        }

        // Configurar High Water Mark para máximo rendimiento
        zmq_socket_->set(zmq::sockopt::sndhwm, 10000);

        // Configurar socket para máximo rendimiento
        zmq_socket_->set(zmq::sockopt::sndbuf, 1024 * 1024);  // 1MB send buffer
        zmq_socket_->set(zmq::sockopt::linger, 0);            // No linger on close

        // Crear endpoint
        std::string endpoint = "tcp://" + net_config.output_socket.address + ":" +
                              std::to_string(net_config.output_socket.port);

        // Siempre bind para socket PUSH
        zmq_socket_->bind(endpoint);
        std::cout << "[INFO] ZeroMQ socket bound to: " << endpoint << std::endl;

        zmq_enabled_ = true;
        std::cout << "[INFO] ZeroMQ initialized for high-performance protobuf transport" << std::endl;
        return true;

    } catch (const zmq::error_t& e) {
        std::cerr << "[ERROR] ZeroMQ initialization failed: " << e.what() << std::endl;
        zmq_enabled_ = false;
        return false;
    }
}

void RingBufferConsumer::set_event_callback(EventCallback callback) {
    event_callback_ = callback;
}

bool RingBufferConsumer::start() {
    if (running_) {
        std::cout << "[INFO] Ring buffer consumer already running" << std::endl;
        return true;
    }

    if (!ring_buf_) {
        std::cerr << "[ERROR] Ring buffer not initialized. Call initialize() first" << std::endl;
        return false;
    }

    if (!zmq_enabled_) {
        std::cerr << "[ERROR] ZeroMQ not initialized. High-performance protobuf transport required" << std::endl;
        return false;
    }

    should_stop_ = false;
    running_ = true;

    // Iniciar thread de consumo
    consumer_thread_ = std::thread(&RingBufferConsumer::consume_events, this);

    std::cout << "[INFO] High-performance ring buffer consumer started" << std::endl;
    std::cout << "[INFO] Protobuf transport via ZeroMQ enabled" << std::endl;
    return true;
}

void RingBufferConsumer::stop() {
    if (!running_) {
        return;
    }

    should_stop_ = true;

    if (consumer_thread_.joinable()) {
        consumer_thread_.join();
    }

    running_ = false;
    std::cout << "[INFO] Ring buffer consumer stopped" << std::endl;
    std::cout << "[INFO] Events processed: " << events_processed_
              << ", sent: " << events_sent_
              << ", dropped: " << events_dropped_ << std::endl;
}

void RingBufferConsumer::consume_events() {
    std::cout << "[INFO] Starting high-performance event consumption loop" << std::endl;

    while (!should_stop_) {
        // Consumir eventos del ring buffer con timeout corto para máximo rendimiento
        int err = ring_buffer__poll(ring_buf_, 10); // 10ms timeout

        if (err < 0 && err != -EINTR) {
            std::cerr << "[ERROR] Ring buffer poll error: " << strerror(-err) << std::endl;
            events_dropped_++;
            break;
        }
    }

    std::cout << "[INFO] Event consumption loop stopped" << std::endl;
}

int RingBufferConsumer::handle_event(void* ctx, void* data, size_t data_sz) {
    RingBufferConsumer* consumer = static_cast<RingBufferConsumer*>(ctx);

    if (data_sz != sizeof(SimpleEvent)) {
        std::cerr << "[WARNING] Unexpected event size: " << data_sz
                  << ", expected: " << sizeof(SimpleEvent) << std::endl;
        consumer->events_dropped_++;
        return 0;
    }

    const SimpleEvent* event = static_cast<const SimpleEvent*>(data);
    consumer->events_processed_++;

    // Llamar callback si está configurado
    if (consumer->event_callback_) {
        consumer->event_callback_(*event);
    }

    // Enviar via ZeroMQ (solo protobuf, no fallback)
    if (consumer->send_event_zmq(*event)) {
        consumer->events_sent_++;
    } else {
        consumer->events_dropped_++;
    }

    return 0;
}

std::string RingBufferConsumer::protocol_to_string(uint8_t protocol) const {
    switch (protocol) {
        case 1: return "ICMP";
        case 6: return "TCP";
        case 17: return "UDP";
        case 47: return "GRE";
        case 50: return "ESP";
        case 51: return "AH";
        case 58: return "ICMPv6";
        default: return "UNKNOWN";
    }
}

void RingBufferConsumer::populate_protobuf_event(const SimpleEvent& event, protobuf::NetworkSecurityEvent& proto_event) {
    // Convertir IPs a strings de manera eficiente usando buffers pre-allocados
    struct in_addr src_addr = {.s_addr = event.src_ip};
    struct in_addr dst_addr = {.s_addr = event.dst_ip};

    inet_ntop(AF_INET, &src_addr, src_ip_buffer_, 16);
    inet_ntop(AF_INET, &dst_addr, dst_ip_buffer_, 16);

    // Generar event ID único basado en timestamp y hash básico
    std::string event_id = std::to_string(event.timestamp) + "_" +
                          std::to_string(event.src_ip ^ event.dst_ip);

    // Poblar campos básicos del NetworkSecurityEvent
    proto_event.set_event_id(event_id);
    proto_event.set_originating_node_id("sniffer_node_001");  // TODO: obtener de config
    proto_event.set_correlation_id(event_id);  // Simple correlation por ahora
    proto_event.set_schema_version(31);  // v3.1
    proto_event.set_protobuf_version("3.1.0");

    // Timestamp del evento
    google::protobuf::Timestamp* timestamp = proto_event.mutable_event_timestamp();
    timestamp->set_seconds(event.timestamp / 1000000000ULL);
    timestamp->set_nanos(event.timestamp % 1000000000ULL);

    // Poblar NetworkFeatures con los datos básicos del sniffer
    protobuf::NetworkFeatures* features = proto_event.mutable_network_features();
    features->set_source_ip(src_ip_buffer_);
    features->set_destination_ip(dst_ip_buffer_);
    features->set_source_port(event.src_port);
    features->set_destination_port(event.dst_port);
    features->set_protocol_number(event.protocol);
    features->set_protocol_name(protocol_to_string(event.protocol));

    // Features básicas que podemos calcular del evento simple
    features->set_total_forward_packets(1);  // Asumimos 1 paquete por evento
    features->set_total_backward_packets(0); // No tenemos info backward en el sniffer
    features->set_total_forward_bytes(event.packet_len);
    features->set_total_backward_bytes(0);
    features->set_minimum_packet_length(event.packet_len);
    features->set_maximum_packet_length(event.packet_len);
    features->set_packet_length_mean(static_cast<double>(event.packet_len));
    features->set_average_packet_size(static_cast<double>(event.packet_len));

    // Timestamp del flow
    google::protobuf::Timestamp* flow_start = features->mutable_flow_start_time();
    flow_start->set_seconds(event.timestamp / 1000000000ULL);
    flow_start->set_nanos(event.timestamp % 1000000000ULL);

    // TimeWindow básico
    protobuf::TimeWindow* time_window = proto_event.mutable_time_window();
    google::protobuf::Timestamp* window_start = time_window->mutable_window_start();
    window_start->set_seconds(event.timestamp / 1000000000ULL);
    window_start->set_nanos(event.timestamp % 1000000000ULL);
    time_window->set_window_type(protobuf::TimeWindow::SLIDING);
    time_window->set_sequence_number(static_cast<uint64_t>(event.timestamp));

    // DistributedNode info básica
    protobuf::DistributedNode* node = proto_event.mutable_capturing_node();
    node->set_node_id("sniffer_node_001");
    node->set_node_hostname("ebpf_sniffer");
    node->set_node_role(protobuf::DistributedNode::PACKET_SNIFFER);
    node->set_node_status(protobuf::DistributedNode::ACTIVE);
    node->set_agent_version("3.1.0");

    // Timestamp de captura
    google::protobuf::Timestamp* heartbeat = node->mutable_last_heartbeat();
    heartbeat->set_seconds(event.timestamp / 1000000000ULL);
    heartbeat->set_nanos(event.timestamp % 1000000000ULL);

    // Clasificación básica (el sniffer no hace ML, solo captura)
    proto_event.set_overall_threat_score(0.0);  // Sin análisis ML en el sniffer
    proto_event.set_final_classification("UNCATEGORIZED");
    proto_event.set_threat_category("RAW_CAPTURE");

    // Agregar tag para indicar que es captura raw
    proto_event.add_event_tags("raw_ebpf_capture");
    proto_event.add_event_tags("kernel_space_origin");
    proto_event.add_event_tags("requires_processing");
}

bool RingBufferConsumer::send_event_zmq(const SimpleEvent& event) {
    try {
        // Crear y poblar mensaje protobuf completo
        protobuf::NetworkSecurityEvent proto_event;
        populate_protobuf_event(event, proto_event);

        // Serializar a string
        std::string serialized_data;
        if (!proto_event.SerializeToString(&serialized_data)) {
            std::cerr << "[ERROR] Failed to serialize protobuf message" << std::endl;
            return false;
        }

        // Enviar via ZeroMQ con dontwait para máximo rendimiento
        zmq::message_t message(serialized_data.size());
        memcpy(message.data(), serialized_data.c_str(), serialized_data.size());

        auto result = zmq_socket_->send(message, zmq::send_flags::dontwait);
        return result.has_value();

    } catch (const zmq::error_t& e) {
        if (e.num() != EAGAIN) {  // EAGAIN es esperado con dontwait cuando buffer está lleno
            std::cerr << "[ERROR] ZeroMQ send failed: " << e.what() << std::endl;
        }
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Protobuf serialization failed: " << e.what() << std::endl;
        return false;
    }
}

} // namespace sniffer