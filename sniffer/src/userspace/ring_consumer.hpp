#pragma once

#include <functional>
#include <atomic>
#include <thread>
#include <memory>
#include <bpf/libbpf.h>
#include <zmq.hpp>
#include <string>
#include <arpa/inet.h>
#include "config_manager.hpp"
#include "../../protobuf/network_security.pb.h"

namespace sniffer {

// Estructura que coincide con el evento eBPF
struct SimpleEvent {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t protocol;
    uint32_t packet_len;
    uint64_t timestamp;
} __attribute__((packed));

// Callback para procesar eventos
using EventCallback = std::function<void(const SimpleEvent& event)>;

class RingBufferConsumer {
public:
    RingBufferConsumer();
    explicit RingBufferConsumer(const NetworkConfig& net_config);
    ~RingBufferConsumer();

    // Inicializar consumer con file descriptor del ring buffer
    bool initialize(int ring_fd);

    // Inicializar ZeroMQ socket
    bool initialize_zmq(const NetworkConfig& net_config);

    // Establecer callback para procesar eventos
    void set_event_callback(EventCallback callback);

    // Iniciar consumo en thread separado
    bool start();

    // Detener consumo
    void stop();

    // Verificar si está ejecutándose
    bool is_running() const { return running_; }

    // Obtener estadísticas
    uint64_t get_events_processed() const { return events_processed_; }
    uint64_t get_events_dropped() const { return events_dropped_; }
    uint64_t get_events_sent() const { return events_sent_; }

private:
    struct ring_buffer* ring_buf_;
    int ring_fd_;

    std::atomic<bool> running_;
    std::atomic<bool> should_stop_;
    std::thread consumer_thread_;

    EventCallback event_callback_;

    std::atomic<uint64_t> events_processed_;
    std::atomic<uint64_t> events_dropped_;
    std::atomic<uint64_t> events_sent_;

    // ZeroMQ components
    std::unique_ptr<zmq::context_t> zmq_context_;
    std::unique_ptr<zmq::socket_t> zmq_socket_;
    NetworkConfig network_config_;
    bool zmq_enabled_;

    // Pre-allocated IP address buffers for efficiency (hardcoded to avoid INET_ADDRSTRLEN issues)
    char src_ip_buffer_[16];  // INET_ADDRSTRLEN = 16
    char dst_ip_buffer_[16];  // INET_ADDRSTRLEN = 16

    // Thread function para consumo
    void consume_events();

    // Callback estático para libbpf
    static int handle_event(void* ctx, void* data, size_t data_sz);

    // Helper para enviar eventos via ZeroMQ (solo protobuf)
    bool send_event_zmq(const SimpleEvent& event);

    // Helper para convertir SimpleEvent a protobuf completo
    void populate_protobuf_event(const SimpleEvent& event, protobuf::NetworkSecurityEvent& proto_event);

    // Helper para convertir protocolo numérico a string
    std::string protocol_to_string(uint8_t protocol) const;
};

} // namespace sniffer