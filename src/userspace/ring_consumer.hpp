#pragma once

#include <functional>
#include <atomic>
#include <thread>
#include <memory>
#include <bpf/libbpf.h>

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
    ~RingBufferConsumer();
    
    // Inicializar consumer con file descriptor del ring buffer
    bool initialize(int ring_fd);
    
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

private:
    struct ring_buffer* ring_buf_;
    int ring_fd_;
    
    std::atomic<bool> running_;
    std::atomic<bool> should_stop_;
    std::thread consumer_thread_;
    
    EventCallback event_callback_;
    
    std::atomic<uint64_t> events_processed_;
    std::atomic<uint64_t> events_dropped_;
    
    // Thread function para consumo
    void consume_events();
    
    // Callback estático para libbpf
    static int handle_event(void* ctx, void* data, size_t data_sz);
};

} // namespace sniffer
