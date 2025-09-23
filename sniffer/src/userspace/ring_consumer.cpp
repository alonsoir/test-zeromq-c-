#include "ring_consumer.hpp"
#include <iostream>
#include <cstring>
#include <arpa/inet.h>

namespace sniffer {

RingBufferConsumer::RingBufferConsumer() 
    : ring_buf_(nullptr), ring_fd_(-1), running_(false), should_stop_(false),
      events_processed_(0), events_dropped_(0) {
}

RingBufferConsumer::~RingBufferConsumer() {
    stop();
    
    if (ring_buf_) {
        ring_buffer__free(ring_buf_);
        std::cout << "[INFO] Ring buffer freed" << std::endl;
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
    
    should_stop_ = false;
    running_ = true;
    
    // Iniciar thread de consumo
    consumer_thread_ = std::thread(&RingBufferConsumer::consume_events, this);
    
    std::cout << "[INFO] Ring buffer consumer started" << std::endl;
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
              << ", dropped: " << events_dropped_ << std::endl;
}

void RingBufferConsumer::consume_events() {
    std::cout << "[INFO] Starting event consumption loop" << std::endl;
    
    while (!should_stop_) {
        // Consumir eventos del ring buffer
        int err = ring_buffer__poll(ring_buf_, 100); // 100ms timeout
        
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
    } else {
        // Default: mostrar evento básico
        char src_ip_str[INET_ADDRSTRLEN];
        char dst_ip_str[INET_ADDRSTRLEN];
        
        struct in_addr src_addr = {.s_addr = event->src_ip};
        struct in_addr dst_addr = {.s_addr = event->dst_ip};
        
        inet_ntop(AF_INET, &src_addr, src_ip_str, INET_ADDRSTRLEN);
        inet_ntop(AF_INET, &dst_addr, dst_ip_str, INET_ADDRSTRLEN);
        
        const char* proto_str = "Unknown";
        if (event->protocol == 6) proto_str = "TCP";
        else if (event->protocol == 17) proto_str = "UDP";
        else if (event->protocol == 1) proto_str = "ICMP";
        
        std::cout << "[PACKET] " << src_ip_str << ":" << event->src_port 
                  << " -> " << dst_ip_str << ":" << event->dst_port 
                  << " (" << proto_str << ", " << event->packet_len << " bytes, "
                  << "ts: " << event->timestamp << ")" << std::endl;
    }
    
    return 0;
}

} // namespace sniffer
