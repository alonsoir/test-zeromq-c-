// ZMQ Pool Manager for eBPF Sniffer
// Manages ZeroMQ connection pooling and load balancing

#include "zmq_pool_manager.hpp"
#include <zmq.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>
// sniffer/src/userspace/zmq_pool_manager.cpp
namespace sniffer {

ZMQPoolManager::ZMQPoolManager(int pool_size)
    : pool_size_(pool_size), current_socket_(0) {
    context_ = zmq_ctx_new();
    if (!context_) {
        throw std::runtime_error("Failed to create ZMQ context");
    }

    // Initialize socket pool
    sockets_.reserve(pool_size);
    for (int i = 0; i < pool_size; ++i) {
        void* socket = zmq_socket(context_, ZMQ_PUSH);
        if (!socket) {
            throw std::runtime_error("Failed to create ZMQ socket");
        }
        sockets_.push_back(socket);
    }
}

ZMQPoolManager::~ZMQPoolManager() {
    shutdown();
}

bool ZMQPoolManager::connect(const std::string& endpoint) {
    endpoint_ = endpoint;

    for (auto socket : sockets_) {
        if (zmq_connect(socket, endpoint.c_str()) != 0) {
            std::cerr << "Failed to connect socket to " << endpoint << std::endl;
            return false;
        }
    }

    connected_ = true;
    return true;
}

void* ZMQPoolManager::get_socket() {
    if (!connected_ || sockets_.empty()) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(socket_mutex_);
    void* socket = sockets_[current_socket_];
    current_socket_ = (current_socket_ + 1) % sockets_.size();
    return socket;
}

bool ZMQPoolManager::send_message(const void* data, size_t size) {
    void* socket = get_socket();
    if (!socket) {
        return false;
    }

    zmq_msg_t message;
    if (zmq_msg_init_size(&message, size) != 0) {
        return false;
    }

    memcpy(zmq_msg_data(&message), data, size);

    int rc = zmq_msg_send(&message, socket, ZMQ_DONTWAIT);
    zmq_msg_close(&message);

    return rc != -1;
}

void ZMQPoolManager::shutdown() {
    if (!context_) {
        return;
    }

    connected_ = false;

    for (auto socket : sockets_) {
        zmq_close(socket);
    }
    sockets_.clear();

    zmq_ctx_destroy(context_);
    context_ = nullptr;
}

size_t ZMQPoolManager::get_pool_size() const {
    return pool_size_;
}

bool ZMQPoolManager::is_connected() const {
    return connected_;
}

bool ZMQPoolManager::bind(const std::string& endpoint) {
    endpoint_ = endpoint;

    for (auto socket : sockets_) {
        if (zmq_bind(socket, endpoint.c_str()) != 0) {
            std::cerr << "Failed to bind socket to " << endpoint
                      << ": " << zmq_strerror(errno) << std::endl;
            return false;
        }
    }

    connected_ = true;
    return true;
}

} // namespace sniffer