#include <iostream>
#include <memory>
#include <csignal>
#include <atomic>

#include "config_manager.hpp"
#include "compression_handler.hpp"
#include "zmq_pool_manager.hpp"

std::atomic<bool> g_shutdown{false};

void signal_handler(int signal) {
    g_shutdown = true;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);

    std::cout << "eBPF Sniffer v3.1 (minimal build)" << std::endl;

    try {
        // Test compression handler
        auto compression = std::make_unique<sniffer::CompressionHandler>();

        // Test ZMQ pool
        auto zmq_pool = std::make_unique<sniffer::ZMQPoolManager>(2);

        std::cout << "Core components initialized successfully" << std::endl;
        std::cout << "Press Ctrl+C to exit" << std::endl;

        while (!g_shutdown) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}