// ADR-029 Variant B — PcapBackend implementation (stub)
// DEBT-VARIANT-B-PCAP-IMPL-001: pcap_loop → queue → ring_consumer pendiente
// DAY 137 — 2026-04-30
#include "pcap_backend.hpp"
#include <iostream>
#include <unistd.h>

namespace sniffer {

PcapBackend::PcapBackend() {
    pipe_fd_[0] = pipe_fd_[1] = -1;
}

PcapBackend::~PcapBackend() {
    close();
}

bool PcapBackend::open(const std::string& interface,
                        PacketCallback cb, void* ctx) {
    cb_  = cb;
    ctx_ = ctx;
    char errbuf[PCAP_ERRBUF_SIZE];
    handle_ = pcap_open_live(interface.c_str(), 65535, 1, 1000, errbuf);
    if (!handle_) {
        std::cerr << "❌ [pcap] pcap_open_live: " << errbuf << std::endl;
        return false;
    }
    if (pipe(pipe_fd_) != 0) {
        std::cerr << "❌ [pcap] pipe() failed" << std::endl;
        pcap_close(handle_);
        handle_ = nullptr;
        return false;
    }
    std::cout << "✅ [pcap] Variant B — libpcap opened on " << interface << std::endl;
    std::cout << "⚠️  [pcap] STUB — DEBT-VARIANT-B-PCAP-IMPL-001 pending" << std::endl;
    return true;
}

int PcapBackend::poll(int /*timeout_ms*/) {
    // DEBT-VARIANT-B-PCAP-IMPL-001: implementar pcap_dispatch aquí
    return 0;
}

void PcapBackend::close() {
    if (handle_) {
        pcap_close(handle_);
        handle_ = nullptr;
    }
    if (pipe_fd_[0] != -1) { ::close(pipe_fd_[0]); pipe_fd_[0] = -1; }
    if (pipe_fd_[1] != -1) { ::close(pipe_fd_[1]); pipe_fd_[1] = -1; }
}

int PcapBackend::get_fd() const {
    return pipe_fd_[0];
}

uint64_t PcapBackend::get_packet_count() {
    return packet_count_;
}

} // namespace sniffer
