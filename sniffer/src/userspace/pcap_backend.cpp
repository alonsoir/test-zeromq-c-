// ADR-029 Variant B — PcapBackend implementation
// DEBT-VARIANT-B-PCAP-IMPL-001: pcap_dispatch → callback → ZeroMQ pipeline
// DAY 138 — 2026-05-01
#include "pcap_backend.hpp"
#include <iostream>
#include <unistd.h>

namespace sniffer {

// Static callback requerido por pcap_dispatch.
// user apunta a PcapCallbackData — sin acceso a miembros privados.
static void pcap_packet_handler(u_char* user,
                                const struct pcap_pkthdr* hdr,
                                const u_char* data) {
    auto* d = reinterpret_cast<PcapCallbackData*>(user);
    if (d->cb && data && hdr->caplen > 0)
        d->cb(d->ctx,
              const_cast<void*>(reinterpret_cast<const void*>(data)),
              static_cast<size_t>(hdr->caplen));
}

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
    cb_data_ = {cb, ctx};

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
    return true;
}

// poll(): despacha hasta 64 paquetes por llamada, no bloqueante.
// Retorna: >0 pkts procesados, 0 timeout/vacío, <0 error.
int PcapBackend::poll(int /*timeout_ms*/) {
    if (!handle_) return -1;
    int n = pcap_dispatch(handle_, 64,
                          pcap_packet_handler,
                          reinterpret_cast<u_char*>(&cb_data_));
    if (n > 0)
        packet_count_ += static_cast<uint64_t>(n);
    else if (n < 0 && n != PCAP_ERROR_BREAK)
        std::cerr << "❌ [pcap] pcap_dispatch error: "
                  << pcap_geterr(handle_) << std::endl;
    return n;
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
