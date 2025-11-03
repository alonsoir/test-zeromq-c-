// sniffer/include/main.h
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

namespace sniffer {

// TCP flag definitions - matching eBPF
constexpr uint8_t TCP_FLAG_FIN = 0x01;
constexpr uint8_t TCP_FLAG_SYN = 0x02;
constexpr uint8_t TCP_FLAG_RST = 0x04;
constexpr uint8_t TCP_FLAG_PSH = 0x08;
constexpr uint8_t TCP_FLAG_ACK = 0x10;
constexpr uint8_t TCP_FLAG_URG = 0x20;
constexpr uint8_t TCP_FLAG_ECE = 0x40;
constexpr uint8_t TCP_FLAG_CWR = 0x80;

// ⭐ ENHANCED - Simple event structure with TCP flags
// CRITICAL: Must match EXACTLY the eBPF struct simple_event
struct SimpleEvent {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t protocol;
    uint8_t tcp_flags;
    uint32_t packet_len;
    uint16_t ip_header_len;
    uint16_t l4_header_len;
    uint64_t timestamp;
    uint16_t payload_len;
    uint8_t payload[512];
} __attribute__((packed));

// Helper functions for TCP flags
inline bool has_tcp_flag(const SimpleEvent& event, uint8_t flag) {
    return (event.tcp_flags & flag) != 0;
}

inline bool is_tcp_syn(const SimpleEvent& event) { return has_tcp_flag(event, TCP_FLAG_SYN); }
inline bool is_tcp_fin(const SimpleEvent& event) { return has_tcp_flag(event, TCP_FLAG_FIN); }
inline bool is_tcp_rst(const SimpleEvent& event) { return has_tcp_flag(event, TCP_FLAG_RST); }
inline bool is_tcp_psh(const SimpleEvent& event) { return has_tcp_flag(event, TCP_FLAG_PSH); }
inline bool is_tcp_ack(const SimpleEvent& event) { return has_tcp_flag(event, TCP_FLAG_ACK); }
inline bool is_tcp_urg(const SimpleEvent& event) { return has_tcp_flag(event, TCP_FLAG_URG); }

// Helper: Get payload length (packet - IP header - L4 header)
inline uint32_t get_payload_length(const SimpleEvent& event) {
    uint32_t headers_len = 14 + event.ip_header_len + event.l4_header_len; // ETH + IP + L4
    return (event.packet_len > headers_len) ? (event.packet_len - headers_len) : 0;
}

// ✅ SAFE VERSION: Pretty print TCP flags without buffer overflow risk
inline const char* tcp_flags_to_string(uint8_t flags) {
    static thread_local char buf[64];

    if (flags == 0) {
        return "NONE";
    }

    std::string out;
    out.reserve(sizeof(buf) - 1);  // Pre-reserva para evitar reallocs

    auto append = [&](const char* s) {
        if (!out.empty()) out.push_back('|');
        out.append(s);
    };

    if (flags & TCP_FLAG_FIN) append("FIN");
    if (flags & TCP_FLAG_SYN) append("SYN");
    if (flags & TCP_FLAG_RST) append("RST");
    if (flags & TCP_FLAG_PSH) append("PSH");
    if (flags & TCP_FLAG_ACK) append("ACK");
    if (flags & TCP_FLAG_URG) append("URG");
    if (flags & TCP_FLAG_ECE) append("ECE");
    if (flags & TCP_FLAG_CWR) append("CWR");

    // Evitar overflow: truncar si excede el buffer
    if (out.size() >= sizeof(buf)) {
        out.resize(sizeof(buf) - 1);
    }

    std::memcpy(buf, out.data(), out.size());
    buf[out.size()] = '\0';

    return buf;
}

} // namespace sniffer