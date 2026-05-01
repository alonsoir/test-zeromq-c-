// aRGus NDR — Sniffer Variant B (libpcap)
// ADR-029 — pipeline completo: pcap → protobuf → LZ4 → ChaCha20 → ZeroMQ
// Wire format idéntico a Variant A (ring_consumer.cpp DAY 98)
// DAY 138 — 2026-05-01
#include "pcap_backend.hpp"
#include "capture_backend.hpp"
#include "network_security.pb.h"
#include <seed_client/seed_client.hpp>
#include <crypto_transport/transport.hpp>
#include <crypto_transport/contexts.hpp>
#include <zmq.hpp>
#include <lz4.h>
#include <iostream>
#include <csignal>
#include <atomic>
#include <cstring>
#include <chrono>
// ETH/IP/TCP/UDP parsing
#include <netinet/ether.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <arpa/inet.h>

// ============================================================================
// Contexto pasado al callback de captura
// ============================================================================
struct PcapPipelineContext {
    zmq::socket_t*                          socket;
    crypto_transport::CryptoTransport*      tx;
    std::atomic<uint64_t>*                  packets_sent;
    std::atomic<uint64_t>*                  send_failures;
};

static std::atomic<bool> g_running{true};
static void signal_handler(int) { g_running = false; }

// ============================================================================
// packet_callback: raw ETH frame → NetworkSecurityEvent → LZ4 → encrypt → ZMQ
// Mismo wire format que Variant A (ring_consumer.cpp:689-760)
// ============================================================================
static int packet_callback(void* ctx, void* data, size_t size) {
    auto* pc = reinterpret_cast<PcapPipelineContext*>(ctx);
    if (!pc || !data || size < sizeof(struct ether_header)) return 0;

    const auto* eth = reinterpret_cast<const struct ether_header*>(data);
    if (ntohs(eth->ether_type) != ETHERTYPE_IP) return 0;

    if (size < sizeof(struct ether_header) + sizeof(struct ip)) return 0;
    const auto* iph = reinterpret_cast<const struct ip*>(
        static_cast<const uint8_t*>(data) + sizeof(struct ether_header));

    // --- Construir NetworkSecurityEvent mínimo ---
    protobuf::NetworkSecurityEvent event;
    auto* nf = event.mutable_network_features();

    char src_str[INET_ADDRSTRLEN], dst_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &iph->ip_src, src_str, sizeof(src_str));
    inet_ntop(AF_INET, &iph->ip_dst, dst_str, sizeof(dst_str));
    nf->set_source_ip(src_str);
    nf->set_destination_ip(dst_str);
    nf->set_protocol_number(iph->ip_p);

    size_t ip_hdr_len = static_cast<size_t>(iph->ip_hl) * 4;
    size_t transport_offset = sizeof(struct ether_header) + ip_hdr_len;

    if (iph->ip_p == IPPROTO_TCP &&
        size >= transport_offset + sizeof(struct tcphdr)) {
        const auto* tcp = reinterpret_cast<const struct tcphdr*>(
            static_cast<const uint8_t*>(data) + transport_offset);
        nf->set_source_port(ntohs(tcp->source));
        nf->set_destination_port(ntohs(tcp->dest));
    } else if (iph->ip_p == IPPROTO_UDP &&
               size >= transport_offset + sizeof(struct udphdr)) {
        const auto* udp = reinterpret_cast<const struct udphdr*>(
            static_cast<const uint8_t*>(data) + transport_offset);
        nf->set_source_port(ntohs(udp->source));
        nf->set_destination_port(ntohs(udp->dest));
    }

    auto ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    event.mutable_event_timestamp()->set_seconds(ts / 1'000'000'000LL);
    event.mutable_event_timestamp()->set_nanos(
        static_cast<int32_t>(ts % 1'000'000'000LL));
    event.mutable_custom_metadata()->insert({"capture_variant", "libpcap"});

    // --- Serialize ---
    std::string serialized;
    if (!event.SerializeToString(&serialized)) return 0;

    // --- LZ4 compress: [uint32_t orig_size LE] + compressed ---
    std::vector<uint8_t> to_encrypt;
    {
        int orig = static_cast<int>(serialized.size());
        int max_c = LZ4_compressBound(orig);
        std::vector<uint8_t> compressed(sizeof(uint32_t) + max_c);
        uint32_t orig_le = static_cast<uint32_t>(orig);
        std::memcpy(compressed.data(), &orig_le, sizeof(orig_le));
        int c_size = LZ4_compress_default(
            serialized.data(),
            reinterpret_cast<char*>(compressed.data() + sizeof(uint32_t)),
            orig, max_c);
        if (c_size > 0) {
            compressed.resize(sizeof(uint32_t) + c_size);
            to_encrypt = std::move(compressed);
        } else {
            to_encrypt.assign(serialized.begin(), serialized.end());
        }
    }

    // --- Encrypt ---
    auto encrypted = pc->tx->encrypt(to_encrypt);

    // --- ZeroMQ PUSH (dontwait) ---
    zmq::message_t msg(encrypted.size());
    std::memcpy(msg.data(), encrypted.data(), encrypted.size());
    auto result = pc->socket->send(msg, zmq::send_flags::dontwait);
    if (result.has_value()) {
        pc->packets_sent->fetch_add(1, std::memory_order_relaxed);
    } else {
        pc->send_failures->fetch_add(1, std::memory_order_relaxed);
    }
    return 0;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║  aRGus NDR — Sniffer Variant B (libpcap)      ║\n";
    std::cout << "║  ADR-029 — pipeline completo DAY 138          ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n\n";

    std::string interface = "eth1";
    if (argc > 1) interface = argv[1];

    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    // --- CryptoTransport (mismo seed que Variant A) ---
    std::unique_ptr<ml_defender::SeedClient> seed_client;
    std::unique_ptr<crypto_transport::CryptoTransport> tx;
    try {
        seed_client = std::make_unique<ml_defender::SeedClient>(
            "/etc/ml-defender/sniffer/sniffer.json");
        seed_client->load();
        tx = std::make_unique<crypto_transport::CryptoTransport>(
            *seed_client, ml_defender::crypto::CTX_SNIFFER_TO_ML);
        std::cout << "✅ CryptoTransport inicializado (HKDF-SHA256 + ChaCha20-Poly1305)\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ CryptoTransport init failed: " << e.what() << "\n";
        return 1;
    }

    // --- ZeroMQ PUSH socket ---
    zmq::context_t zmq_ctx(1);
    zmq::socket_t  zmq_sock(zmq_ctx, zmq::socket_type::push);
    zmq_sock.connect("tcp://127.0.0.1:5571");
    std::cout << "✅ ZeroMQ PUSH conectado a tcp://127.0.0.1:5571\n";

    // --- Stats ---
    std::atomic<uint64_t> packets_sent{0};
    std::atomic<uint64_t> send_failures{0};

    PcapPipelineContext pc{&zmq_sock, tx.get(), &packets_sent, &send_failures};

    // --- Abrir captura ---
    sniffer::PcapBackend backend;
    if (!backend.open(interface, packet_callback, &pc)) {
        std::cerr << "❌ Failed to open libpcap on " << interface << "\n";
        return 1;
    }

    std::cout << "✅ Variant B running — interface: " << interface << "\n";
    std::cout << "   Ctrl+C para detener\n\n";

    auto t_start = std::chrono::steady_clock::now();
    while (g_running) {
        backend.poll(100);
    }

    backend.close();

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cout << "\n✅ Variant B stopped — "
              << backend.get_packet_count() << " pkts capturados, "
              << packets_sent.load() << " enviados, "
              << send_failures.load() << " fallos, "
              << elapsed << "s\n";
    return 0;
}
