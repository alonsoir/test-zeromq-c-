// aRGus NDR — Sniffer Variant B (libpcap)
// ADR-029 — delta XDP vs libpcap: contribución científica para paper + FEDER
// DEBT-VARIANT-B-PCAP-IMPL-001: poll loop pendiente (stub compilable)
// DAY 137 — 2026-04-30
#include "pcap_backend.hpp"
#include "capture_backend.hpp"
#include <iostream>
#include <csignal>
#include <atomic>

static std::atomic<bool> g_running{true};

static void signal_handler(int) { g_running = false; }

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║  aRGus NDR — Sniffer Variant B (libpcap)      ║\n";
    std::cout << "║  ADR-029 — delta científico XDP vs libpcap    ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n\n";

    std::string interface = "eth0";
    if (argc > 1) interface = argv[1];

    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    sniffer::PcapBackend backend;
    if (!backend.open(interface, nullptr, nullptr)) {
        std::cerr << "❌ Failed to open libpcap on " << interface << std::endl;
        return 1;
    }

    std::cout << "✅ Variant B running — interface: " << interface << "\n";
    std::cout << "⚠️  DEBT-VARIANT-B-PCAP-IMPL-001: poll loop pendiente\n\n";

    // DEBT-VARIANT-B-PCAP-IMPL-001: aquí irá pcap_dispatch → ZeroMQ → ml-detector
    while (g_running) {
        backend.poll(100);
    }

    backend.close();
    std::cout << "\n✅ Variant B stopped cleanly\n";
    return 0;
}
