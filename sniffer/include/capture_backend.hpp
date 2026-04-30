#pragma once
// ADR-029 — CaptureBackend: interfaz abstracta para captura de paquetes
// Variant A: EbpfBackend (eBPF/XDP + ring buffer)
// Variant B: PcapBackend (libpcap, stub — DEBT-VARIANT-B-PCAP-IMPL-001)
// DAY 137 — 2026-04-30
#include <string>
#include <cstdint>
#include <cstddef>

namespace sniffer {

class CaptureBackend {
public:
    // Callback de paquete: (ctx, data, size) → 0 OK
    using PacketCallback = int(*)(void* ctx, void* data, size_t size);

    virtual ~CaptureBackend() = default;

    // Abrir dispositivo y registrar callback
    virtual bool open(const std::string& interface,
                      PacketCallback cb, void* ctx) = 0;

    // Poll bloqueante con timeout en ms — devuelve nº eventos o <0 error
    virtual int poll(int timeout_ms) = 0;

    // Cerrar y liberar recursos
    virtual void close() = 0;

    // fd para select/epoll externo (ring buffer fd en eBPF, pipe en pcap)
    virtual int get_fd() const = 0;

    virtual uint64_t get_packet_count() = 0;

    // Filter map fds — solo significativos en eBPF, -1 en pcap
    // attach/detach — no-op en PcapBackend (open() lo gestiona todo)
    virtual bool attach_skb(const std::string& /*iface*/) { return true; }
    virtual bool detach_skb(const std::string& /*iface*/) { return true; }

    // Alias de get_fd() para compatibilidad con código existente
    virtual int get_ringbuf_fd() const { return get_fd(); }

    virtual int get_excluded_ports_fd()    const { return -1; }
    virtual int get_included_ports_fd()    const { return -1; }
    virtual int get_filter_settings_fd()   const { return -1; }
    virtual int get_interface_configs_fd() const { return -1; }
};

} // namespace sniffer
