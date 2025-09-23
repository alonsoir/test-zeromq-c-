#pragma once

#include <string>
#include <memory>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

namespace sniffer {

class EbpfLoader {
public:
    EbpfLoader();
    ~EbpfLoader();

    // Cargar programa eBPF desde archivo objeto
    bool load_program(const std::string& bpf_obj_path);
    
    // Adjuntar programa XDP a interfaz
    bool attach_xdp(const std::string& interface_name);
    
    // Desadjuntar programa XDP de interfaz
    bool detach_xdp(const std::string& interface_name);
    
    // Obtener file descriptor del ring buffer map
    int get_ringbuf_fd() const;
    
    // Obtener file descriptor del stats map
    int get_stats_fd() const;
    
    // Verificar si el programa está cargado y adjuntado
    bool is_loaded() const { return program_loaded_; }
    bool is_attached() const { return xdp_attached_; }
    
    // Obtener estadísticas del programa eBPF
    uint64_t get_packet_count();
    
private:
    struct bpf_object* bpf_obj_;
    struct bpf_program* xdp_prog_;
    struct bpf_map* events_map_;
    struct bpf_map* stats_map_;
    
    int prog_fd_;
    int events_fd_;
    int stats_fd_;
    
    bool program_loaded_;
    bool xdp_attached_;
    int attached_ifindex_;
    
    // Helper para obtener ifindex desde nombre
    int get_ifindex(const std::string& interface_name);
};

} // namespace sniffer
