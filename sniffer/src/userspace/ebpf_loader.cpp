#include "ebpf_loader.hpp"
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <net/if.h>
#include <linux/if_link.h>
// sniffer/src/userspace/ebpf_loader.hpp

namespace sniffer {

EbpfLoader::EbpfLoader() 
    : bpf_obj_(nullptr), 
      xdp_prog_(nullptr), 
      events_map_(nullptr), 
      stats_map_(nullptr),
      excluded_ports_map_(nullptr),
      included_ports_map_(nullptr),
      filter_settings_map_(nullptr),
      prog_fd_(-1), 
      events_fd_(-1), 
      stats_fd_(-1),
      excluded_ports_fd_(-1),
      included_ports_fd_(-1),
      filter_settings_fd_(-1),
      program_loaded_(false),
      xdp_attached_(false),
      skb_attached_(false),
      attached_ifindex_(-1) {
}

EbpfLoader::~EbpfLoader() {
    if (xdp_attached_ && attached_ifindex_ >= 0) {
        // Desadjuntar XDP
        bpf_xdp_detach(attached_ifindex_, 0, nullptr);
    }

    if (skb_attached_ && attached_ifindex_ >= 0) {
        // Desadjuntar SKB (TC)
        LIBBPF_OPTS(bpf_tc_hook, hook,
                    .ifindex = attached_ifindex_,
                    .attach_point = BPF_TC_INGRESS);
        LIBBPF_OPTS(bpf_tc_opts, opts, .handle = 1, .priority = 1);
        bpf_tc_detach(&hook, &opts);
        bpf_tc_hook_destroy(&hook);
    }
    
    if (bpf_obj_) {
        bpf_object__close(bpf_obj_);
        std::cout << "[INFO] eBPF object closed" << std::endl;
    }
}

bool EbpfLoader::load_program(const std::string& bpf_obj_path) {
    std::cout << "[INFO] Loading eBPF program from: " << bpf_obj_path << std::endl;
    
    // Abrir objeto eBPF
    bpf_obj_ = bpf_object__open(bpf_obj_path.c_str());
    if (!bpf_obj_) {
        std::cerr << "[ERROR] Failed to open eBPF object: " << bpf_obj_path << std::endl;
        return false;
    }
    
    // Cargar programa en el kernel
    int err = bpf_object__load(bpf_obj_);
    if (err) {
        std::cerr << "[ERROR] Failed to load eBPF program: " << strerror(-err) << std::endl;
        bpf_object__close(bpf_obj_);
        bpf_obj_ = nullptr;
        return false;
    }
    
    // Obtener el programa XDP. Está definido en src/kernel/sniffer.bpf.c
    xdp_prog_ = bpf_object__find_program_by_name(bpf_obj_, "xdp_sniffer_enhanced");
    if (!xdp_prog_) {
        std::cerr << "[ERROR] Failed to find XDP program 'xdp_sniffer_enhanced'" << std::endl;
        bpf_object__close(bpf_obj_);
        bpf_obj_ = nullptr;
        return false;
    }
    
    prog_fd_ = bpf_program__fd(xdp_prog_);
    if (prog_fd_ < 0) {
        std::cerr << "[ERROR] Failed to get program file descriptor" << std::endl;
        bpf_object__close(bpf_obj_);
        bpf_obj_ = nullptr;
        return false;
    }
    
    // Obtener maps
    events_map_ = bpf_object__find_map_by_name(bpf_obj_, "events");
    if (!events_map_) {
        std::cerr << "[ERROR] Failed to find 'events' map" << std::endl;
        bpf_object__close(bpf_obj_);
        bpf_obj_ = nullptr;
        return false;
    }
    
    stats_map_ = bpf_object__find_map_by_name(bpf_obj_, "stats");
    if (!stats_map_) {
        std::cerr << "[ERROR] Failed to find 'stats' map" << std::endl;
        bpf_object__close(bpf_obj_);
        bpf_obj_ = nullptr;
        return false;
    }
    
    events_fd_ = bpf_map__fd(events_map_);
    stats_fd_ = bpf_map__fd(stats_map_);
    
    if (events_fd_ < 0 || stats_fd_ < 0) {
        std::cerr << "[ERROR] Failed to get map file descriptors" << std::endl;
        bpf_object__close(bpf_obj_);
        bpf_obj_ = nullptr;
        return false;
    }

    // Get filter maps
    excluded_ports_map_ = bpf_object__find_map_by_name(bpf_obj_, "excluded_ports");
    if (excluded_ports_map_) {
        excluded_ports_fd_ = bpf_map__fd(excluded_ports_map_);
        std::cout << "[INFO] Found excluded_ports map, FD: " << excluded_ports_fd_ << std::endl;
    } else {
        std::cerr << "[WARNING] excluded_ports map not found in eBPF program" << std::endl;
    }

    included_ports_map_ = bpf_object__find_map_by_name(bpf_obj_, "included_ports");
    if (included_ports_map_) {
        included_ports_fd_ = bpf_map__fd(included_ports_map_);
        std::cout << "[INFO] Found included_ports map, FD: " << included_ports_fd_ << std::endl;
    } else {
        std::cerr << "[WARNING] included_ports map not found in eBPF program" << std::endl;
    }

    filter_settings_map_ = bpf_object__find_map_by_name(bpf_obj_, "filter_settings");
    if (filter_settings_map_) {
        filter_settings_fd_ = bpf_map__fd(filter_settings_map_);
        std::cout << "[INFO] Found filter_settings map, FD: " << filter_settings_fd_ << std::endl;
    } else {
        std::cerr << "[WARNING] filter_settings map not found in eBPF program" << std::endl;
    }

    program_loaded_ = true;
    std::cout << "[INFO] eBPF program loaded successfully" << std::endl;
    std::cout << "[INFO] Program FD: " << prog_fd_ << ", Events FD: " << events_fd_ 
              << ", Stats FD: " << stats_fd_ << std::endl;
    
    return true;
}

bool EbpfLoader::attach_xdp(const std::string& interface_name) {
    if (!program_loaded_) {
        std::cerr << "[ERROR] Program not loaded. Call load_program() first" << std::endl;
        return false;
    }
    
    int ifindex = get_ifindex(interface_name);
    if (ifindex < 0) {
        std::cerr << "[ERROR] Interface not found: " << interface_name << std::endl;
        return false;
    }
    
    std::cout << "[INFO] Attaching XDP program to interface: " << interface_name 
              << " (ifindex: " << ifindex << ")" << std::endl;
    
    // Adjuntar programa XDP
    int err = bpf_xdp_attach(ifindex, prog_fd_, 0, nullptr);
    if (err) {
        std::cerr << "[ERROR] Failed to attach XDP program: " << strerror(-err) << std::endl;
        return false;
    }
    
    xdp_attached_ = true;
    attached_ifindex_ = ifindex;
    
    std::cout << "[INFO] XDP program attached successfully to " << interface_name << std::endl;
    return true;
}

bool EbpfLoader::detach_xdp(const std::string& interface_name) {
    if (!xdp_attached_) {
        std::cout << "[INFO] No XDP program attached" << std::endl;
        return true;
    }
    
    int ifindex = get_ifindex(interface_name);
    if (ifindex < 0) {
        std::cerr << "[ERROR] Interface not found: " << interface_name << std::endl;
        return false;
    }
    
    std::cout << "[INFO] Detaching XDP program from interface: " << interface_name << std::endl;
    
    int err = bpf_xdp_detach(ifindex, 0, nullptr);
    if (err) {
        std::cerr << "[ERROR] Failed to detach XDP program: " << strerror(-err) << std::endl;
        return false;
    }
    
    xdp_attached_ = false;
    attached_ifindex_ = -1;
    
    std::cout << "[INFO] XDP program detached successfully" << std::endl;
    return true;
}

bool EbpfLoader::attach_skb(const std::string& interface_name) {
    if (!program_loaded_) {
        std::cerr << "[ERROR] eBPF program not loaded" << std::endl;
        return false;
    }

    if (skb_attached_) {
        std::cout << "[WARN] SKB program already attached" << std::endl;
        return true;
    }

    int ifindex = get_ifindex(interface_name);
    if (ifindex < 0) {
        std::cerr << "[ERROR] Failed to get interface index for " << interface_name << std::endl;
        return false;
    }

    std::cout << "[INFO] Attaching XDP program in SKB/Generic mode to interface: " << interface_name
              << " (ifindex: " << ifindex << ")" << std::endl;

    // Usar XDP en modo genérico (software) - compatible con virtio_net
    // XDP_FLAGS_SKB_MODE = (1U << 1) = modo genérico/software
    __u32 xdp_flags = (1U << 1);  // XDP_FLAGS_SKB_MODE

    int err = bpf_xdp_attach(ifindex, prog_fd_, xdp_flags, nullptr);
    if (err) {
        std::cerr << "[ERROR] Failed to attach XDP in SKB mode: " << strerror(-err) << std::endl;
        return false;
    }

    skb_attached_ = true;
    attached_ifindex_ = ifindex;

    std::cout << "[INFO] XDP program attached successfully in SKB/Generic mode to " << interface_name << std::endl;
    return true;
}


bool EbpfLoader::detach_skb(const std::string& interface_name) {
    if (!skb_attached_) {
        return true;
    }

    int ifindex = get_ifindex(interface_name);
    if (ifindex < 0) {
        return false;
    }

    std::cout << "[INFO] Detaching XDP program (SKB mode) from interface: " << interface_name << std::endl;

    // Desadjuntar XDP
    int err = bpf_xdp_detach(ifindex, 0, nullptr);
    if (err) {
        std::cerr << "[WARN] Failed to detach XDP (SKB mode): " << strerror(-err) << std::endl;
    }

    skb_attached_ = false;
    attached_ifindex_ = -1;

    std::cout << "[INFO] XDP program (SKB mode) detached from " << interface_name << std::endl;
    return true;
}

int EbpfLoader::get_ringbuf_fd() const {
    return events_fd_;
}

int EbpfLoader::get_stats_fd() const {
    return stats_fd_;
}

uint64_t EbpfLoader::get_packet_count() {
    if (!program_loaded_ || stats_fd_ < 0) {
        return 0;
    }
    
    uint32_t key = 0;
    uint64_t value = 0;
    
    int err = bpf_map_lookup_elem(stats_fd_, &key, &value);
    if (err) {
        return 0;
    }
    
    return value;
}

int EbpfLoader::get_ifindex(const std::string& interface_name) {
    return if_nametoindex(interface_name.c_str());
}

    int EbpfLoader::get_excluded_ports_fd() const {
    return excluded_ports_fd_;
}

    int EbpfLoader::get_included_ports_fd() const {
    return included_ports_fd_;
}

    int EbpfLoader::get_filter_settings_fd() const {
    return filter_settings_fd_;
}

} // namespace sniffer
