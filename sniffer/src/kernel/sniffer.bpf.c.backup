//sniffer/src/kernel/sniffer.bpf.c
// Definiciones completas de tipos
typedef unsigned char __u8;
typedef unsigned short __u16;
typedef unsigned int __u32;
typedef unsigned long long __u64;
typedef signed char __s8;
typedef signed short __s16;
typedef signed int __s32;
typedef signed long long __s64;

// Big endian types
typedef __u16 __be16;
typedef __u32 __be32;
typedef __u64 __be64;

// Checksum type
typedef __u32 __wsum;

// Forward declarations
struct iphdr;
struct ipv6hdr;
struct tcphdr;
struct __sk_buff;

// Incluir headers de eBPF
#include <bpf/bpf_helpers.h>

// Definiciones básicas
#define ETH_HLEN 14
#define XDP_PASS 2
#define BPF_MAP_TYPE_RINGBUF 27
#define BPF_MAP_TYPE_ARRAY 2
#define BPF_ANY 0

// Estructura del contexto XDP
struct xdp_md {
    __u32 data;
    __u32 data_end;
    __u32 data_meta;
    __u32 ingress_ifindex;
    __u32 rx_queue_index;
    __u32 egress_ifindex;
};

// Estructura simple para eventos
struct simple_event {
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u8 protocol;
    __u32 packet_len;
    __u64 timestamp;
} __attribute__((packed));

// Ring buffer
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 20);
} events SEC(".maps");

// Estadísticas
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} stats SEC(".maps");

SEC("xdp")
int xdp_sniffer_simple(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    // Verificar que tenemos al menos ethernet header
    if (data + ETH_HLEN > data_end)
        return XDP_PASS;

    // Saltamos ethernet header
    void *ip_start = data + ETH_HLEN;

    // Verificar que tenemos al menos 20 bytes de IP header
    if (ip_start + 20 > data_end)
        return XDP_PASS;

    // Leer campos básicos del IP header
    __u8 *ip = (__u8*)ip_start;
    if ((ip[0] >> 4) != 4) // Verificar IPv4
        return XDP_PASS;

    struct simple_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event)
        return XDP_PASS;

    // Extraer IPs (bytes 12-15 src, 16-19 dst)
    event->src_ip = (ip[12] << 24) | (ip[13] << 16) | (ip[14] << 8) | ip[15];
    event->dst_ip = (ip[16] << 24) | (ip[17] << 16) | (ip[18] << 8) | ip[19];
    event->protocol = ip[9];
    event->packet_len = data_end - data;
    event->timestamp = bpf_ktime_get_ns();
    event->src_port = 0;
    event->dst_port = 0;

    // Si es TCP o UDP, extraer puertos
    if (event->protocol == 6 || event->protocol == 17) {
        __u8 ihl = (ip[0] & 0x0F) * 4;
        if (ip_start + ihl + 4 <= data_end) {
            __u8 *l4 = (__u8*)(ip_start + ihl);
            event->src_port = (l4[0] << 8) | l4[1];
            event->dst_port = (l4[2] << 8) | l4[3];
        }
    }

    bpf_ringbuf_submit(event, 0);

    // Actualizar estadísticas
    __u32 key = 0;
    __u64 *count = bpf_map_lookup_elem(&stats, &key);
    if (count)
        __sync_fetch_and_add(count, 1);
    else {
        __u64 initial = 1;
        bpf_map_update_elem(&stats, &key, &initial, BPF_ANY);
    }

    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";