//sniffer/src/kernel/sniffer.bpf.c
// Enhanced eBPF sniffer v3.2 - Hybrid filtering system
// Supports both blacklist (excluded_ports) and whitelist (included_ports)

// Type definitions
typedef unsigned char __u8;
typedef unsigned short __u16;
typedef unsigned int __u32;
typedef unsigned long long __u64;
typedef signed char __s8;
typedef signed short __s16;
typedef signed int __s32;
typedef signed long long __s64;

typedef __u16 __be16;
typedef __u32 __be32;
typedef __u64 __be64;
typedef __u32 __wsum;

struct iphdr;
struct ipv6hdr;
struct tcphdr;
struct __sk_buff;

#include <bpf/bpf_helpers.h>

// Basic definitions
#define ETH_HLEN 14
#define XDP_PASS 2
#define BPF_MAP_TYPE_RINGBUF 27
#define BPF_MAP_TYPE_ARRAY 2
#define BPF_MAP_TYPE_HASH 1
#define BPF_ANY 0

// TCP flags
#define TCP_FLAG_FIN 0x01
#define TCP_FLAG_SYN 0x02
#define TCP_FLAG_RST 0x04
#define TCP_FLAG_PSH 0x08
#define TCP_FLAG_ACK 0x10
#define TCP_FLAG_URG 0x20
#define TCP_FLAG_ECE 0x40
#define TCP_FLAG_CWR 0x80

// Filter actions
#define ACTION_DROP 0
#define ACTION_CAPTURE 1

// XDP context
struct xdp_md {
    __u32 data;
    __u32 data_end;
    __u32 data_meta;
    __u32 ingress_ifindex;
    __u32 rx_queue_index;
    __u32 egress_ifindex;
};

// Event structure
struct simple_event {
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u8 protocol;
    __u8 tcp_flags;
    __u32 packet_len;
    __u16 ip_header_len;
    __u16 l4_header_len;
    __u64 timestamp;
} __attribute__((packed));

// Filter configuration
struct filter_config {
    __u8 default_action;  // 0 = drop, 1 = capture
    __u8 reserved[7];
};

// 🔥 MAP 1: Excluded ports (blacklist)
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u16);    // Port number
    __type(value, __u8);   // 1 = excluded
} excluded_ports SEC(".maps");

// 🔥 MAP 2: Included ports (whitelist - HIGH PRIORITY)
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u16);    // Port number
    __type(value, __u8);   // 1 = included
} included_ports SEC(".maps");

// 🔥 MAP 3: Global filter settings
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct filter_config);
} filter_settings SEC(".maps");

// Ring buffer for events
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 20);  // 1MB
} events SEC(".maps");

// Statistics
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} stats SEC(".maps");

// 🔥 FILTER LOGIC: Decide if port should be captured
// Returns: 1 = capture, 0 = drop
static __always_inline int should_capture_port(__u16 port) {
    // STEP 1: Check whitelist (HIGHEST PRIORITY)
    __u8 *included = bpf_map_lookup_elem(&included_ports, &port);
    if (included && *included == 1) {
        return ACTION_CAPTURE;  // ✅ Whitelist always wins
    }

    // STEP 2: Check blacklist
    __u8 *excluded = bpf_map_lookup_elem(&excluded_ports, &port);
    if (excluded && *excluded == 1) {
        return ACTION_DROP;  // ❌ In blacklist, drop
    }

    // STEP 3: Apply default action
    __u32 key = 0;
    struct filter_config *config = bpf_map_lookup_elem(&filter_settings, &key);
    if (config) {
        return config->default_action;
    }

    return ACTION_CAPTURE;  // Fallback: capture
}

// Helper: Extract TCP flags
static __always_inline __u8 extract_tcp_flags(void *tcp_start, void *data_end) {
    if (tcp_start + 20 > data_end)
        return 0;
    __u8 *tcp = (__u8*)tcp_start;
    return tcp[13];  // Flags byte
}

// Helper: Get TCP header length
static __always_inline __u16 get_tcp_header_len(void *tcp_start, void *data_end) {
    if (tcp_start + 13 > data_end)
        return 0;
    __u8 *tcp = (__u8*)tcp_start;
    __u8 data_offset = (tcp[12] >> 4) & 0x0F;
    return data_offset * 4;
}

SEC("xdp")
int xdp_sniffer_enhanced(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    // Verify ethernet header
    if (data + ETH_HLEN > data_end)
        return XDP_PASS;

    void *ip_start = data + ETH_HLEN;

    // Verify IP header
    if (ip_start + 20 > data_end)
        return XDP_PASS;

    __u8 *ip = (__u8*)ip_start;

    // Verify IPv4
    if ((ip[0] >> 4) != 4)
        return XDP_PASS;

    // Reserve ring buffer space
    struct simple_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return XDP_PASS;
    }

    __builtin_memset(event, 0, sizeof(*event));

    // Extract IPs
    event->src_ip = (ip[12] << 24) | (ip[13] << 16) | (ip[14] << 8) | ip[15];
    event->dst_ip = (ip[16] << 24) | (ip[17] << 16) | (ip[18] << 8) | ip[19];

    // Protocol
    event->protocol = ip[9];

    // Packet length
    event->packet_len = data_end - data;

    // IP header length
    __u8 ihl = (ip[0] & 0x0F) * 4;
    event->ip_header_len = ihl;

    // Timestamp
    event->timestamp = bpf_ktime_get_ns();

    // Layer 4 processing
    void *l4_start = ip_start + ihl;

    if (event->protocol == 6) {
        // ============ TCP ============
        if (l4_start + 4 > data_end) {
            bpf_ringbuf_discard(event, 0);
            return XDP_PASS;
        }

        __u8 *tcp = (__u8*)l4_start;

        // Extract ports
        event->src_port = (tcp[0] << 8) | tcp[1];
        event->dst_port = (tcp[2] << 8) | tcp[3];

        // 🔥 APPLY FILTER - Check destination port
        if (!should_capture_port(event->dst_port)) {
            bpf_ringbuf_discard(event, 0);
            return XDP_PASS;
        }

        // 🔥 APPLY FILTER - Check source port
        if (!should_capture_port(event->src_port)) {
            bpf_ringbuf_discard(event, 0);
            return XDP_PASS;
        }

        // Extract TCP flags
        event->tcp_flags = extract_tcp_flags(l4_start, data_end);

        // TCP header length
        event->l4_header_len = get_tcp_header_len(l4_start, data_end);

    } else if (event->protocol == 17) {
        // ============ UDP ============
        if (l4_start + 4 > data_end) {
            bpf_ringbuf_discard(event, 0);
            return XDP_PASS;
        }

        __u8 *udp = (__u8*)l4_start;

        // Extract ports
        event->src_port = (udp[0] << 8) | udp[1];
        event->dst_port = (udp[2] << 8) | udp[3];

        // 🔥 APPLY FILTER - Check destination port
        if (!should_capture_port(event->dst_port)) {
            bpf_ringbuf_discard(event, 0);
            return XDP_PASS;
        }

        // 🔥 APPLY FILTER - Check source port
        if (!should_capture_port(event->src_port)) {
            bpf_ringbuf_discard(event, 0);
            return XDP_PASS;
        }

        // UDP header is always 8 bytes
        event->l4_header_len = 8;
        event->tcp_flags = 0;

    } else {
        // ============ OTHER PROTOCOLS ============
        event->src_port = 0;
        event->dst_port = 0;
        event->tcp_flags = 0;
        event->l4_header_len = 0;
    }

    // Submit event to ring buffer
    bpf_ringbuf_submit(event, 0);

    // Update statistics
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