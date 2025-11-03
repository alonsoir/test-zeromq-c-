// sniffer/tests/test_fast_detector.cpp
// Unit tests for FastDetector (Layer 1 heuristic detection)

#include "fast_detector.hpp"
#include "main.h"
#include <iostream>
#include <cassert>
#include <cstring>
#include <chrono>

using namespace sniffer;

// Helper: Convert IP string to uint32_t (host byte order)
uint32_t ip_to_uint32(const std::string& ip_str) {
    unsigned int a, b, c, d;
    sscanf(ip_str.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d);
    return (a << 24) | (b << 16) | (c << 8) | d;
}

// Helper: Create SimpleEvent
SimpleEvent create_event(uint32_t src_ip, uint32_t dst_ip, uint16_t dst_port, 
                        uint8_t protocol, uint8_t tcp_flags, uint64_t timestamp_ns) {
    SimpleEvent evt;
    std::memset(&evt, 0, sizeof(evt));
    evt.src_ip = src_ip;
    evt.dst_ip = dst_ip;
    evt.src_port = 50000;
    evt.dst_port = dst_port;
    evt.protocol = protocol;
    evt.tcp_flags = tcp_flags;
    evt.packet_len = 512;
    evt.ip_header_len = 20;
    evt.l4_header_len = (protocol == 6) ? 20 : 8;
    evt.timestamp = timestamp_ns;
    return evt;
}

void test_basic_functionality() {
    std::cout << "\n=== Test 1: Basic Functionality ===\n";
    FastDetector fd;
    
    uint64_t t0 = 1'000'000'000ULL;  // 1 second
    SimpleEvent evt = create_event(
        ip_to_uint32("192.168.1.100"),
        ip_to_uint32("8.8.8.8"),
        80, 6, 0x18, t0
    );
    
    fd.ingest(evt);
    assert(!fd.is_suspicious());
    std::cout << "✅ Single event not suspicious\n";
}

void test_external_ips_threshold() {
    std::cout << "\n=== Test 2: External IPs Threshold ===\n";
    FastDetector fd;
    
    uint64_t t0 = 1'000'000'000ULL;
    uint32_t src_ip = ip_to_uint32("192.168.1.100");
    
    // Inject 15 connections to different external IPs
    for (int i = 0; i < 15; i++) {
        std::string ext_ip = "200." + std::to_string(i) + ".50.1";
        uint32_t dst_ip = ip_to_uint32(ext_ip);
        SimpleEvent evt = create_event(src_ip, dst_ip, 443, 6, 0x18, t0 + i * 100'000'000ULL);
        fd.ingest(evt);
    }
    
    auto snapshot = fd.snapshot();
    std::cout << "External IPs detected: " << snapshot.external_ips_10s << "\n";
    assert(snapshot.external_ips_10s == 15);
    assert(fd.is_suspicious());
    std::cout << "✅ Detected C&C scanning (>10 external IPs)\n";
}

void test_smb_lateral_movement() {
    std::cout << "\n=== Test 3: SMB Lateral Movement ===\n";
    FastDetector fd;
    
    uint64_t t0 = 1'000'000'000ULL;
    uint32_t src_ip = ip_to_uint32("192.168.1.100");
    
    // Inject 5 SMB connections to different internal hosts
    for (int i = 1; i <= 5; i++) {
        std::string internal_ip = "192.168.1." + std::to_string(i);
        uint32_t dst_ip = ip_to_uint32(internal_ip);
        SimpleEvent evt = create_event(src_ip, dst_ip, 445, 6, 0x18, t0 + i * 100'000'000ULL);
        fd.ingest(evt);
    }
    
    auto snapshot = fd.snapshot();
    std::cout << "SMB connections detected: " << snapshot.smb_conns << "\n";
    assert(snapshot.smb_conns == 5);
    assert(fd.is_suspicious());
    std::cout << "✅ Detected lateral movement (>3 SMB connections)\n";
}

void test_port_scanning() {
    std::cout << "\n=== Test 4: Port Scanning ===\n";
    FastDetector fd;
    
    uint64_t t0 = 1'000'000'000ULL;
    uint32_t src_ip = ip_to_uint32("192.168.1.100");
    uint32_t dst_ip = ip_to_uint32("192.168.1.200");
    
    // Scan 15 different ports
    for (int i = 1; i <= 15; i++) {
        SimpleEvent evt = create_event(src_ip, dst_ip, 8000 + i, 6, 0x18, t0 + i * 100'000'000ULL);
        fd.ingest(evt);
    }
    
    auto snapshot = fd.snapshot();
    std::cout << "Unique ports scanned: " << snapshot.unique_ports << "\n";
    assert(snapshot.unique_ports == 15);
    assert(fd.is_suspicious());
    std::cout << "✅ Detected port scanning (>10 ports)\n";
}

void test_rst_ratio() {
    std::cout << "\n=== Test 5: High RST Ratio ===\n";
    FastDetector fd;
    
    uint64_t t0 = 1'000'000'000ULL;
    uint32_t src_ip = ip_to_uint32("192.168.1.100");
    uint32_t dst_ip = ip_to_uint32("192.168.1.200");
    
    // Send 10 TCP packets, 5 with RST flag
    for (int i = 0; i < 10; i++) {
        uint8_t flags = (i % 2 == 0) ? 0x04 : 0x18;  // RST or PSH+ACK
        SimpleEvent evt = create_event(src_ip, dst_ip, 80, 6, flags, t0 + i * 100'000'000ULL);
        fd.ingest(evt);
    }
    
    auto snapshot = fd.snapshot();
    double rst_ratio = static_cast<double>(snapshot.resets) / snapshot.total_tcp;
    std::cout << "RST ratio: " << rst_ratio << " (5/10)\n";
    assert(rst_ratio >= 0.2);
    assert(fd.is_suspicious());
    std::cout << "✅ Detected high RST ratio (reconnaissance)\n";
}

void test_window_expiration() {
    std::cout << "\n=== Test 6: Window Expiration ===\n";
    FastDetector fd;
    
    uint64_t t0 = 1'000'000'000ULL;
    uint32_t src_ip = ip_to_uint32("192.168.1.100");
    
    // Inject 15 external IPs
    for (int i = 0; i < 15; i++) {
        std::string ext_ip = "200." + std::to_string(i) + ".50.1";
        SimpleEvent evt = create_event(src_ip, ip_to_uint32(ext_ip), 443, 6, 0x18, t0);
        fd.ingest(evt);
    }
    
    assert(fd.is_suspicious());
    std::cout << "Suspicious at t=1s: YES\n";
    
    // Move to t=15s (window expired)
    uint64_t t1 = t0 + 15'000'000'000ULL;
    SimpleEvent evt_new = create_event(src_ip, ip_to_uint32("8.8.8.8"), 443, 6, 0x18, t1);
    fd.ingest(evt_new);
    
    assert(!fd.is_suspicious());
    std::cout << "Suspicious at t=15s (after window): NO\n";
    std::cout << "✅ Window expiration works correctly\n";
}

void test_no_false_positives() {
    std::cout << "\n=== Test 7: No False Positives ===\n";
    FastDetector fd;
    
    uint64_t t0 = 1'000'000'000ULL;
    uint32_t src_ip = ip_to_uint32("192.168.1.100");
    
    // Normal traffic: 5 external IPs, 2 SMB, 5 ports
    for (int i = 0; i < 5; i++) {
        std::string ext_ip = "200." + std::to_string(i) + ".50.1";
        SimpleEvent evt = create_event(src_ip, ip_to_uint32(ext_ip), 443, 6, 0x18, 
                                      t0 + i * 100'000'000ULL);
        fd.ingest(evt);
    }
    
    auto snapshot = fd.snapshot();
    std::cout << "External IPs: " << snapshot.external_ips_10s << " (below threshold)\n";
    assert(!fd.is_suspicious());
    std::cout << "✅ Normal traffic not flagged as suspicious\n";
}

int main() {
    std::cout << "\n╔════════════════════════════════════════════╗\n";
    std::cout << "║  FastDetector Unit Tests                  ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n";
    
    try {
        test_basic_functionality();
        test_external_ips_threshold();
        test_smb_lateral_movement();
        test_port_scanning();
        test_rst_ratio();
        test_window_expiration();
        test_no_false_positives();
        
        std::cout << "\n╔════════════════════════════════════════════╗\n";
        std::cout << "║  ✅ ALL TESTS PASSED                      ║\n";
        std::cout << "╚════════════════════════════════════════════╝\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
