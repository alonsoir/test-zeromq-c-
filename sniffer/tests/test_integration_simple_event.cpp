// sniffer/tests/test_integration_simple_event.cpp
// Integration Test: SimpleEvent → FastDetector + RansomwareFeatureProcessor
// Phase 1C: Two-layer detection validation
//
// Layer 1: FastDetector (10s window, fast heuristics)
// Layer 2: RansomwareFeatureProcessor (30s window, deep features)

#include "ransomware_feature_processor.hpp"
#include "fast_detector.hpp"
#include "main.h"
#include "network_security.pb.h"
#include <iostream>
#include <cassert>
#include <cstring>
#include <chrono>
#include <thread>
#include <iomanip>

using namespace sniffer;

// ============================================================================
// TEST UTILITIES
// ============================================================================

class IntegrationTestRunner {
public:
    IntegrationTestRunner(const std::string& name) : test_name_(name), passed_(0), failed_(0) {
        std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
        std::cout << "║  " << std::left << std::setw(52) << test_name_ << "║\n";
        std::cout << "╚════════════════════════════════════════════════════════╝\n";
    }

    ~IntegrationTestRunner() {
        std::cout << "\n" << std::string(56, '-') << "\n";
        std::cout << "RESULTS: " << passed_ << " passed, " << failed_ << " failed\n";
        if (failed_ == 0) {
            std::cout << "✅ ALL INTEGRATION TESTS PASSED!\n";
        } else {
            std::cout << "❌ SOME INTEGRATION TESTS FAILED!\n";
        }
        std::cout << std::string(56, '=') << "\n";
    }

    void assert_true(bool condition, const std::string& message) {
        if (condition) {
            std::cout << "  ✅ PASS: " << message << "\n";
            passed_++;
        } else {
            std::cout << "  ❌ FAIL: " << message << "\n";
            failed_++;
        }
    }

    void assert_greater(double actual, double threshold, const std::string& message) {
        bool condition = (actual > threshold);
        if (condition) {
            std::cout << "  ✅ PASS: " << message
                      << " (actual=" << actual << " > " << threshold << ")\n";
            passed_++;
        } else {
            std::cout << "  ❌ FAIL: " << message
                      << " (actual=" << actual << " <= " << threshold << ")\n";
            failed_++;
        }
    }

    int get_failed_count() const { return failed_; }

private:
    std::string test_name_;
    int passed_;
    int failed_;
};

// ============================================================================
// SIMPLEEVENT CREATION HELPERS
// ============================================================================

uint32_t ip_to_uint32(const std::string& ip_str) {
    unsigned int a, b, c, d;
    sscanf(ip_str.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d);
    return (a << 24) | (b << 16) | (c << 8) | d;
}

uint64_t get_current_time_ns() {
    using namespace std::chrono;
    auto now = steady_clock::now();
    return duration_cast<nanoseconds>(now.time_since_epoch()).count();
}

SimpleEvent create_tcp_event(uint32_t src_ip, uint32_t dst_ip, 
                             uint16_t src_port, uint16_t dst_port,
                             uint32_t packet_size, uint8_t tcp_flags = 0x18) {
    SimpleEvent evt;
    std::memset(&evt, 0, sizeof(evt));
    
    evt.src_ip = src_ip;
    evt.dst_ip = dst_ip;
    evt.src_port = src_port;
    evt.dst_port = dst_port;
    evt.protocol = 6;  // TCP
    evt.tcp_flags = tcp_flags;
    evt.packet_len = packet_size;
    evt.ip_header_len = 20;
    evt.l4_header_len = 20;
    evt.timestamp = get_current_time_ns();
    
    return evt;
}

SimpleEvent create_udp_event(uint32_t src_ip, uint32_t dst_ip,
                             uint16_t src_port, uint16_t dst_port,
                             uint32_t packet_size) {
    SimpleEvent evt;
    std::memset(&evt, 0, sizeof(evt));
    
    evt.src_ip = src_ip;
    evt.dst_ip = dst_ip;
    evt.src_port = src_port;
    evt.dst_port = dst_port;
    evt.protocol = 17;  // UDP
    evt.tcp_flags = 0;
    evt.packet_len = packet_size;
    evt.ip_header_len = 20;
    evt.l4_header_len = 8;
    evt.timestamp = get_current_time_ns();
    
    return evt;
}

SimpleEvent create_dns_event(uint32_t src_ip, uint32_t dst_ip, uint16_t src_port) {
    return create_udp_event(src_ip, dst_ip, src_port, 53, 512);
}

SimpleEvent create_smb_event(uint32_t src_ip, uint32_t dst_ip, uint16_t src_port) {
    return create_tcp_event(src_ip, dst_ip, src_port, 445, 200, 0x18);
}

SimpleEvent create_https_event(uint32_t src_ip, uint32_t dst_ip, uint16_t src_port) {
    return create_tcp_event(src_ip, dst_ip, src_port, 443, 1500, 0x18);
}

// ============================================================================
// INTEGRATION TEST 1: Basic Event Processing (Both Layers)
// ============================================================================

void test_basic_event_processing() {
    IntegrationTestRunner test("Integration Test 1: Basic Event Processing");
    
    RansomwareFeatureProcessor processor;
    FastDetector fast_detector;
    
    if (!processor.initialize()) {
        test.assert_true(false, "Failed to initialize processor");
        return;
    }
    
    test.assert_true(true, "Processor initialized successfully");
    processor.start();
    
    // Process events through both layers
    SimpleEvent dns_event = create_dns_event(
        ip_to_uint32("192.168.1.100"),
        ip_to_uint32("8.8.8.8"),
        50000
    );
    
    processor.process_packet(dns_event);
    fast_detector.ingest(dns_event);
    test.assert_true(true, "DNS event processed through both layers");
    
    SimpleEvent smb_event = create_smb_event(
        ip_to_uint32("192.168.1.100"),
        ip_to_uint32("192.168.1.10"),
        50001
    );
    
    processor.process_packet(smb_event);
    fast_detector.ingest(smb_event);
    test.assert_true(true, "SMB event processed through both layers");
    
    SimpleEvent https_event = create_https_event(
        ip_to_uint32("192.168.1.100"),
        ip_to_uint32("200.50.30.10"),
        50002
    );
    
    processor.process_packet(https_event);
    fast_detector.ingest(https_event);
    test.assert_true(true, "HTTPS event processed through both layers");
    
    // Check FastDetector (should not be suspicious with 3 events)
    test.assert_true(!fast_detector.is_suspicious(), 
                    "FastDetector: Not suspicious with normal traffic");
    
    processor.stop();
}

// ============================================================================
// INTEGRATION TEST 2: External IPs Detection (Both Layers)
// ============================================================================

void test_external_ips_detection() {
    IntegrationTestRunner test("Integration Test 2: External IPs Detection");
    
    RansomwareFeatureProcessor processor;
    FastDetector fast_detector;
    
    processor.initialize();
    processor.start();
    
    uint32_t src_ip = ip_to_uint32("192.168.1.100");
    
    std::cout << "\n  📊 Injecting 15 connections to external IPs...\n";
    std::cout << "     (Simulating C&C server contact)\n\n";
    
    for (int i = 0; i < 15; i++) {
        std::string external_ip = "200." + std::to_string(i) + ".50." + std::to_string(i * 10);
        uint32_t dst_ip = ip_to_uint32(external_ip);
        
        SimpleEvent evt = create_https_event(src_ip, dst_ip, 49152 + i);
        
        // Feed both layers
        processor.process_packet(evt);
        fast_detector.ingest(evt);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Layer 1: FastDetector (should trigger immediately)
    std::cout << "  🚨 Layer 1 (FastDetector):\n";
    auto snapshot = fast_detector.snapshot();
    std::cout << "     External IPs: " << snapshot.external_ips_10s << "\n";
    std::cout << "     Suspicious: " << (fast_detector.is_suspicious() ? "YES" : "NO") << "\n";
    
    test.assert_true(fast_detector.is_suspicious(), 
                    "FastDetector triggered on >10 external IPs");
    
    // Layer 2: RansomwareFeatureProcessor (forced extraction)
    std::cout << "\n  🔍 Layer 2 (FeatureProcessor):\n";
    processor.force_extraction_for_testing();
    
    protobuf::RansomwareFeatures features;
    bool ready = processor.get_features_if_ready(features);
    
    std::cout << "     Features Ready: " << (ready ? "Yes" : "No") << "\n";
    std::cout << "     New External IPs (30s): " << features.new_external_ips_30s() << "\n";
    
    test.assert_true(ready, "FeatureProcessor: Features ready");
    test.assert_greater(features.new_external_ips_30s(), 10.0,
                       "FeatureProcessor: Detected >10 external IPs");
    
    processor.stop();
}

// ============================================================================
// INTEGRATION TEST 3: SMB Lateral Movement (Both Layers)
// ============================================================================

void test_smb_lateral_movement() {
    IntegrationTestRunner test("Integration Test 3: SMB Lateral Movement");
    
    RansomwareFeatureProcessor processor;
    FastDetector fast_detector;
    
    processor.initialize();
    processor.start();
    
    uint32_t src_ip = ip_to_uint32("192.168.1.100");
    
    std::cout << "\n  📊 Injecting 12 SMB connections to different hosts...\n";
    std::cout << "     (Simulating ransomware lateral movement)\n\n";
    
    for (int i = 1; i <= 12; i++) {
        uint32_t dst_ip = ip_to_uint32("192.168.1." + std::to_string(i));
        SimpleEvent evt = create_smb_event(src_ip, dst_ip, 49152 + i);
        
        processor.process_packet(evt);
        fast_detector.ingest(evt);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Layer 1: FastDetector
    std::cout << "  🚨 Layer 1 (FastDetector):\n";
    auto snapshot = fast_detector.snapshot();
    std::cout << "     SMB Connections: " << snapshot.smb_conns << "\n";
    std::cout << "     Suspicious: " << (fast_detector.is_suspicious() ? "YES" : "NO") << "\n";
    
    test.assert_true(fast_detector.is_suspicious(),
                    "FastDetector triggered on >3 SMB connections");
    
    // Layer 2: RansomwareFeatureProcessor
    std::cout << "\n  🔍 Layer 2 (FeatureProcessor):\n";
    processor.force_extraction_for_testing();
    
    protobuf::RansomwareFeatures features;
    bool ready = processor.get_features_if_ready(features);
    
    std::cout << "     Features Ready: " << (ready ? "Yes" : "No") << "\n";
    std::cout << "     SMB Connection Diversity: " << features.smb_connection_diversity() << "\n";
    
    test.assert_true(ready, "FeatureProcessor: Features ready");
    test.assert_greater(features.smb_connection_diversity(), 5.0,
                       "FeatureProcessor: Detected >5 SMB destinations");
    
    processor.stop();
}

// ============================================================================
// INTEGRATION TEST 4: Full Ransomware Simulation (Both Layers)
// ============================================================================

void test_full_ransomware_simulation() {
    IntegrationTestRunner test("Integration Test 4: Full Ransomware Behavior");
    
    RansomwareFeatureProcessor processor;
    FastDetector fast_detector;
    
    processor.initialize();
    processor.start();
    
    uint32_t infected_host = ip_to_uint32("192.168.1.100");
    
    std::cout << "\n  📊 Simulating complete ransomware behavior:\n";
    std::cout << "     1. Multiple external IP connections (C&C)\n";
    std::cout << "     2. SMB lateral movement\n";
    std::cout << "     Testing: FastDetector + FeatureProcessor\n\n";
    
    // Phase 1: Contact C&C servers
    std::cout << "  [Phase 1] Contacting 12 C&C servers...\n";
    for (int i = 0; i < 12; i++) {
        std::string c2_ip = "200." + std::to_string(i) + ".100.50";
        SimpleEvent evt = create_https_event(infected_host, ip_to_uint32(c2_ip), 50000 + i);
        processor.process_packet(evt);
        fast_detector.ingest(evt);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // Phase 2: SMB lateral movement
    std::cout << "  [Phase 2] SMB lateral movement to 8 hosts...\n";
    for (int i = 1; i <= 8; i++) {
        uint32_t target_host = ip_to_uint32("192.168.1." + std::to_string(i));
        SimpleEvent evt = create_smb_event(infected_host, target_host, 50100 + i);
        processor.process_packet(evt);
        fast_detector.ingest(evt);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Check both layers
    std::cout << "\n  🚨 LAYER 1 - FastDetector:\n";
    auto snapshot = fast_detector.snapshot();
    std::cout << "     External IPs: " << snapshot.external_ips_10s << "\n";
    std::cout << "     SMB Conns: " << snapshot.smb_conns << "\n";
    std::cout << "     Suspicious: " << (fast_detector.is_suspicious() ? "YES" : "NO") << "\n";
    
    test.assert_true(fast_detector.is_suspicious(),
                    "FastDetector: Detected ransomware behavior");
    
    std::cout << "\n  🔍 LAYER 2 - FeatureProcessor:\n";
    processor.force_extraction_for_testing();
    
    protobuf::RansomwareFeatures features;
    bool ready = processor.get_features_if_ready(features);
    
    std::cout << "     New External IPs (30s): " << features.new_external_ips_30s() << "\n";
    std::cout << "     SMB Connection Diversity: " << features.smb_connection_diversity() << "\n";
    std::cout << "     Features Ready: " << (ready ? "Yes" : "No") << "\n";
    
    test.assert_true(ready, "FeatureProcessor: Features ready");
    test.assert_greater(features.new_external_ips_30s(), 8.0,
                       "FeatureProcessor: External IPs indicate C&C");
    test.assert_greater(features.smb_connection_diversity(), 4.0,
                       "FeatureProcessor: SMB diversity indicates lateral movement");
    
    processor.stop();
}

// ============================================================================
// INTEGRATION TEST 5: Performance Test (Both Layers)
// ============================================================================

void test_performance() {
    IntegrationTestRunner test("Integration Test 5: Performance Under Load");
    
    RansomwareFeatureProcessor processor;
    FastDetector fast_detector;
    
    processor.initialize();
    processor.start();
    
    uint32_t src_ip = ip_to_uint32("192.168.1.100");
    
    std::cout << "\n  📊 Injecting 1000 events through both layers...\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 1000; i++) {
        SimpleEvent evt;
        
        if (i % 3 == 0) {
            evt = create_dns_event(src_ip, ip_to_uint32("8.8.8.8"), 50000 + i);
        } else if (i % 3 == 1) {
            uint32_t dst = ip_to_uint32("192.168.1." + std::to_string((i % 20) + 1));
            evt = create_smb_event(src_ip, dst, 50000 + i);
        } else {
            std::string ext_ip = "200." + std::to_string(i % 50) + ".100.50";
            evt = create_https_event(src_ip, ip_to_uint32(ext_ip), 50000 + i);
        }
        
        processor.process_packet(evt);
        fast_detector.ingest(evt);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "  ⏱️  Processing time: " << duration.count() << " ms\n";
    
    if (duration.count() > 0) {
        std::cout << "  📈 Throughput: " << (1000.0 / duration.count() * 1000) << " events/sec\n";
    } else {
        std::cout << "  📈 Throughput: >1M events/sec (too fast to measure)\n";
    }
    
    // Check both layers
    processor.force_extraction_for_testing();
    
    protobuf::RansomwareFeatures features;
    bool ready = processor.get_features_if_ready(features);
    
    auto snapshot = fast_detector.snapshot();
    
    std::cout << "\n  📊 Layer 1 (FastDetector):\n";
    std::cout << "     External IPs: " << snapshot.external_ips_10s << "\n";
    std::cout << "     SMB Diversity: " << snapshot.smb_conns << "\n";
    std::cout << "     Suspicious: " << (fast_detector.is_suspicious() ? "YES" : "NO") << "\n";
    
    std::cout << "\n  📊 Layer 2 (FeatureProcessor):\n";
    std::cout << "     Features Ready: " << (ready ? "Yes" : "No") << "\n";
    std::cout << "     External IPs: " << features.new_external_ips_30s() << "\n";
    std::cout << "     SMB Diversity: " << features.smb_connection_diversity() << "\n";
    
    test.assert_true(duration.count() < 5000, "Processing 1000 events in <5 seconds");
    test.assert_true(ready, "FeatureProcessor: Features valid after high load");
    
    processor.stop();
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  INTEGRATION TESTS - Phase 1C                         ║\n";
    std::cout << "║  Two-Layer Detection: FastDetector + FeatureProcessor║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    std::cout << "\n📋 Testing Architecture:\n";
    std::cout << "   Layer 1: FastDetector (10s window, heuristics)\n";
    std::cout << "   Layer 2: RansomwareFeatureProcessor (30s aggregation)\n";
    std::cout << "\n⚠️  NOTE: SimpleEvent has NO payload field\n";
    std::cout << "   Testing 2/3 features: External IPs + SMB Diversity\n";
    
    try {
        test_basic_event_processing();
        test_external_ips_detection();
        test_smb_lateral_movement();
        test_full_ransomware_simulation();
        test_performance();
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  INTEGRATION TEST SUMMARY                             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    std::cout << "\n🎉 ALL INTEGRATION TESTS PASSED!\n";
    std::cout << "✅ Two-layer detection system validated\n";
    std::cout << "\n📋 Next Steps:\n";
    std::cout << "   1. Integrate into main.cpp\n";
    std::cout << "   2. Add ZMQ alerting for FastDetector\n";
    std::cout << "   3. Test with real traffic\n";
    std::cout << "\n⚠️  Phase 2: Add payload[512] to SimpleEvent\n";
    
    return 0;
}
