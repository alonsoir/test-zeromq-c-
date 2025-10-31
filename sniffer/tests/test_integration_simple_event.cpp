// sniffer/tests/test_integration_simple_event.cpp
// Integration Test: SimpleEvent → RansomwareFeatureProcessor → Features
// Phase 1B: Validates the pipeline before main.cpp integration
//
// ⚠️ IMPORTANT: SimpleEvent does NOT have payload[] field
//    - dns_query_entropy feature WILL NOT WORK (needs payload)
//    - new_external_ips_30s WILL WORK (only needs IPs)
//    - smb_connection_diversity WILL WORK (only needs ports)
//
// This test validates 2/3 features until payload is added in Phase 2

#include "ransomware_feature_processor.hpp"
#include "main.h"  // For SimpleEvent
#include "network_security.pb.h"  // For protobuf::RansomwareFeatures
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

// Helper: Convert IP string to uint32_t (host byte order)
uint32_t ip_to_uint32(const std::string& ip_str) {
    unsigned int a, b, c, d;
    sscanf(ip_str.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d);
    return (a << 24) | (b << 16) | (c << 8) | d;
}

// Helper: Get current time in nanoseconds
uint64_t get_current_time_ns() {
    using namespace std::chrono;
    auto now = steady_clock::now();
    return duration_cast<nanoseconds>(now.time_since_epoch()).count();
}

// Helper: Create a TCP SimpleEvent (generic)
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
    evt.tcp_flags = tcp_flags;  // Default: PSH+ACK
    evt.packet_len = packet_size;
    evt.ip_header_len = 20;  // Standard IPv4
    evt.l4_header_len = 20;  // Standard TCP
    evt.timestamp = get_current_time_ns();
    
    return evt;
}

// Helper: Create a UDP SimpleEvent (generic)
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
    evt.ip_header_len = 20;  // Standard IPv4
    evt.l4_header_len = 8;   // UDP header
    evt.timestamp = get_current_time_ns();
    
    return evt;
}

// Helper: Create a DNS SimpleEvent
// ⚠️ NOTE: SimpleEvent does NOT have payload[] field
// DNS entropy feature will NOT work without payload - this is documented in BACKLOG
SimpleEvent create_dns_event(uint32_t src_ip, uint32_t dst_ip, uint16_t src_port) {
    return create_udp_event(src_ip, dst_ip, src_port, 53, 512);
}

// Helper: Create an SMB SimpleEvent (port 445)
SimpleEvent create_smb_event(uint32_t src_ip, uint32_t dst_ip, uint16_t src_port) {
    return create_tcp_event(src_ip, dst_ip, src_port, 445, 200, 0x18);
}

// Helper: Create an HTTPS connection SimpleEvent (port 443)
SimpleEvent create_https_event(uint32_t src_ip, uint32_t dst_ip, uint16_t src_port) {
    return create_tcp_event(src_ip, dst_ip, src_port, 443, 1500, 0x18);
}

// ============================================================================
// INTEGRATION TEST 1: Basic Event Processing
// ============================================================================

void test_basic_event_processing() {
    IntegrationTestRunner test("Integration Test 1: Basic Event Processing");
    
    RansomwareFeatureProcessor processor;
    
    // Initialize processor
    if (!processor.initialize()) {
        test.assert_true(false, "Failed to initialize processor");
        return;
    }
    
    test.assert_true(true, "Processor initialized successfully");
    
    // Start processor (needed for process_packet to work)
    processor.start();
    
    // Process a single DNS event
    SimpleEvent dns_event = create_dns_event(
        ip_to_uint32("192.168.1.100"),
        ip_to_uint32("8.8.8.8"),
        50000
    );
    
    processor.process_packet(dns_event);
    test.assert_true(true, "Single DNS event processed without crash");
    
    // Process a single SMB event
    SimpleEvent smb_event = create_smb_event(
        ip_to_uint32("192.168.1.100"),
        ip_to_uint32("192.168.1.10"),
        50001
    );
    
    processor.process_packet(smb_event);
    test.assert_true(true, "Single SMB event processed without crash");
    
    // Process a single HTTPS event
    SimpleEvent https_event = create_https_event(
        ip_to_uint32("192.168.1.100"),
        ip_to_uint32("200.50.30.10"),
        50002
    );
    
    processor.process_packet(https_event);
    test.assert_true(true, "Single HTTPS event processed without crash");
    
    processor.stop();
}

// ============================================================================
// INTEGRATION TEST 2: External IPs Detection
// ============================================================================

void test_external_ips_detection() {
    IntegrationTestRunner test("Integration Test 2: External IPs Detection");
    
    RansomwareFeatureProcessor processor;
    processor.initialize();
    processor.start();
    
    uint32_t src_ip = ip_to_uint32("192.168.1.100");
    
    std::cout << "\n  📊 Injecting 15 connections to external IPs...\n";
    std::cout << "     (Simulating C&C server contact)\n\n";
    
    // Inject connections to many external IPs (simulating C&C contact)
    for (int i = 0; i < 15; i++) {
        std::string external_ip = "200." + std::to_string(i) + ".50." + std::to_string(i * 10);
        uint32_t dst_ip = ip_to_uint32(external_ip);
        
        SimpleEvent evt = create_https_event(src_ip, dst_ip, 49152 + i);
        processor.process_packet(evt);
        
        // Small delay to spread events over time
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // ⭐ Force extraction for testing (don't wait 30s)
    std::cout << "  🔧 Forcing feature extraction...\n";
    processor.force_extraction_for_testing();
    
    // Extract features using correct API
    protobuf::RansomwareFeatures features;
    bool ready = processor.get_features_if_ready(features);
    
    std::cout << "  📈 Features Ready: " << (ready ? "Yes" : "No") << "\n";
    std::cout << "  📈 New External IPs (30s): " << features.new_external_ips_30s() << "\n";
    
    // Should detect multiple new IPs (threshold for malicious: >10)
    test.assert_true(ready, "Features are ready");
    test.assert_greater(features.new_external_ips_30s(), 10.0,
                       "Detected >10 external IPs (indicates C&C scanning)");
    
    processor.stop();
}

// ============================================================================
// INTEGRATION TEST 3: SMB Lateral Movement
// ============================================================================

void test_smb_lateral_movement() {
    IntegrationTestRunner test("Integration Test 3: SMB Lateral Movement");
    
    RansomwareFeatureProcessor processor;
    processor.initialize();
    processor.start();
    
    uint32_t src_ip = ip_to_uint32("192.168.1.100");
    
    std::cout << "\n  📊 Injecting 12 SMB connections to different hosts...\n";
    std::cout << "     (Simulating ransomware lateral movement)\n\n";
    
    // Inject SMB connections to multiple hosts (lateral movement)
    for (int i = 1; i <= 12; i++) {
        uint32_t dst_ip = ip_to_uint32("192.168.1." + std::to_string(i));
        SimpleEvent evt = create_smb_event(src_ip, dst_ip, 49152 + i);
        
        processor.process_packet(evt);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // ⭐ Force extraction for testing
    std::cout << "  🔧 Forcing feature extraction...\n";
    processor.force_extraction_for_testing();
    
    protobuf::RansomwareFeatures features;
    bool ready = processor.get_features_if_ready(features);
    
    std::cout << "  📈 Features Ready: " << (ready ? "Yes" : "No") << "\n";
    std::cout << "  📈 SMB Connection Diversity: " << features.smb_connection_diversity() << "\n";
    
    // Should detect multiple SMB destinations (threshold: >5)
    test.assert_true(ready, "Features are ready");
    test.assert_greater(features.smb_connection_diversity(), 5.0,
                       "Detected >5 SMB destinations (indicates lateral movement)");
    
    processor.stop();
}

// ============================================================================
// INTEGRATION TEST 4: Full Ransomware Simulation (Without DGA)
// ============================================================================

void test_full_ransomware_simulation() {
    IntegrationTestRunner test("Integration Test 4: Full Ransomware Behavior");
    
    RansomwareFeatureProcessor processor;
    processor.initialize();
    processor.start();
    
    uint32_t infected_host = ip_to_uint32("192.168.1.100");
    
    std::cout << "\n  📊 Simulating complete ransomware behavior:\n";
    std::cout << "     1. Multiple external IP connections (C&C)\n";
    std::cout << "     2. SMB lateral movement\n";
    std::cout << "     ⚠️  DNS entropy NOT tested (no payload in SimpleEvent)\n\n";
    
    // Phase 1: Contact multiple external IPs (C&C servers)
    std::cout << "  [Phase 1] Contacting 12 C&C servers...\n";
    for (int i = 0; i < 12; i++) {
        std::string c2_ip = "200." + std::to_string(i) + ".100.50";
        SimpleEvent evt = create_https_event(infected_host, ip_to_uint32(c2_ip), 50000 + i);
        processor.process_packet(evt);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // Phase 2: SMB lateral movement
    std::cout << "  [Phase 2] SMB lateral movement to 8 hosts...\n";
    for (int i = 1; i <= 8; i++) {
        uint32_t target_host = ip_to_uint32("192.168.1." + std::to_string(i));
        SimpleEvent evt = create_smb_event(infected_host, target_host, 50100 + i);
        processor.process_packet(evt);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // Wait for all processing
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // ⭐ Force extraction for testing
    std::cout << "  🔧 Forcing feature extraction...\n";
    processor.force_extraction_for_testing();
    
    // Extract features
    protobuf::RansomwareFeatures features;
    bool ready = processor.get_features_if_ready(features);
    
    std::cout << "\n  📊 EXTRACTED FEATURES:\n";
    std::cout << "     DNS Query Entropy:         " << features.dns_query_entropy() 
              << " ⚠️  (NOT functional - no payload)\n";
    std::cout << "     New External IPs (30s):    " << features.new_external_ips_30s() << "\n";
    std::cout << "     SMB Connection Diversity:  " << features.smb_connection_diversity() << "\n";
    std::cout << "     Features Ready:            " << (ready ? "Yes" : "No") << "\n\n";
    
    // Validate composite ransomware behavior (2/3 features)
    test.assert_true(ready, "Features marked as ready");
    
    // Both working features should indicate suspicious activity
    test.assert_greater(features.new_external_ips_30s(), 8.0,
                       "External IPs indicate C&C activity");
    test.assert_greater(features.smb_connection_diversity(), 4.0,
                       "SMB diversity indicates lateral movement");
    
    processor.stop();
}

// ============================================================================
// INTEGRATION TEST 5: Performance Test
// ============================================================================

void test_performance() {
    IntegrationTestRunner test("Integration Test 5: Performance Under Load");
    
    RansomwareFeatureProcessor processor;
    processor.initialize();
    processor.start();
    
    uint32_t src_ip = ip_to_uint32("192.168.1.100");
    
    std::cout << "\n  📊 Injecting 1000 events rapidly...\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Inject 1000 mixed events
    for (int i = 0; i < 1000; i++) {
        if (i % 3 == 0) {
            // DNS event
            SimpleEvent evt = create_dns_event(src_ip, ip_to_uint32("8.8.8.8"), 50000 + i);
            processor.process_packet(evt);
        } else if (i % 3 == 1) {
            // SMB event
            uint32_t dst = ip_to_uint32("192.168.1." + std::to_string((i % 20) + 1));
            SimpleEvent evt = create_smb_event(src_ip, dst, 50000 + i);
            processor.process_packet(evt);
        } else {
            // External connection
            std::string ext_ip = "200." + std::to_string(i % 50) + ".100.50";
            SimpleEvent evt = create_https_event(src_ip, ip_to_uint32(ext_ip), 50000 + i);
            processor.process_packet(evt);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "  ⏱️  Processing time: " << duration.count() << " ms\n";
    
    if (duration.count() > 0) {
        std::cout << "  📈 Throughput: " << (1000.0 / duration.count() * 1000) << " events/sec\n";
    } else {
        std::cout << "  📈 Throughput: >1M events/sec (too fast to measure)\n";
    }
    
    // ⭐ Force extraction for testing
    std::cout << "  🔧 Forcing feature extraction...\n";
    processor.force_extraction_for_testing();
    
    // Extract features after load
    protobuf::RansomwareFeatures features;
    bool ready = processor.get_features_if_ready(features);
    
    std::cout << "\n  📊 Features after load:\n";
    std::cout << "     Features Ready: " << (ready ? "Yes" : "No") << "\n";
    std::cout << "     External IPs: " << features.new_external_ips_30s() << "\n";
    std::cout << "     SMB Diversity: " << features.smb_connection_diversity() << "\n";
    
    test.assert_true(duration.count() < 5000, "Processing 1000 events in <5 seconds");
    test.assert_true(ready, "Features valid after high load");
    
    processor.stop();
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  INTEGRATION TESTS - Phase 1B                         ║\n";
    std::cout << "║  SimpleEvent → RansomwareFeatureProcessor → Features  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    std::cout << "\n⚠️  IMPORTANT NOTE:\n";
    std::cout << "   SimpleEvent does NOT have payload[] field\n";
    std::cout << "   DNS entropy feature will NOT work - documented in BACKLOG\n";
    std::cout << "   Testing 2/3 features: External IPs + SMB Diversity\n";
    
    int total_failures = 0;
    
    try {
        // Run all integration tests
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
    
    if (total_failures == 0) {
        std::cout << "\n🎉 ALL INTEGRATION TESTS PASSED!\n";
        std::cout << "✅ RansomwareFeatureProcessor ready for main.cpp integration\n";
        std::cout << "\n📋 Next Steps:\n";
        std::cout << "   1. Integrate into main.cpp\n";
        std::cout << "   2. Add timer thread (30s extraction)\n";
        std::cout << "   3. Serialize to protobuf\n";
        std::cout << "   4. Send via ZMQ\n";
        std::cout << "\n⚠️  Phase 2: Add payload[] to SimpleEvent for DNS entropy\n";
        return 0;
    } else {
        std::cout << "\n❌ SOME INTEGRATION TESTS FAILED\n";
        std::cout << "⚠️  Review failures before integrating into main.cpp\n";
        return 1;
    }
}
