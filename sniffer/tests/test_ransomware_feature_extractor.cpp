// sniffer/tests/test_ransomware_feature_extractor.cpp
// Unit tests for RansomwareFeatureExtractor - Phase 1A (3 critical features)

#include "ransomware_feature_extractor.hpp"
#include "flow_tracker.hpp"
#include "dns_analyzer.hpp"
#include "ip_whitelist.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <thread>
#include <chrono>

using namespace sniffer;

// ============================================================================
// TEST UTILITIES
// ============================================================================

class TestRunner {
public:
    TestRunner(const std::string& name) : test_name_(name), passed_(0), failed_(0) {
        std::cout << "\n========================================\n";
        std::cout << "TEST: " << test_name_ << "\n";
        std::cout << "========================================\n";
    }

    ~TestRunner() {
        std::cout << "\n----------------------------------------\n";
        std::cout << "RESULTS: " << passed_ << " passed, " << failed_ << " failed\n";
        if (failed_ == 0) {
            std::cout << "✅ ALL TESTS PASSED!\n";
        } else {
            std::cout << "❌ SOME TESTS FAILED!\n";
        }
        std::cout << "========================================\n";
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

    void assert_near(double actual, double expected, double tolerance, const std::string& message) {
        bool condition = std::abs(actual - expected) <= tolerance;
        if (condition) {
            std::cout << "  ✅ PASS: " << message
                      << " (actual=" << actual << ", expected=" << expected << ")\n";
            passed_++;
        } else {
            std::cout << "  ❌ FAIL: " << message
                      << " (actual=" << actual << ", expected=" << expected << ")\n";
            failed_++;
        }
    }

    void assert_in_range(double actual, double min_val, double max_val, const std::string& message) {
        bool condition = (actual >= min_val && actual <= max_val);
        if (condition) {
            std::cout << "  ✅ PASS: " << message
                      << " (actual=" << actual << ", range=[" << min_val << ", " << max_val << "])\n";
            passed_++;
        } else {
            std::cout << "  ❌ FAIL: " << message
                      << " (actual=" << actual << ", range=[" << min_val << ", " << max_val << "])\n";
            failed_++;
        }
    }

    int get_failed_count() const { return failed_; }

private:
    std::string test_name_;
    int passed_;
    int failed_;
};

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

// ============================================================================
// TEST 1: Basic Initialization
// ============================================================================

void test_basic_initialization() {
    TestRunner test("Basic Initialization");

    FlowTracker flow_tracker(10000, 300);
    DNSAnalyzer dns_analyzer(1000);
    IPWhitelist ip_whitelist(10000, 24 * 3600);

    RansomwareFeatureExtractor extractor(flow_tracker, dns_analyzer, ip_whitelist);

    test.assert_true(extractor.get_event_count() == 0,
                     "Initial event count is 0");

    // Extract features with no data
    auto features = extractor.extract_features_phase1a();

    test.assert_true(features.dns_query_entropy == 0.0f,
                     "DNS entropy is 0 with no data");
    test.assert_true(features.new_external_ips_30s == 0,
                     "New external IPs is 0 with no data");
    test.assert_true(features.smb_connection_diversity == 0,
                     "SMB diversity is 0 with no data");
}

// ============================================================================
// TEST 2: Feature 1 - DNS Query Entropy (Benign)
// ============================================================================

void test_dns_entropy_benign() {
    TestRunner test("DNS Entropy - Benign Traffic");

    FlowTracker flow_tracker(10000, 300);
    DNSAnalyzer dns_analyzer(1000);
    IPWhitelist ip_whitelist(10000, 24 * 3600);

    RansomwareFeatureExtractor extractor(flow_tracker, dns_analyzer, ip_whitelist);

    uint64_t now = get_current_time_ns();
    uint32_t src_ip = ip_to_uint32("192.168.1.100");

    // Add benign DNS queries (low entropy)
    dns_analyzer.add_query("google.com", now, true, src_ip);
    dns_analyzer.add_query("facebook.com", now + 1000000, true, src_ip);
    dns_analyzer.add_query("youtube.com", now + 2000000, true, src_ip);
    dns_analyzer.add_query("amazon.com", now + 3000000, true, src_ip);
    dns_analyzer.add_query("twitter.com", now + 4000000, true, src_ip);

    float entropy = extractor.extract_dns_query_entropy();

    std::cout << "  📊 Benign DNS Entropy: " << entropy << "\n";

    // Benign traffic should have entropy 2.0 - 4.5
    test.assert_in_range(entropy, 2.0, 4.5,
                         "Benign DNS entropy in expected range [2.0, 4.5]");
}

// ============================================================================
// TEST 3: Feature 1 - DNS Query Entropy (Malicious DGA)
// ============================================================================

void test_dns_entropy_malicious() {
    TestRunner test("DNS Entropy - Malicious DGA");

    FlowTracker flow_tracker(10000, 300);
    DNSAnalyzer dns_analyzer(1000);
    IPWhitelist ip_whitelist(10000, 24 * 3600);

    RansomwareFeatureExtractor extractor(flow_tracker, dns_analyzer, ip_whitelist);

    uint64_t now = get_current_time_ns();
    uint32_t src_ip = ip_to_uint32("192.168.1.100");

    // Add DGA-like DNS queries (high entropy, random strings)
    dns_analyzer.add_query("xjf8dk2jf93.com", now, false, src_ip);
    dns_analyzer.add_query("9fj3kd8s2df.com", now + 1000000, false, src_ip);
    dns_analyzer.add_query("2kd9fj3sdf8.net", now + 2000000, false, src_ip);
    dns_analyzer.add_query("8sdk2jf93kd.org", now + 3000000, false, src_ip);
    dns_analyzer.add_query("3jf8dk2j9sd.com", now + 4000000, false, src_ip);
    dns_analyzer.add_query("k2jf93kd8sd.com", now + 5000000, false, src_ip);
    dns_analyzer.add_query("f93kd8sdk2j.net", now + 6000000, false, src_ip);
    dns_analyzer.add_query("d8sdk2jf93k.org", now + 7000000, false, src_ip);

    float entropy = extractor.extract_dns_query_entropy();

    std::cout << "  📊 Malicious DGA Entropy: " << entropy << "\n";

    // DGA should have entropy > 6.0
    test.assert_true(entropy > 6.0,
                     "DGA DNS entropy > 6.0 (indicates malicious)");
}

// ============================================================================
// TEST 4: Feature 2 - New External IPs (Benign)
// ============================================================================

void test_new_external_ips_benign() {
    TestRunner test("New External IPs - Benign Traffic");

    FlowTracker flow_tracker(10000, 300);
    DNSAnalyzer dns_analyzer(1000);
    IPWhitelist ip_whitelist(10000, 24 * 3600);

    RansomwareFeatureExtractor extractor(flow_tracker, dns_analyzer, ip_whitelist);

    uint64_t now = get_current_time_ns();

    // Add a few external IPs (normal browsing)
    uint32_t google_ip = ip_to_uint32("142.250.80.46");      // Google
    uint32_t cloudflare_ip = ip_to_uint32("104.16.132.229"); // Cloudflare
    uint32_t amazon_ip = ip_to_uint32("54.239.28.85");       // Amazon

    ip_whitelist.add_ip(google_ip, now);
    ip_whitelist.add_ip(cloudflare_ip, now + 1'000'000'000);
    ip_whitelist.add_ip(amazon_ip, now + 2'000'000'000);

    // Count new IPs in 30s window
    uint64_t window_start = now;
    uint64_t window_end = now + (30 * 1'000'000'000ULL);

    int32_t new_ips = extractor.extract_new_external_ips_30s(window_start, window_end);

    std::cout << "  📊 Benign New External IPs: " << new_ips << "\n";

    // Benign traffic: 0-5 new IPs
    test.assert_in_range(new_ips, 0.0, 5.0,
                         "Benign new external IPs in range [0, 5]");
}

// ============================================================================
// TEST 5: Feature 2 - New External IPs (Malicious C&C)
// ============================================================================

void test_new_external_ips_malicious() {
    TestRunner test("New External IPs - Malicious C&C");

    FlowTracker flow_tracker(10000, 300);
    DNSAnalyzer dns_analyzer(1000);
    IPWhitelist ip_whitelist(10000, 24 * 3600);

    RansomwareFeatureExtractor extractor(flow_tracker, dns_analyzer, ip_whitelist);

    uint64_t now = get_current_time_ns();

    // Simulate ransomware contacting many C&C servers
    for (int i = 0; i < 20; i++) {
        uint32_t random_ip = ip_to_uint32(
            "200." + std::to_string(i) + ".50." + std::to_string(i * 10));

        ip_whitelist.add_ip(random_ip, now + (i * 100'000'000ULL));
    }

    // Count new IPs in 30s window
    uint64_t window_start = now;
    uint64_t window_end = now + (30 * 1'000'000'000ULL);

    int32_t new_ips = extractor.extract_new_external_ips_30s(window_start, window_end);

    std::cout << "  📊 Malicious New External IPs: " << new_ips << "\n";

    // Malicious traffic: >10 new IPs
    test.assert_true(new_ips > 10,
                     "Malicious new external IPs > 10 (indicates C&C contact)");
}

// ============================================================================
// TEST 6: Feature 3 - SMB Connection Diversity (Benign)
// ============================================================================

void test_smb_diversity_benign() {
    TestRunner test("SMB Diversity - Benign Traffic");

    FlowTracker flow_tracker(10000, 300);
    DNSAnalyzer dns_analyzer(1000);
    IPWhitelist ip_whitelist(10000, 24 * 3600);

    RansomwareFeatureExtractor extractor(flow_tracker, dns_analyzer, ip_whitelist);

    uint64_t now = get_current_time_ns();
    uint32_t src_ip = ip_to_uint32("192.168.1.100");

    // Normal SMB traffic: connect to 1-2 servers (file shares)
    uint32_t file_server = ip_to_uint32("192.168.1.10");
    uint32_t print_server = ip_to_uint32("192.168.1.11");

    TimeWindowEvent event1(now, src_ip, file_server, 49152, 445, 6, 1500);
    TimeWindowEvent event2(now + 1'000'000'000, src_ip, print_server, 49153, 445, 6, 1500);

    extractor.add_event(event1);
    extractor.add_event(event2);

    int32_t smb_diversity = extractor.extract_smb_connection_diversity();

    std::cout << "  📊 Benign SMB Diversity: " << smb_diversity << "\n";

    // Benign: 0-2 SMB destinations
    test.assert_in_range(smb_diversity, 0.0, 2.0,
                         "Benign SMB diversity in range [0, 2]");
}

// ============================================================================
// TEST 7: Feature 3 - SMB Connection Diversity (Malicious Lateral Movement)
// ============================================================================

void test_smb_diversity_malicious() {
    TestRunner test("SMB Diversity - Malicious Lateral Movement");

    FlowTracker flow_tracker(10000, 300);
    DNSAnalyzer dns_analyzer(1000);
    IPWhitelist ip_whitelist(10000, 24 * 3600);

    RansomwareFeatureExtractor extractor(flow_tracker, dns_analyzer, ip_whitelist);

    uint64_t now = get_current_time_ns();
    uint32_t src_ip = ip_to_uint32("192.168.1.100");

    // Ransomware lateral movement: scanning many hosts via SMB
    for (int i = 1; i <= 15; i++) {
        uint32_t target_ip = ip_to_uint32("192.168.1." + std::to_string(i));

        TimeWindowEvent event(
            now + (i * 100'000'000ULL),  // Spread over 1.5 seconds
            src_ip,
            target_ip,
            49152 + i,
            445,  // SMB port
            6,    // TCP
            200   // Small packets (scanning)
        );

        extractor.add_event(event);
    }

    int32_t smb_diversity = extractor.extract_smb_connection_diversity();

    std::cout << "  📊 Malicious SMB Diversity: " << smb_diversity << "\n";

    // Malicious: >5 SMB destinations
    test.assert_true(smb_diversity > 5,
                     "Malicious SMB diversity > 5 (indicates lateral movement)");
}

// ============================================================================
// TEST 8: Integration Test - Full Phase 1A Feature Extraction
// ============================================================================

void test_full_phase1a_extraction() {
    TestRunner test("Full Phase 1A Feature Extraction");

    FlowTracker flow_tracker(10000, 300);
    DNSAnalyzer dns_analyzer(1000);
    IPWhitelist ip_whitelist(10000, 24 * 3600);

    RansomwareFeatureExtractor extractor(flow_tracker, dns_analyzer, ip_whitelist);

    uint64_t now = get_current_time_ns();
    uint32_t src_ip = ip_to_uint32("192.168.1.100");

    // Simulate ransomware behavior: DGA + new IPs + SMB scanning

    // 1. Add DGA queries
    for (int i = 0; i < 10; i++) {
        std::string random_domain = "x" + std::to_string(rand()) + "f8dk.com";
        dns_analyzer.add_query(random_domain, now + (i * 100'000'000ULL), false, src_ip);
    }

    // 2. Add many external IPs
    for (int i = 0; i < 15; i++) {
        uint32_t random_ip = ip_to_uint32("200." + std::to_string(i) + ".50.1");
        ip_whitelist.add_ip(random_ip, now + (i * 100'000'000ULL));
    }

    // 3. Add SMB scanning
    for (int i = 1; i <= 10; i++) {
        uint32_t target_ip = ip_to_uint32("192.168.1." + std::to_string(i));
        TimeWindowEvent event(now + (i * 100'000'000ULL), src_ip, target_ip,
                             49152 + i, 445, 6, 200);
        extractor.add_event(event);
    }

    // Extract all Phase 1A features
    auto features = extractor.extract_features_phase1a();

    std::cout << "\n  📊 EXTRACTED FEATURES:\n";
    std::cout << "    - DNS Query Entropy: " << features.dns_query_entropy << "\n";
    std::cout << "    - New External IPs (30s): " << features.new_external_ips_30s << "\n";
    std::cout << "    - SMB Connection Diversity: " << features.smb_connection_diversity << "\n";

    // Verify all features indicate ransomware
    test.assert_true(features.dns_query_entropy > 6.0,
                     "DNS entropy indicates DGA");
    test.assert_true(features.new_external_ips_30s > 10,
                     "New external IPs indicate C&C");
    test.assert_true(features.smb_connection_diversity > 5,
                     "SMB diversity indicates lateral movement");
    test.assert_true(features.features_valid,
                     "Features marked as valid");
}

// ============================================================================
// TEST 9: Edge Cases
// ============================================================================

void test_edge_cases() {
    TestRunner test("Edge Cases");

    FlowTracker flow_tracker(10000, 300);
    DNSAnalyzer dns_analyzer(1000);
    IPWhitelist ip_whitelist(10000, 24 * 3600);

    RansomwareFeatureExtractor extractor(flow_tracker, dns_analyzer, ip_whitelist);

    // Test 1: Empty data
    auto features_empty = extractor.extract_features_phase1a();
    test.assert_true(features_empty.dns_query_entropy == 0.0f,
                     "Empty data returns 0 entropy");

    // Test 2: Single event
    uint64_t now = get_current_time_ns();
    uint32_t src_ip = ip_to_uint32("192.168.1.100");
    uint32_t dst_ip = ip_to_uint32("192.168.1.10");

    TimeWindowEvent event(now, src_ip, dst_ip, 49152, 445, 6, 1500);
    extractor.add_event(event);

    auto features_single = extractor.extract_features_phase1a();
    test.assert_true(features_single.smb_connection_diversity == 1,
                     "Single SMB event returns diversity of 1");

    // Test 3: Cleanup old events
    extractor.clear();
    test.assert_true(extractor.get_event_count() == 0,
                     "Clear removes all events");

    // Test 4: Window stats with no data
    auto stats = extractor.get_window_stats(now, now + 1'000'000'000);
    test.assert_true(stats.event_count == 0,
                     "Window stats returns 0 events when empty");
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  RANSOMWARE FEATURE EXTRACTOR - UNIT TESTS            ║\n";
    std::cout << "║  Phase 1A: 3 Critical Features                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    int total_failures = 0;

    // Run all tests
    test_basic_initialization();
    test_dns_entropy_benign();
    test_dns_entropy_malicious();
    test_new_external_ips_benign();
    test_new_external_ips_malicious();
    test_smb_diversity_benign();
    test_smb_diversity_malicious();
    test_full_phase1a_extraction();
    test_edge_cases();

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  FINAL RESULTS                                         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    if (total_failures == 0) {
        std::cout << "\n🎉 ALL TESTS PASSED! Phase 1A is ready for production.\n";
        return 0;
    } else {
        std::cout << "\n❌ SOME TESTS FAILED. Review failures above.\n";
        return 1;
    }
}