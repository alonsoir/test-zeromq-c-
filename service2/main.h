#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <google/protobuf/message.h>

// Forward declarations
namespace protobuf {
    class NetworkFeatures;
    class GeoEnrichment;
    class DistributedNode;
    class NetworkSecurityEvent;
}

// Estructura para resultados de análisis
struct AnalysisResult {
    double ddos_probability;
    double anomaly_score;
    std::string threat_type;
    std::vector<std::string> suspicious_features;
    bool should_block;
};

// Global variables for signal handling
extern std::atomic<bool> g_running;

// Function declarations for service2
void signalHandler(int signum);
std::string createServiceConfig();
AnalysisResult analyzeNetworkEvent(const protobuf::NetworkSecurityEvent& event);

// Display functions for network analysis
void displayNetworkFeatures(const protobuf::NetworkFeatures& features);
void displayGeoEnrichment(const protobuf::GeoEnrichment& geo);
void displayDistributedNode(const protobuf::DistributedNode& node);