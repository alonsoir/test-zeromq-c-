#pragma once

#include <string>
#include <thread>
#include <chrono>
#include <random>
#include <atomic>
#include <google/protobuf/message.h>

// Forward declarations
namespace protobuf {
    class NetworkFeatures;
    class GeoEnrichment;
    class DistributedNode;
    class NetworkSecurityEvent;
}

// Global variables for signal handling
extern std::atomic<bool> g_running;

// Function declarations for service1
void signalHandler(int signum);
std::string createServiceConfig();
protobuf::NetworkSecurityEvent generateNetworkEvent();

// Helper functions for network event generation
std::string generateRandomIP();
uint32_t generateRandomPort();
void populateNetworkFeatures(protobuf::NetworkFeatures* features);
void populateGeoEnrichment(protobuf::GeoEnrichment* geo, double suspicious_probability);
void populateDistributedNode(protobuf::DistributedNode* node, const std::string& node_id);