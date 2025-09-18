#pragma once

#include <thread>
#include <chrono>
#include <google/protobuf/util/time_util.h>
#include <google/protobuf/message.h>

// Forward declarations
namespace protobuf {
    class NetworkFeatures;
    class GeoEnrichment;
}

// Function declarations
std::string generateRandomIP();
uint32_t generateRandomPort();
void populateNetworkFeatures(protobuf::NetworkFeatures* features);
void populateGeoEnrichment(protobuf::GeoEnrichment* geo);