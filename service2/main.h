#pragma once

#include <ctime>
#include <google/protobuf/message.h>

// Forward declarations
namespace protobuf {
    class NetworkFeatures;
    class GeoEnrichment;
    class DistributedNode;
}

// Function declarations for service2
void displayNetworkFeatures(const protobuf::NetworkFeatures& features);
void displayGeoEnrichment(const protobuf::GeoEnrichment& geo);
void displayDistributedNode(const protobuf::DistributedNode& node);