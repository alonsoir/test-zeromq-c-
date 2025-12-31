cmake_minimum_required(VERSION 3.16)
project(etcd_client VERSION 1.0.0 LANGUAGES CXX)

# C++20 required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

# Find dependencies
find_package(PkgConfig REQUIRED)

# Find crypto-transport library (our new dependency)
find_library(CRYPTO_TRANSPORT_LIB crypto_transport PATHS /usr/local/lib REQUIRED)
if(NOT CRYPTO_TRANSPORT_LIB)
    message(FATAL_ERROR "crypto-transport library not found. Install it first from /vagrant/crypto-transport")
endif()

# nlohmann/json (header-only, should be available)
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found via find_package, will use system headers")
endif()

# Sources (crypto and compression removed - now in crypto-transport)
set(SOURCES
        src/etcd_client.cpp
        src/config_loader.cpp
        src/http_client.cpp
        src/component_registration.cpp
)

# Create shared library
add_library(etcd_client SHARED ${SOURCES})

# Include directories
target_include_directories(etcd_client
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        /usr/local/include  # For crypto-transport headers
)

# Link libraries
target_link_libraries(etcd_client
        PRIVATE
        ${CRYPTO_TRANSPORT_LIB}
)

# Add nlohmann_json if found
if(nlohmann_json_FOUND)
    target_link_libraries(etcd_client PRIVATE nlohmann_json::nlohmann_json)
endif()

# Set library properties
set_target_properties(etcd_client PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        PUBLIC_HEADER include/etcd_client/etcd_client.hpp
)

# Install rules
install(TARGETS etcd_client
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include/etcd_client
)

# Install all headers
install(DIRECTORY include/etcd_client
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp"
)

# Tests (optional)
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "========================================")
message(STATUS "etcd-client Library Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "crypto-transport: Found")
message(STATUS "  Encryption: ChaCha20-Poly1305")
message(STATUS "  Compression: LZ4")
message(STATUS "========================================")