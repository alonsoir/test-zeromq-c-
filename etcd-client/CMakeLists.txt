cmake_minimum_required(VERSION 3.16)
project(etcd_client VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# C++20 STANDARD
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# COMPILER FLAGS - CONTROLLED BY ROOT MAKEFILE
# ============================================================================
# ⚠️  NO HARDCODED FLAGS - All flags come from root Makefile via CMAKE_CXX_FLAGS
# ⚠️  DO NOT set CMAKE_BUILD_TYPE, CMAKE_CXX_FLAGS_RELEASE here
# ⚠️  DO NOT set -O, -g, -march, -fsanitize flags here
#
# The root Makefile controls all optimization and sanitizer flags via:
#   make PROFILE=production  → -O3 -march=native -DNDEBUG -flto
#   make PROFILE=debug       → -g -O0 -fno-omit-frame-pointer
#   make PROFILE=tsan        → -fsanitize=thread -g -O1
#   make PROFILE=asan        → -fsanitize=address -g -O1
# ============================================================================

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CXX Flags (from Makefile): ${CMAKE_CXX_FLAGS}")

# ============================================================================
# Find Dependencies
# ============================================================================
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

# ============================================================================
# Sources
# ============================================================================
set(SOURCES
        src/etcd_client.cpp
        src/config_loader.cpp
        src/http_client.cpp
        src/component_registration.cpp
)

# ============================================================================
# Library Target
# ============================================================================
add_library(etcd_client SHARED ${SOURCES})

# Include directories
target_include_directories(etcd_client
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        /usr/local/include  # For crypto-transport headers
        ${OPENSSL_INCLUDE_DIR}
)

# ============================================================================
# COMPILER OPTIONS - Warnings Only
# ============================================================================
# ⚠️  Optimization flags (-O3, -g, -march, -fsanitize) are controlled by root Makefile
# ⚠️  Only warnings are set here

target_compile_options(etcd_client
        PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# ============================================================================
# Link Libraries
# ============================================================================
# Find OpenSSL (for HMAC - Day 53)
find_package(OpenSSL REQUIRED)

target_link_libraries(etcd_client
        PRIVATE
        ${CRYPTO_TRANSPORT_LIB}
        OpenSSL::SSL
        OpenSSL::Crypto
)

# Add nlohmann_json if found
if(nlohmann_json_FOUND)
    target_link_libraries(etcd_client PRIVATE nlohmann_json::nlohmann_json)
endif()

# ============================================================================
# Library Properties
# ============================================================================
set_target_properties(etcd_client PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        PUBLIC_HEADER include/etcd_client/etcd_client.hpp
)

# ============================================================================
# Install Rules
# ============================================================================
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

# ============================================================================
# Tests (optional)
# ============================================================================
option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Configuration Summary
# ============================================================================
message(STATUS "========================================")
message(STATUS "etcd-client Library Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "crypto-transport: Found")
message(STATUS "  Encryption: ChaCha20-Poly1305")
message(STATUS "  Compression: LZ4")
message(STATUS "")
message(STATUS "🎯 Single Source of Truth:")
message(STATUS "  Compiler Flags: Controlled by root Makefile via PROFILE")
message(STATUS "========================================")