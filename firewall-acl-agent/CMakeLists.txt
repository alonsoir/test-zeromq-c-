#===----------------------------------------------------------------------===//
# ML Defender - Firewall ACL Agent
# CMakeLists.txt - Build Configuration (Day 50 - Observability Enhanced)
#===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.20)

project(firewall-acl-agent
        VERSION 1.0.0
        DESCRIPTION "ML Defender - High-Performance Firewall ACL Agent (Day 50)"
        LANGUAGES CXX
)

#===----------------------------------------------------------------------===//
# C++ Standard
#===----------------------------------------------------------------------===//

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# COMPILER FLAGS - CONTROLLED BY ROOT MAKEFILE
# ============================================================================
# ⚠️  NO HARDCODED FLAGS - All flags come from root Makefile via CMAKE_CXX_FLAGS
# ⚠️  DO NOT set CMAKE_CXX_FLAGS_RELEASE, CMAKE_CXX_FLAGS_DEBUG here
# ⚠️  DO NOT set -O, -g, -march, -flto, -fsanitize flags here
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

#===----------------------------------------------------------------------===//
# COMPILER OPTIONS - Warnings Only
#===----------------------------------------------------------------------===//
# ⚠️  Optimization flags are controlled by root Makefile
# ⚠️  Only warnings are set here

add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -Wno-unused-parameter
)

#===----------------------------------------------------------------------===//
# Find Required Dependencies
#===----------------------------------------------------------------------===//

find_package(PkgConfig REQUIRED)

pkg_check_modules(ZMQ REQUIRED libzmq)
if(NOT ZMQ_FOUND)
    message(FATAL_ERROR "ZeroMQ not found. Install: sudo apt-get install libzmq3-dev")
endif()

find_package(Protobuf REQUIRED)
if(NOT Protobuf_FOUND)
    message(FATAL_ERROR "Protobuf not found. Install: sudo apt-get install libprotobuf-dev protobuf-compiler")
endif()

find_package(Boost 1.71 REQUIRED COMPONENTS
        system
        thread
        filesystem
)
if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost not found. Install: sudo apt-get install libboost-all-dev")
endif()

pkg_check_modules(JSONCPP REQUIRED jsoncpp)
if(NOT JSONCPP_FOUND)
    message(FATAL_ERROR "jsoncpp not found. Install: sudo apt-get install libjsoncpp-dev")
endif()

find_package(Threads REQUIRED)

# ============================================================================
# Day 50: Backtrace Support for Crash Diagnostics
# ============================================================================
find_library(BACKTRACE_LIB backtrace)
if(BACKTRACE_LIB)
    message(STATUS "✅ Found backtrace library: ${BACKTRACE_LIB}")
    set(HAVE_BACKTRACE TRUE)
else()
    message(STATUS "⚠️  backtrace library not found - using execinfo.h fallback")
    set(HAVE_BACKTRACE FALSE)
endif()

# ============================================================================
# crypto-transport Library
# ============================================================================
find_library(CRYPTO_TRANSPORT_LIB crypto_transport PATHS /usr/local/lib REQUIRED)
if(CRYPTO_TRANSPORT_LIB)
    message(STATUS "✅ Found crypto-transport: ${CRYPTO_TRANSPORT_LIB}")
    include_directories(/usr/local/include)
else()
    message(FATAL_ERROR "❌ crypto-transport not found. Install: cd /vagrant/crypto-transport/build && sudo make install")
endif()

# ============================================================================
# etcd-client Library (Custom)
# ============================================================================
set(ETCD_CLIENT_LIB /vagrant/etcd-client/build/libetcd_client.so)
set(ETCD_CLIENT_INCLUDE_DIR /vagrant/etcd-client/include)

if(EXISTS ${ETCD_CLIENT_LIB})
    message(STATUS "✅ Found etcd-client library: ${ETCD_CLIENT_LIB}")
    include_directories(${ETCD_CLIENT_INCLUDE_DIR})
    set(HAVE_ETCD_CLIENT TRUE)
else()
    message(STATUS "⚠️  etcd-client library not found - will build without etcd integration")
    set(HAVE_ETCD_CLIENT FALSE)
endif()

#===----------------------------------------------------------------------===//
# Protobuf Unificado
#===----------------------------------------------------------------------===//

if(NOT EXISTS "${CMAKE_BINARY_DIR}/proto/network_security.pb.cc")
    message(FATAL_ERROR "❌ Protobuf unificado no encontrado. Ejecuta: cd /vagrant/protobuf && ./generate.sh")
else()
    message(STATUS "✅ Protobuf unificado encontrado: ${CMAKE_BINARY_DIR}/proto/network_security.pb.cc")
endif()

set(PROTO_SOURCES ${CMAKE_BINARY_DIR}/proto/network_security.pb.cc)
set(PROTO_HEADERS ${CMAKE_BINARY_DIR}/proto/network_security.pb.h)

#===----------------------------------------------------------------------===//
# Include Directories (Day 50: Enhanced)
#===----------------------------------------------------------------------===//

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/proto

        ${ZMQ_INCLUDE_DIRS}
        ${Protobuf_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${JSONCPP_INCLUDE_DIRS}
)

#===----------------------------------------------------------------------===//
# Source Files
#===----------------------------------------------------------------------===//

set(FIREWALL_CORE_SOURCES
        src/core/ipset_wrapper.cpp
        src/core/iptables_wrapper.cpp
        src/core/batch_processor.cpp
        src/core/config_loader.cpp
        src/core/etcd_client.cpp
        # src/core/acl_intelligence.cpp     # TODO: Phase 2+
)

set(FIREWALL_API_SOURCES
        src/api/zmq_subscriber.cpp
)

set(FIREWALL_UTIL_SOURCES
        src/core/logger.cpp              # Logger asíncrono + payload storage
        # src/utils/metrics.cpp           # TODO: Phase 2 (unified metrics)
)

set(FIREWALL_MAIN_SOURCE
        src/main.cpp
)

#===----------------------------------------------------------------------===//
# Core Library (Day 50: Enhanced)
#===----------------------------------------------------------------------===//

add_library(firewall_core STATIC
        ${FIREWALL_CORE_SOURCES}
        ${FIREWALL_API_SOURCES}
        ${FIREWALL_UTIL_SOURCES}
        ${PROTO_SOURCES}
)

target_link_libraries(firewall_core
        PUBLIC
        ${Protobuf_LIBRARIES}
        ${ZMQ_LIBRARIES}
        ${Boost_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        Threads::Threads
        ${CRYPTO_TRANSPORT_LIB}
)

# Day 50: Add backtrace library if available
if(HAVE_BACKTRACE)
    target_link_libraries(firewall_core PUBLIC ${BACKTRACE_LIB})
    message(STATUS "✅ Linking backtrace library to firewall_core")
endif()

# etcd-client (custom library)
if(HAVE_ETCD_CLIENT)
    target_link_libraries(firewall_core PUBLIC ${ETCD_CLIENT_LIB})
    message(STATUS "✅ Linking etcd-client library to firewall_core")
endif()

target_include_directories(firewall_core
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/proto
        ${ZMQ_INCLUDE_DIRS}
        ${JSONCPP_INCLUDE_DIRS}
)

#===----------------------------------------------------------------------===//
# Main Executable (Day 50: Enhanced)
#===----------------------------------------------------------------------===//

add_executable(firewall-acl-agent
        ${FIREWALL_MAIN_SOURCE}
)

target_link_libraries(firewall-acl-agent
        PRIVATE
        firewall_core
)

# Day 50: Link additional libraries for crash diagnostics
# Note: execinfo.h is provided by glibc, no additional linking needed on Linux
# backtrace library is already linked via firewall_core

#===----------------------------------------------------------------------===//
# Installation
#===----------------------------------------------------------------------===//

install(TARGETS firewall-acl-agent
        RUNTIME DESTINATION bin
)

install(DIRECTORY config/
        DESTINATION etc/ml-defender/firewall
        FILES_MATCHING PATTERN "*.json"
)

install(FILES systemd/firewall-acl-agent.service
        DESTINATION /etc/systemd/system/
        OPTIONAL
)

#===----------------------------------------------------------------------===//
# Testing
#===----------------------------------------------------------------------===//

option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()

    find_package(GTest)
    if(NOT GTest_FOUND)
        message(STATUS "GTest not found, fetching from GitHub...")
        include(FetchContent)
        FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG release-1.12.1
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    set(TEST_SOURCES
            tests/unit/test_ipset_wrapper.cpp
            tests/unit/test_logger.cpp
    )

    add_executable(firewall_tests
            ${TEST_SOURCES}
    )

    target_link_libraries(firewall_tests
            PRIVATE
            firewall_core
            GTest::gtest
            GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(firewall_tests)

    message(STATUS "✅ Unit tests enabled (including logger tests)")
endif()

#===----------------------------------------------------------------------===//
# Build Summary (Day 50: Enhanced)
#===----------------------------------------------------------------------===//

message(STATUS "")
message(STATUS "╔════════════════════════════════════════════════════════╗")
message(STATUS "║  ML Defender - Firewall ACL Agent Configuration       ║")
message(STATUS "║  Day 50: Comprehensive Observability Integration      ║")
message(STATUS "╚════════════════════════════════════════════════════════╝")
message(STATUS "")
message(STATUS "Version:           ${PROJECT_VERSION}")
message(STATUS "C++ Standard:      C++${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Flags:         ${CMAKE_CXX_FLAGS}")
message(STATUS "")
message(STATUS "✅ Components:")
message(STATUS "  Logger:          IMPLEMENTED (async + payload storage)")
message(STATUS "  Observability:   IMPLEMENTED (Day 50 - microsecond logging)")
message(STATUS "  Crash Diag:      IMPLEMENTED (backtrace + state dumps)")
message(STATUS "  Metrics:         TODO (unified system)")
message(STATUS "  ACL Intelligence: TODO (Phase 2+)")
message(STATUS "")
message(STATUS "🔬 Day 50 Observability:")
message(STATUS "  Headers:         firewall_observability_logger.hpp")
message(STATUS "                   crash_diagnostics.hpp")
message(STATUS "  Log Levels:      DEBUG, INFO, BATCH, IPSET, WARN, ERROR, CRASH")
message(STATUS "  Precision:       Microsecond timestamps")
message(STATUS "  Thread Safety:   Mutex-protected + atomic counters")
message(STATUS "  Crash Handling:  Signal handlers with backtrace")
message(STATUS "  Backtrace Lib:   ${HAVE_BACKTRACE}")
message(STATUS "")
message(STATUS "📊 Logging Output:")
message(STATUS "  Observability:   /vagrant/logs/firewall-acl-agent/firewall_detailed.log")
message(STATUS "  JSON metadata:   /vagrant/logs/blocked/TIMESTAMP.json")
message(STATUS "  Protobuf payload: /vagrant/logs/blocked/TIMESTAMP.proto")
message(STATUS "  Format:          Timestamp-based (unique, sortable)")
message(STATUS "")
message(STATUS "🔗 RAG Integration Ready:")
message(STATUS "  ✅ Structured JSON for vector DB")
message(STATUS "  ✅ Full protobuf for forensic analysis")
message(STATUS "  ✅ Async design (non-blocking)")
message(STATUS "")
message(STATUS "🎯 Single Source of Truth:")
message(STATUS "  Compiler Flags: Controlled by root Makefile via PROFILE")
message(STATUS "")
message(STATUS "⚡ Performance Target: 1M+ packets/sec DROP rate")
message(STATUS "🎯 Design Philosophy: Via Appia Quality")
message(STATUS "🔬 Day 50 Motto: Fiat Lux - Let there be light")
message(STATUS "")
message(STATUS "╚════════════════════════════════════════════════════════╝")
message(STATUS "")