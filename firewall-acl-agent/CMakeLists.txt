#===----------------------------------------------------------------------===//
# ML Defender - Firewall ACL Agent
# CMakeLists.txt - Build Configuration ACTUALIZADO con Logger
#===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.20)

project(firewall-acl-agent
        VERSION 1.0.0
        DESCRIPTION "ML Defender - High-Performance Firewall ACL Agent"
        LANGUAGES CXX
)

#===----------------------------------------------------------------------===//
# C++ Standard and Compiler Settings
#===----------------------------------------------------------------------===//

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address,undefined")

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
# Include Directories
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
# Source Files - ACTUALIZADO
#===----------------------------------------------------------------------===//

set(FIREWALL_CORE_SOURCES
        src/core/ipset_wrapper.cpp
        src/core/iptables_wrapper.cpp
        src/core/batch_processor.cpp
        src/core/config_loader.cpp
        # src/core/acl_intelligence.cpp     # TODO: Phase 2+
)

set(FIREWALL_API_SOURCES
        src/api/zmq_subscriber.cpp
)

# ✅ Utilidades con Logger
set(FIREWALL_UTIL_SOURCES
        src/core/logger.cpp              # ✅ Logger asíncrono + payload storage
        # src/utils/metrics.cpp             # TODO: Phase 2 (unified metrics)
)

set(FIREWALL_MAIN_SOURCE
        src/main.cpp
)

#===----------------------------------------------------------------------===//
# Core Library - ACTUALIZADO
#===----------------------------------------------------------------------===//

add_library(firewall_core STATIC
        ${FIREWALL_CORE_SOURCES}
        ${FIREWALL_API_SOURCES}
        ${FIREWALL_UTIL_SOURCES}          # ✅ AÑADIDO
        ${PROTO_SOURCES}
)

target_link_libraries(firewall_core
        PUBLIC
        ${Protobuf_LIBRARIES}
        ${ZMQ_LIBRARIES}
        ${Boost_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        Threads::Threads
)

target_include_directories(firewall_core
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/proto

        ${ZMQ_INCLUDE_DIRS}
        ${JSONCPP_INCLUDE_DIRS}
)

#===----------------------------------------------------------------------===//
# Main Executable
#===----------------------------------------------------------------------===//

add_executable(firewall-acl-agent
        ${FIREWALL_MAIN_SOURCE}
)

target_link_libraries(firewall-acl-agent
        PRIVATE
        firewall_core
)

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
# Testing - ACTUALIZADO
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
            tests/unit/test_logger.cpp        # ✅ NUEVO TEST
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
# Build Summary - ACTUALIZADO
#===----------------------------------------------------------------------===//

message(STATUS "")
message(STATUS "╔════════════════════════════════════════════════════════╗")
message(STATUS "║  ML Defender - Firewall ACL Agent Configuration       ║")
message(STATUS "║           ✅ Logger Asíncrono Implementado            ║")
message(STATUS "╚════════════════════════════════════════════════════════╝")
message(STATUS "")
message(STATUS "Version:           ${PROJECT_VERSION}")
message(STATUS "C++ Standard:      C++${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "✅ Nuevos Componentes:")
message(STATUS "  Logger:          IMPLEMENTADO (async + payload storage)")
message(STATUS "  Metrics:         TODO (unified system)")
message(STATUS "  ACL Intelligence: TODO (Phase 2+)")
message(STATUS "")
message(STATUS "📊 Logging Output:")
message(STATUS "  JSON metadata:   /vagrant/logs/blocked/TIMESTAMP.json")
message(STATUS "  Protobuf payload: /vagrant/logs/blocked/TIMESTAMP.proto")
message(STATUS "  Format:          Timestamp-based (unique, sortable)")
message(STATUS "")
message(STATUS "🔗 RAG Integration Ready:")
message(STATUS "  ✅ Structured JSON for vector DB")
message(STATUS "  ✅ Full protobuf for forensic analysis")
message(STATUS "  ✅ Async design (non-blocking)")
message(STATUS "")
message(STATUS "⚡ Performance Target: 1M+ packets/sec DROP rate")
message(STATUS "🎯 Design Philosophy: Via Appia Quality")
message(STATUS "")
message(STATUS "╚════════════════════════════════════════════════════════╝")
message(STATUS "")