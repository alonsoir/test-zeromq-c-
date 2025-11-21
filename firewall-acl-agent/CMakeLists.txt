#===----------------------------------------------------------------------===//
# ML Defender - Firewall ACL Agent
# CMakeLists.txt - Build Configuration
#
# Target: Ultra-high performance packet DROP agent
# Performance: 1M+ packets/sec on commodity hardware
#
# Dependencies:
#   - libipset (kernel interface)
#   - iptables (static rules setup)
#   - ZMQ (detection stream from ml-detector)
#   - Protobuf (message parsing)
#   - Boost (lock-free queues)
#   - jsoncpp (configuration)
#
# Via Appia Quality: Methodical build system
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

# Optimization flags for extreme performance
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address,undefined")

# Warning flags
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

# ZeroMQ - High-performance messaging
pkg_check_modules(ZMQ REQUIRED libzmq)
if(NOT ZMQ_FOUND)
    message(FATAL_ERROR "ZeroMQ not found. Install: sudo apt-get install libzmq3-dev")
endif()

# NOTE: We do NOT need libipset-dev - we use system ipset commands
# This is simpler, more maintainable, and benefits from ipset optimizations

# Protobuf - Message serialization
find_package(Protobuf REQUIRED)
if(NOT Protobuf_FOUND)
    message(FATAL_ERROR "Protobuf not found. Install: sudo apt-get install libprotobuf-dev protobuf-compiler")
endif()

# Boost - Lock-free data structures
find_package(Boost 1.71 REQUIRED COMPONENTS
        system
        thread
        filesystem
)
if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost not found. Install: sudo apt-get install libboost-all-dev")
endif()

# jsoncpp - Configuration parsing
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
if(NOT JSONCPP_FOUND)
    message(FATAL_ERROR "jsoncpp not found. Install: sudo apt-get install libjsoncpp-dev")
endif()

# Threads
find_package(Threads REQUIRED)

#===----------------------------------------------------------------------===//
# Protobuf Generation - Shared Schema
#===----------------------------------------------------------------------===//

# Path to shared protobuf schema
# TODO: Update this path to point to the real shared protobuf once project structure is set
# Real path should be: ../../protobuf/network_security.proto (shared between all components)
# For now, using local protobuf directory for development
set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../protobuf)
set(PROTO_FILE ${PROTO_DIR}/network_security.proto)

# Verify proto file exists
if(NOT EXISTS ${PROTO_FILE})
    message(FATAL_ERROR
            "Protobuf schema not found: ${PROTO_FILE}\n"
            "Expected structure: ml-defender/protobuf/network_security.proto"
    )
endif()

# Generate C++ files from .proto (into build directory)
protobuf_generate_cpp(PROTO_SOURCES PROTO_HEADERS ${PROTO_FILE})

message(STATUS "Protobuf schema: ${PROTO_FILE}")
message(STATUS "Generated sources: ${PROTO_SOURCES}")
message(STATUS "Generated headers: ${PROTO_HEADERS}")

# Create protobuf library
add_library(firewall_proto STATIC
        ${PROTO_SOURCES}
        ${PROTO_HEADERS}
)

target_link_libraries(firewall_proto
        PUBLIC
        ${Protobuf_LIBRARIES}
)

target_include_directories(firewall_proto
        PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}  # Generated protobuf headers
        ${PROTO_DIR}                 # Original .proto location
)

#===----------------------------------------------------------------------===//
# Include Directories
#===----------------------------------------------------------------------===//

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}      # Generated protobuf headers

        ${ZMQ_INCLUDE_DIRS}
        ${Protobuf_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${JSONCPP_INCLUDE_DIRS}
)

#===----------------------------------------------------------------------===//
# Source Files (Only files that exist)
#===----------------------------------------------------------------------===//

# Core components (created so far)
set(FIREWALL_CORE_SOURCES
        src/core/ipset_wrapper.cpp
        src/core/iptables_wrapper.cpp
        src/core/batch_processor.cpp
        # src/core/acl_intelligence.cpp     # TODO: Next to implement
)

# API layer (to be created)
set(FIREWALL_API_SOURCES
        src/api/zmq_subscriber.cpp
)

# Utilities (to be created)
set(FIREWALL_UTIL_SOURCES
        # src/utils/config_loader.cpp       # TODO: Next to implement
        # src/utils/logger.cpp              # TODO: Next to implement
        # src/utils/metrics.cpp             # TODO: Next to implement
)

# Main executable (to be created)
set(FIREWALL_MAIN_SOURCE
        src/main.cpp
)

#===----------------------------------------------------------------------===//
# Core Library (for testing) - Only with existing sources
#===----------------------------------------------------------------------===//

add_library(firewall_core STATIC
        ${FIREWALL_CORE_SOURCES}
        ${FIREWALL_API_SOURCES}   # Uncomment when implemented
        # ${FIREWALL_UTIL_SOURCES}  # Uncomment when implemented
)

target_link_libraries(firewall_core
        PUBLIC
        firewall_proto

        ${ZMQ_LIBRARIES}
        ${Boost_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        Threads::Threads
)

target_include_directories(firewall_core
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}  # For generated protobuf headers

        ${ZMQ_INCLUDE_DIRS}
        ${JSONCPP_INCLUDE_DIRS}
)

#===----------------------------------------------------------------------===//
# Main Executable (TODO: Uncomment when main.cpp is ready)
#===----------------------------------------------------------------------===//

add_executable(firewall-acl-agent
     ${FIREWALL_MAIN_SOURCE}
 )
#
target_link_libraries(firewall-acl-agent
     PRIVATE
         firewall_core
 )

# NOTE: Executable will be created once we have src/main.cpp
message(STATUS "⚠️  Main executable disabled - waiting for src/main.cpp")
message(STATUS "   Current focus: Core library and unit tests")

#===----------------------------------------------------------------------===//
# Installation (Disabled until executable is ready)
#===----------------------------------------------------------------------===//

# TODO: Uncomment when firewall-acl-agent executable exists
install(TARGETS firewall-acl-agent
     RUNTIME DESTINATION bin
)
#
install(DIRECTORY config/
     DESTINATION etc/ml-defender/firewall
     FILES_MATCHING PATTERN "*.json"
)
#
install(FILES systemd/firewall-acl-agent.service
     DESTINATION /etc/systemd/system/
     OPTIONAL
)

message(STATUS "📦 Installation targets disabled - waiting for main executable")

#===----------------------------------------------------------------------===//
# Testing
#===----------------------------------------------------------------------===//

option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()

    # Google Test
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

    # Test sources (only existing tests)
    set(TEST_SOURCES
            tests/unit/test_ipset_wrapper.cpp
            # tests/test_batch_processor.cpp     # TODO: Create when batch_processor is ready
            # tests/test_acl_intelligence.cpp    # TODO: Create when acl_intelligence is ready
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

    # Discover tests
    include(GoogleTest)
    gtest_discover_tests(firewall_tests)

    message(STATUS "✅ Unit tests enabled")
    message(STATUS "   Run: sudo ./firewall_tests  (requires root for ipset operations)")
endif()

#===----------------------------------------------------------------------===//
# Benchmarks
#===----------------------------------------------------------------------===//

option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)

if(BUILD_BENCHMARKS)
    # Google Benchmark
    find_package(benchmark)
    if(NOT benchmark_FOUND)
        message(STATUS "Google Benchmark not found, fetching...")
        include(FetchContent)
        FetchContent_Declare(
                benchmark
                GIT_REPOSITORY https://github.com/google/benchmark.git
                GIT_TAG v1.8.3
        )
        FetchContent_MakeAvailable(benchmark)
    endif()

    add_executable(firewall_bench
            benchmarks/bench_batch_operations.cpp
            benchmarks/bench_queue_throughput.cpp
    )

    target_link_libraries(firewall_bench
            PRIVATE
            firewall_core
            benchmark::benchmark
            benchmark::benchmark_main
    )

    message(STATUS "✅ Benchmarks enabled")
    message(STATUS "   Run: sudo ./firewall_bench  (requires root)")
endif()

#===----------------------------------------------------------------------===//
# Documentation
#===----------------------------------------------------------------------===//

option(BUILD_DOCS "Build documentation with Doxygen" OFF)

if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

        add_custom_target(docs
                COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Generating API documentation with Doxygen"
                VERBATIM
        )

        message(STATUS "✅ Documentation enabled: make docs")
    else()
        message(WARNING "Doxygen not found, documentation disabled")
    endif()
endif()

#===----------------------------------------------------------------------===//
# Performance Profiling Support
#===----------------------------------------------------------------------===//

option(ENABLE_PROFILING "Enable profiling with gprof/perf" OFF)

if(ENABLE_PROFILING)
    add_compile_options(-pg -fno-omit-frame-pointer)
    add_link_options(-pg)
    message(STATUS "✅ Profiling enabled (-pg)")
endif()

#===----------------------------------------------------------------------===//
# Sanitizers (Development)
#===----------------------------------------------------------------------===//

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
    message(STATUS "✅ AddressSanitizer enabled")
endif()

if(ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
    message(STATUS "✅ ThreadSanitizer enabled")
endif()

if(ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
    message(STATUS "✅ UndefinedBehaviorSanitizer enabled")
endif()

#===----------------------------------------------------------------------===//
# Build Summary
#===----------------------------------------------------------------------===//

message(STATUS "")
message(STATUS "╔════════════════════════════════════════════════════════╗")
message(STATUS "║  ML Defender - Firewall ACL Agent Configuration       ║")
message(STATUS "╚════════════════════════════════════════════════════════╝")
message(STATUS "")
message(STATUS "Version:           ${PROJECT_VERSION}")
message(STATUS "C++ Standard:      C++${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  ZeroMQ:          ${ZMQ_VERSION}")
message(STATUS "  Protobuf:        ${Protobuf_VERSION}")
message(STATUS "  Boost:           ${Boost_VERSION}")
message(STATUS "  jsoncpp:         ${JSONCPP_VERSION}")
message(STATUS "  NOTE: Using system ipset commands (no libipset dependency)")
message(STATUS "")
message(STATUS "Optional Features:")
message(STATUS "  Tests:           ${BUILD_TESTS}")
message(STATUS "  Benchmarks:      ${BUILD_BENCHMARKS}")
message(STATUS "  Documentation:   ${BUILD_DOCS}")
message(STATUS "  Profiling:       ${ENABLE_PROFILING}")
message(STATUS "")
message(STATUS "⚡ Performance Target: 1M+ packets/sec DROP rate")
message(STATUS "🎯 Design Philosophy: Via Appia Quality")
message(STATUS "")
message(STATUS "Build Commands:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake -DCMAKE_BUILD_TYPE=Release ..")
message(STATUS "  make -j$(nproc)")
message(STATUS "  sudo ./firewall-acl-agent -c ../config/firewall.json")
message(STATUS "")
message(STATUS "╚════════════════════════════════════════════════════════╝")
message(STATUS "")