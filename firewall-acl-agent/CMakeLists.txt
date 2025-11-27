#===----------------------------------------------------------------------===//
# ML Defender - Firewall ACL Agent
# CMakeLists.txt - Build Configuration CORREGIDO
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
# PROTOBUF UNIFICADO - USAR MISMO QUE ml-detector y sniffer
#===----------------------------------------------------------------------===//

# Verificar que los archivos protobuf unificados existen
if(NOT EXISTS "${CMAKE_BINARY_DIR}/proto/network_security.pb.cc")
    message(FATAL_ERROR "❌ Protobuf unificado no encontrado. Ejecuta: cd /vagrant/protobuf && ./generate.sh")
else()
    message(STATUS "✅ Protobuf unificado encontrado: ${CMAKE_BINARY_DIR}/proto/network_security.pb.cc")
endif()

set(PROTO_SOURCES ${CMAKE_BINARY_DIR}/proto/network_security.pb.cc)
set(PROTO_HEADERS ${CMAKE_BINARY_DIR}/proto/network_security.pb.h)

#===----------------------------------------------------------------------===//
# Include Directories - ACTUALIZADO
#===----------------------------------------------------------------------===//

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/proto  # ✅ ARCHIVOS UNIFICADOS (igual que ml-detector)

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
        src/core/config_loader.cpp
        # src/core/acl_intelligence.cpp     # TODO: Next to implement
)

# API layer (to be created)
set(FIREWALL_API_SOURCES
        src/api/zmq_subscriber.cpp
)

# Utilities (to be created)
set(FIREWALL_UTIL_SOURCES
        src/core/config_loader.cpp
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
        ${PROTO_SOURCES}          # ✅ AGREGADO: Protobuf unificado
)

target_link_libraries(firewall_core
        PUBLIC
        # ❌ ELIMINADO: firewall_proto (ya no existe)
        ${Protobuf_LIBRARIES}     # ✅ MOVIDO: Enlazar directamente

        ${ZMQ_LIBRARIES}
        ${Boost_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        Threads::Threads
)

target_include_directories(firewall_core
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/proto  # ✅ Para generated protobuf headers unificados

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
# Build Summary - ACTUALIZADO
#===----------------------------------------------------------------------===//

message(STATUS "")
message(STATUS "╔════════════════════════════════════════════════════════╗")
message(STATUS "║  ML Defender - Firewall ACL Agent Configuration       ║")
message(STATUS "║                ✅ CORREGIDO - Protobuf Unificado      ║")
message(STATUS "╚════════════════════════════════════════════════════════╝")
message(STATUS "")
message(STATUS "Version:           ${PROJECT_VERSION}")
message(STATUS "C++ Standard:      C++${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  ZeroMQ:          ${ZMQ_VERSION}")
message(STATUS "  Protobuf:        ${Protobuf_VERSION} ✅ UNIFICADO")
message(STATUS "  Boost:           ${Boost_VERSION}")
message(STATUS "  jsoncpp:         ${JSONCPP_VERSION}")
message(STATUS "  NOTE: Using system ipset commands (no libipset dependency)")
message(STATUS "")
message(STATUS "🎯 Protobuf Configuration:")
message(STATUS "  ✅ Usando archivos unificados: ${CMAKE_BINARY_DIR}/proto/")
message(STATUS "  ✅ Mismo que ml-detector y sniffer")
message(STATUS "  ✅ Compatibilidad garantizada")
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