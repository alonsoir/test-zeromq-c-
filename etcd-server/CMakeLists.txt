cmake_minimum_required(VERSION 3.16)
project(etcd-server VERSION 1.0.0)

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
# ⚠️  DO NOT set -O, -g, -march, -fsanitize flags here
#
# The root Makefile controls all optimization and sanitizer flags via:
#   make PROFILE=production  → -O3 -march=native -DNDEBUG -flto
#   make PROFILE=debug       → -g -O0 -fno-omit-frame-pointer
#   make PROFILE=tsan        → -fsanitize=thread -g -O1
#   make PROFILE=asan        → -fsanitize=address -g -O1
# ============================================================================

message(STATUS "========================================")
message(STATUS "etcd-server Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CXX Flags (from Makefile): ${CMAKE_CXX_FLAGS}")

# ============================================================================
# DEPENDENCIES
# ============================================================================
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# ============================================================================
# LIBSODIUM (for crypto_manager.cpp direct usage)
# ============================================================================
pkg_check_modules(LIBSODIUM REQUIRED libsodium)
if(LIBSODIUM_FOUND)
    message(STATUS "✅ libsodium found - Using ChaCha20-Poly1305")
else()
    message(FATAL_ERROR "libsodium not found - Install: sudo apt-get install libsodium-dev")
endif()

# ============================================================================
# LZ4 (for compression_lz4.cpp)
# ============================================================================
pkg_check_modules(LZ4 REQUIRED liblz4)
if(NOT LZ4_FOUND)
    message(FATAL_ERROR "liblz4 not found. Install: sudo apt-get install liblz4-dev")
endif()
message(STATUS "✅ LZ4 found - Compression enabled")

# ============================================================================
# CRYPTO-TRANSPORT LIBRARY (Unified cryptography)
# ============================================================================
# crypto-transport provides: ChaCha20-Poly1305 + utilities
find_library(CRYPTO_TRANSPORT_LIB
        NAMES crypto_transport
        PATHS
        /usr/local/lib
        /vagrant/crypto-transport/build
        NO_DEFAULT_PATH
)

find_path(CRYPTO_TRANSPORT_INCLUDE_DIR
        NAMES crypto_transport/crypto_manager.hpp
        PATHS
        /usr/local/include
        /vagrant/crypto-transport/include
        NO_DEFAULT_PATH
)

if(CRYPTO_TRANSPORT_LIB AND CRYPTO_TRANSPORT_INCLUDE_DIR)
    message(STATUS "✅ Found crypto-transport library: ${CRYPTO_TRANSPORT_LIB}")
    message(STATUS "✅ Found crypto-transport headers: ${CRYPTO_TRANSPORT_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "❌ crypto-transport not found
    Library: ${CRYPTO_TRANSPORT_LIB}
    Headers: ${CRYPTO_TRANSPORT_INCLUDE_DIR}

    Fix: Run 'make crypto-transport-build' from project root")
endif()

# ============================================================================
# INCLUDE DIRECTORIES
# ============================================================================
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CRYPTO_TRANSPORT_INCLUDE_DIR}
        ${LIBSODIUM_INCLUDE_DIRS}
        ${LZ4_INCLUDE_DIRS}
        /usr/local/include
)

# ============================================================================
# SOURCES
# ============================================================================
set(SOURCES
        src/main.cpp
        src/etcd_server.cpp
        src/component_registry.cpp
        src/crypto_manager.cpp
        src/compression_lz4.cpp
)

# ============================================================================
# EXECUTABLE
# ============================================================================
add_executable(etcd-server ${SOURCES})

# ============================================================================
# COMPILER OPTIONS - Warnings Only
# ============================================================================
# ⚠️  Optimization flags (-O2, -O3, -g, -march, -fsanitize)
# ⚠️  are controlled by root Makefile
# ⚠️  Only warnings are set here

target_compile_options(etcd-server
        PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# ============================================================================
# LINK LIBRARIES
# ============================================================================
target_link_libraries(etcd-server
        PRIVATE
        Threads::Threads
        ssl
        crypto
        ${CRYPTO_TRANSPORT_LIB}
        ${LIBSODIUM_LIBRARIES}
        ${LZ4_LIBRARIES}
)

# ============================================================================
# INCLUDE DIRECTORIES (target-specific)
# ============================================================================
target_include_directories(etcd-server
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CRYPTO_TRANSPORT_INCLUDE_DIR}
        ${LIBSODIUM_INCLUDE_DIRS}
        ${LZ4_INCLUDE_DIRS}
)

# ============================================================================
# SUMMARY
# ============================================================================
message(STATUS "========================================")
message(STATUS "Build Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "========================================")
message(STATUS "Dependencies:")
message(STATUS "  crypto-transport: ${CRYPTO_TRANSPORT_LIB}")
message(STATUS "  libsodium: ${LIBSODIUM_LIBRARIES}")
message(STATUS "  LZ4: ${LZ4_LIBRARIES}")
message(STATUS "========================================")
message(STATUS "Sources:")
foreach(src ${SOURCES})
    message(STATUS "  - ${src}")
endforeach()
message(STATUS "========================================")
message(STATUS "🎯 Single Source of Truth:")
message(STATUS "  Compiler Flags: Controlled by root Makefile via PROFILE")
message(STATUS "========================================")