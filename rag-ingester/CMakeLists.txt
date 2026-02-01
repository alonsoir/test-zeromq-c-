cmake_minimum_required(VERSION 3.20)
project(rag-ingester VERSION 0.1.0 LANGUAGES CXX)

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
# ⚠️  DO NOT set CMAKE_CXX_FLAGS_RELEASE, CMAKE_CXX_FLAGS_DEBUG here
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
# Required Dependencies
# ============================================================================
find_package(Threads REQUIRED)
find_package(Protobuf REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(SQLite3 REQUIRED)

# etcd-client (REQUIRED)
find_library(ETCD_CLIENT_LIB
        NAMES etcd_client
        PATHS /usr/local/lib /usr/lib /vagrant/etcd-client/build
        REQUIRED
)
message(STATUS "✅ Found etcd_client: ${ETCD_CLIENT_LIB}")

# crypto-transport (REQUIRED)
find_library(CRYPTO_TRANSPORT_LIB
        NAMES crypto_transport
        PATHS /usr/local/lib /usr/lib /vagrant/crypto-transport/build
        REQUIRED
)
message(STATUS "✅ Found crypto_transport: ${CRYPTO_TRANSPORT_LIB}")

# ============================================================================
# Optional Dependencies
# ============================================================================

# common-rag-ingester (OPTIONAL - PCA dimensionality reduction)
find_library(COMMON_RAG_INGESTER_LIB
        NAMES common-rag-ingester
        PATHS /usr/local/lib /usr/lib /vagrant/common-rag-ingester/build
)
if(COMMON_RAG_INGESTER_LIB)
    message(STATUS "✅ Found common-rag-ingester: ${COMMON_RAG_INGESTER_LIB}")
else()
    message(WARNING "⚠️  common-rag-ingester not found (optional - PCA disabled)")
    set(COMMON_RAG_INGESTER_LIB "")
endif()

# FAISS (OPTIONAL - Day 38)
find_package(faiss QUIET)
if(faiss_FOUND)
    message(STATUS "✅ Found FAISS")
else()
    message(WARNING "⚠️  FAISS not found - indexing disabled (Day 38)")
    add_definitions(-DNO_FAISS)
endif()

# ONNX Runtime (OPTIONAL - Day 37)
find_library(ONNXRUNTIME_LIB onnxruntime
        PATHS /usr/local/lib /usr/lib
)
if(ONNXRUNTIME_LIB)
    message(STATUS "✅ Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
else()
    message(WARNING "⚠️  ONNX Runtime not found - embedders will be stubs (Day 37)")
    set(ONNXRUNTIME_LIB "")
endif()

# ============================================================================
# Include Directories
# ============================================================================
include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/proto          # ← CRÍTICO: protobuf copiado aquí
        ${Protobuf_INCLUDE_DIRS}
        /usr/local/include
        /vagrant/crypto-transport/include  # ← crypto-transport headers
        /vagrant/common/include            # ← reason_codes.hpp (ADR-002)
)

message(STATUS "✅ protobuf headers: ${CMAKE_BINARY_DIR}/proto")
message(STATUS "✅ crypto-transport headers: /vagrant/crypto-transport/include")
message(STATUS "✅ reason_codes.hpp headers: /vagrant/common/include")

# ============================================================================
# Protobuf Files (copiados por Makefile raíz)
# ============================================================================
set(PROTO_SRCS
        ${CMAKE_BINARY_DIR}/proto/network_security.pb.cc
)

set(PROTO_HDRS
        ${CMAKE_BINARY_DIR}/proto/network_security.pb.h
)

# ============================================================================
# Day 36 Components
# ============================================================================

# FileWatcher library
add_library(file_watcher STATIC src/file_watcher.cpp)
target_include_directories(file_watcher PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(file_watcher PRIVATE Threads::Threads)

# Warnings only (no optimization flags - controlled by Makefile)
target_compile_options(file_watcher PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# EventLoader library
add_library(event_loader STATIC
        src/event_loader.cpp
        ${PROTO_SRCS}  # ← Include protobuf sources
)
target_include_directories(event_loader PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(event_loader
        PRIVATE
        ${CRYPTO_TRANSPORT_LIB}
        ${Protobuf_LIBRARIES}
)

# Warnings only (no optimization flags - controlled by Makefile)
target_compile_options(event_loader PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# ============================================================================
# Main Executable
# ============================================================================
set(SOURCES
        src/main.cpp
        src/ingester_service.cpp
        src/common/thread_pool.cpp
        src/common/config_parser.cpp
        src/embedders/chronos_embedder.cpp
        src/embedders/sbert_embedder.cpp
        src/embedders/attack_embedder.cpp
        src/embedders/simple_embedder.cpp
        src/indexers/multi_index_manager.cpp
        src/indexers/index_health_monitor.cpp
        src/indexers/eventual_consistency.cpp
        src/metadata_db.cpp                    # ← DAY 40: Producer metadata
        ${PROTO_SRCS}  # ← Include protobuf sources
)

add_executable(rag-ingester ${SOURCES})

# ============================================================================
# COMPILER OPTIONS - Warnings Only
# ============================================================================
# ⚠️  Optimization flags (-O3, -g, -march, -fsanitize) are controlled by root Makefile
# ⚠️  Only warnings are set here

target_compile_options(rag-ingester PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# ============================================================================
# Link Libraries
# ============================================================================

# Link all required libraries
target_link_libraries(rag-ingester
        PRIVATE
        file_watcher
        event_loader
        Threads::Threads
        ${Protobuf_LIBRARIES}
        ${ETCD_CLIENT_LIB}
        ${CRYPTO_TRANSPORT_LIB}
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        SQLite::SQLite3
)

# Link optional libraries
if(COMMON_RAG_INGESTER_LIB)
    target_link_libraries(rag-ingester PRIVATE ${COMMON_RAG_INGESTER_LIB})
endif()

if(faiss_FOUND)
    target_link_libraries(rag-ingester PRIVATE faiss)
endif()

if(ONNXRUNTIME_LIB)
    target_link_libraries(rag-ingester PRIVATE ${ONNXRUNTIME_LIB})
endif()

# ============================================================================
# Tests
# ============================================================================
enable_testing()

# Test: FileWatcher (Day 36)
add_executable(test_file_watcher tests/test_file_watcher.cpp)
target_link_libraries(test_file_watcher PRIVATE file_watcher)
add_test(NAME FileWatcherTest COMMAND test_file_watcher)

# Test: EventLoader (Day 36)
# add_executable(test_event_loader
#         tests/test_event_loader.cpp
#         ${PROTO_SRCS}  # ← Include protobuf sources
# )
#
# target_link_libraries(test_event_loader
#         PRIVATE
#         event_loader
#         ${Protobuf_LIBRARIES}
# )
# add_test(NAME EventLoaderTest COMMAND test_event_loader)

# ============================================================================
# Installation
# ============================================================================
install(TARGETS rag-ingester DESTINATION bin)
install(DIRECTORY config/ DESTINATION etc/rag-ingester)

# ============================================================================
# Build Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== RAG Ingester Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "")
message(STATUS "Required libraries:")
message(STATUS "  etcd-client: ${ETCD_CLIENT_LIB}")
message(STATUS "  crypto-transport: ${CRYPTO_TRANSPORT_LIB}")
message(STATUS "  protobuf: ${Protobuf_LIBRARIES}")
message(STATUS "")
message(STATUS "Optional libraries:")
if(COMMON_RAG_INGESTER_LIB)
    message(STATUS "  common-rag-ingester: ENABLED")
else()
    message(STATUS "  common-rag-ingester: DISABLED")
endif()
if(faiss_FOUND)
    message(STATUS "  FAISS: ENABLED")
else()
    message(STATUS "  FAISS: DISABLED (Day 38)")
endif()
if(ONNXRUNTIME_LIB)
    message(STATUS "  ONNX Runtime: ENABLED")
else()
    message(STATUS "  ONNX Runtime: DISABLED (Day 37)")
endif()
message(STATUS "")
message(STATUS "Day 40 Status:")
message(STATUS "  EventLoader: crypto-transport ready")
message(STATUS "  FileWatcher: Active")
message(STATUS "  MetadataDB: Producer (write)")
message(STATUS "  FAISS Persistence: Enabled")
message(STATUS "  Features: 101 (61 base + 40 embedded)")
message(STATUS "")
message(STATUS "🎯 Single Source of Truth:")
message(STATUS "  Compiler Flags: Controlled by root Makefile via PROFILE")
message(STATUS "========================================")
message(STATUS "")