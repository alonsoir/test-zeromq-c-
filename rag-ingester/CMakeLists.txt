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
find_package(OpenSSL REQUIRED)   # Day 67: CsvEventLoader HMAC-SHA256
# Day 72: trace_id_generator SHA256

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
        ${CMAKE_SOURCE_DIR}/include          # covers include/utils/ (header-only, Day 72)
        ${CMAKE_BINARY_DIR}/proto            # ← CRÍTICO: protobuf copiado aquí
        ${Protobuf_INCLUDE_DIRS}
        ${OPENSSL_INCLUDE_DIR}               # Day 67: CsvEventLoader / Day 69: FirewallCsvEventLoader
        # Day 72: trace_id_generator (openssl/sha.h)
        /usr/local/include
        /vagrant/crypto-transport/include    # ← crypto-transport headers
        /vagrant/common/include              # ← reason_codes.hpp (ADR-002)
)

message(STATUS "✅ protobuf headers: ${CMAKE_BINARY_DIR}/proto")
message(STATUS "✅ crypto-transport headers: /vagrant/crypto-transport/include")
message(STATUS "✅ reason_codes.hpp headers: /vagrant/common/include")
message(STATUS "✅ OpenSSL headers: ${OPENSSL_INCLUDE_DIR}")

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

# FileWatcher library — monitors directory for *.pb files (IN_CLOSE_WRITE)
add_library(file_watcher STATIC src/file_watcher.cpp)
target_include_directories(file_watcher PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(file_watcher PRIVATE Threads::Threads)
target_compile_options(file_watcher PRIVATE -Wall -Wextra -Wpedantic)

# EventLoader library — decrypt + decompress + parse .pb protobuf files
add_library(event_loader STATIC
        src/event_loader.cpp
        ${PROTO_SRCS}
)
target_include_directories(event_loader PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(event_loader
        PRIVATE
        ${CRYPTO_TRANSPORT_LIB}
        ${Protobuf_LIBRARIES}
)
target_compile_options(event_loader PRIVATE -Wall -Wextra -Wpedantic)

# ============================================================================
# Day 67 Components — CSV ingestion pipeline (ml-detector path)
# ============================================================================

# CsvFileWatcher library — inotify tail semantics for append-only CSV
add_library(csv_file_watcher STATIC src/csv_file_watcher.cpp)
target_include_directories(csv_file_watcher PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(csv_file_watcher PRIVATE Threads::Threads)
target_compile_options(csv_file_watcher PRIVATE -Wall -Wextra -Wpedantic)

# CsvEventLoader library — parse 127-column CSV lines + HMAC-SHA256 verification
add_library(csv_event_loader STATIC src/csv_event_loader.cpp)
target_include_directories(csv_event_loader PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(csv_event_loader PRIVATE OpenSSL::Crypto)
target_compile_options(csv_event_loader PRIVATE -Wall -Wextra -Wpedantic)

# ============================================================================
# Day 69 Components — Dual CSV sources
# ============================================================================

# CsvDirWatcher library — inotify on directory for daily-rotating CSV files
add_library(csv_dir_watcher STATIC src/csv_dir_watcher.cpp)
target_include_directories(csv_dir_watcher PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(csv_dir_watcher PRIVATE Threads::Threads)
target_compile_options(csv_dir_watcher PRIVATE -Wall -Wextra -Wpedantic)

# FirewallCsvEventLoader library — parse 7-column firewall CSV + HMAC-SHA256
add_library(firewall_csv_event_loader STATIC src/firewall_csv_event_loader.cpp)
target_include_directories(firewall_csv_event_loader PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(firewall_csv_event_loader PRIVATE OpenSSL::Crypto)
target_compile_options(firewall_csv_event_loader PRIVATE -Wall -Wextra -Wpedantic)

# ============================================================================
# Day 72 Components
# ============================================================================
# trace_id_generator — header-only, no static library needed.
# Located at include/utils/trace_id_generator.hpp.
# Included via #include "utils/trace_id_generator.hpp" in main.cpp.
# Already covered by include_directories(${CMAKE_SOURCE_DIR}/include) above.
# Dependencies: OpenSSL::Crypto (SHA256) + spdlog::spdlog (sentinel warn logs).
# Both are already linked into rag-ingester target below.

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
        src/metadata_db.cpp
        ${PROTO_SRCS}
)

add_executable(rag-ingester ${SOURCES})
target_compile_options(rag-ingester PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(rag-ingester
        PRIVATE
        file_watcher
        event_loader
        csv_file_watcher
        csv_event_loader
        csv_dir_watcher
        firewall_csv_event_loader
        Threads::Threads
        ${Protobuf_LIBRARIES}
        ${ETCD_CLIENT_LIB}
        ${CRYPTO_TRANSPORT_LIB}
        OpenSSL::Crypto
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        SQLite::SQLite3
)

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
# Tests — ALL test targets live in tests/CMakeLists.txt
# ============================================================================
enable_testing()
add_subdirectory(tests)

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
message(STATUS "  etcd-client:      ${ETCD_CLIENT_LIB}")
message(STATUS "  crypto-transport: ${CRYPTO_TRANSPORT_LIB}")
message(STATUS "  protobuf:         ${Protobuf_LIBRARIES}")
message(STATUS "  OpenSSL:          ${OPENSSL_VERSION}")
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
    message(STATUS "  FAISS: DISABLED")
endif()
if(ONNXRUNTIME_LIB)
    message(STATUS "  ONNX Runtime: ENABLED")
else()
    message(STATUS "  ONNX Runtime: DISABLED")
endif()
message(STATUS "")
message(STATUS "Day 40 Components:")
message(STATUS "  EventLoader:   crypto-transport (.pb protobuf)")
message(STATUS "  FileWatcher:   inotify IN_CLOSE_WRITE (*.pb)")
message(STATUS "  MetadataDB:    Producer (write)")
message(STATUS "  FAISS:         Persistence enabled")
message(STATUS "  Features:      101 (61 base + 40 embedded)")
message(STATUS "")
message(STATUS "Day 67 Components:")
message(STATUS "  CsvFileWatcher:  inotify IN_MODIFY + offset (append-only CSV)")
message(STATUS "  CsvEventLoader:  127-col CSV parser + HMAC-SHA256 verification")
message(STATUS "  Feature vector:  62 raw NetworkFeatures (Section 2)")
message(STATUS "  WAL mode:        SQLite WAL required for streaming inserts")
message(STATUS "")
message(STATUS "Day 69 Components:")
message(STATUS "  CsvDirWatcher:          inotify on directory, daily rotation")
message(STATUS "  FirewallCsvEventLoader: 7-col CSV parser + HMAC-SHA256")
message(STATUS "  Source A (ml-detector): CsvDirWatcher → FAISS + MetadataDB INSERT")
message(STATUS "  Source B (firewall):    CsvFileWatcher → MetadataDB UPDATE")
message(STATUS "  Correlation:            provisional by src_ip+ts_window (±5s)")
message(STATUS "                          clean path via trace_id (Day 72)")
message(STATUS "")
message(STATUS "Day 72 Components:")
message(STATUS "  Idempotency guard:      exists() before embed+index (FAISS/DB sync)")
message(STATUS "  trace_id_generator:     header-only, include/utils/")
message(STATUS "                          SHA256(src|dst|attack|bucket) prefix 16B")
message(STATUS "                          O(1), zero-coordination, deterministic")
message(STATUS "                          Sentinel: empty IP → 0.0.0.0 (warn logged)")
message(STATUS "                          Sentinel: empty attack → unknown (warn logged)")
message(STATUS "  MetadataDB schema:      complete from birth (no ALTER TABLE needed)")
message(STATUS "  test_trace_id:          6 test groups incl. edge cases")
message(STATUS "")
message(STATUS "🎯 Single Source of Truth:")
message(STATUS "  Compiler Flags: Controlled by root Makefile via PROFILE")
message(STATUS "========================================")
message(STATUS "")