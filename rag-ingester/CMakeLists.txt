cmake_minimum_required(VERSION 3.20)
project(rag-ingester VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -pedantic")

# Dependencies
find_package(Threads REQUIRED)
find_package(Protobuf REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

# etcd-client (buscar en sistema primero, luego local)
find_library(ETCD_CLIENT_LIB 
    NAMES etcd_client
    PATHS /usr/local/lib /usr/lib /vagrant/etcd-client/build
)
if(ETCD_CLIENT_LIB)
    message(STATUS "✅ Found etcd_client: ${ETCD_CLIENT_LIB}")
else()
    message(FATAL_ERROR "❌ etcd_client not found")
endif()

# crypto-transport (buscar en sistema primero, luego local)
find_library(CRYPTO_TRANSPORT_LIB 
    NAMES crypto_transport
    PATHS /usr/local/lib /usr/lib /vagrant/crypto-transport/build
)
if(CRYPTO_TRANSPORT_LIB)
    message(STATUS "✅ Found crypto_transport: ${CRYPTO_TRANSPORT_LIB}")
else()
    message(FATAL_ERROR "❌ crypto_transport not found")
endif()

# common-rag-ingester (opcional)
find_library(COMMON_RAG_INGESTER_LIB 
    NAMES common-rag-ingester
    PATHS /usr/local/lib /usr/lib /vagrant/common-rag-ingester/build
)
if(COMMON_RAG_INGESTER_LIB)
    message(STATUS "✅ Found common-rag-ingester: ${COMMON_RAG_INGESTER_LIB}")
else()
    message(WARNING "⚠️  common-rag-ingester not found (optional)")
    set(COMMON_RAG_INGESTER_LIB "")
endif()

# FAISS (opcional)
find_package(faiss QUIET)
if(NOT faiss_FOUND)
    message(WARNING "⚠️  FAISS not found - will be disabled")
    add_definitions(-DNO_FAISS)
endif()

# ONNX Runtime (opcional)
find_library(ONNXRUNTIME_LIB onnxruntime 
    PATHS /usr/local/lib /usr/lib
)
if(ONNXRUNTIME_LIB)
    message(STATUS "✅ Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
else()
    message(WARNING "⚠️  ONNX Runtime not found - embedders will be stubs")
    set(ONNXRUNTIME_LIB "")
endif()

# Include directories (sistema primero, luego local)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(/usr/local/include)

# Source files
set(SOURCES
    src/main.cpp
    src/ingester_service.cpp
    src/file_watcher.cpp
    src/event_loader.cpp
    src/common/thread_pool.cpp
    src/common/config_parser.cpp
    src/embedders/chronos_embedder.cpp
    src/embedders/sbert_embedder.cpp
    src/embedders/attack_embedder.cpp
    src/indexers/multi_index_manager.cpp
    src/indexers/index_health_monitor.cpp
    src/indexers/eventual_consistency.cpp
)

# Executable
add_executable(rag-ingester ${SOURCES})

# Link libraries
target_link_libraries(rag-ingester
    PRIVATE
        Threads::Threads
        ${Protobuf_LIBRARIES}
        ${ETCD_CLIENT_LIB}
        ${CRYPTO_TRANSPORT_LIB}
        spdlog::spdlog
        nlohmann_json::nlohmann_json
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

# Install
install(TARGETS rag-ingester DESTINATION bin)
install(DIRECTORY config/ DESTINATION etc/rag-ingester)

# Tests
enable_testing()
add_subdirectory(tests)
