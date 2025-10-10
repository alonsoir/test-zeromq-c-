cmake_minimum_required(VERSION 3.20)
project(cpp_evolutionary_sniffer VERSION 3.1.0 LANGUAGES C CXX)
# sniffer/CMakeLists.txt
# C++20 requirement
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -mtune=native)
        # Enable LTO for maximum performance
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

# Threading support
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# ============================================================================
# DEPENDENCIES DISCOVERY
# ============================================================================

# Required programs
find_program(CLANG clang REQUIRED)
find_program(LLVM_STRIP llvm-strip REQUIRED)
find_program(BPFTOOL bpftool REQUIRED)

# Required packages
find_package(PkgConfig REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)

# Core eBPF and networking dependencies (REQUIRED)
pkg_check_modules(LIBBPF REQUIRED libbpf>=0.8)
pkg_check_modules(ZMQ REQUIRED libzmq>=4.3)
pkg_check_modules(JSONCPP REQUIRED jsoncpp>=1.9)

# Compression libraries (MANDATORY for security sniffer)
pkg_check_modules(LZ4 REQUIRED liblz4>=1.8)
pkg_check_modules(ZSTD REQUIRED libzstd>=1.4)

# Optional compression
pkg_check_modules(SNAPPY libsnappy)

# Optional libraries for advanced features
find_library(CURL_LIBRARY curl)
find_library(NUMA_LIBRARY numa)

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

# Validate critical dependencies
if(NOT LZ4_FOUND)
    message(FATAL_ERROR "❌ LZ4 es obligatorio para el sniffer - ejecuta: sudo apt install liblz4-dev")
endif()

if(NOT ZSTD_FOUND)
    message(FATAL_ERROR "❌ Zstd es obligatorio para el sniffer - ejecuta: sudo apt install libzstd-dev")
endif()

# ============================================================================
# INCLUDE DIRECTORIES
# ============================================================================
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_BINARY_DIR}/proto)

# Core dependencies includes
include_directories(${LIBBPF_INCLUDE_DIRS})
include_directories(${ZMQ_INCLUDE_DIRS})
include_directories(${JSONCPP_INCLUDE_DIRS})

# Compression includes (always present since they're mandatory)
include_directories(${LZ4_INCLUDE_DIRS})
include_directories(${ZSTD_INCLUDE_DIRS})

if(SNAPPY_FOUND)
    include_directories(${SNAPPY_INCLUDE_DIRS})
endif()

# ============================================================================
# eBPF PROGRAM COMPILATION
# ============================================================================

set(BPF_SOURCE ${CMAKE_SOURCE_DIR}/src/kernel/sniffer.bpf.c)
set(BPF_OBJECT ${CMAKE_BINARY_DIR}/sniffer.bpf.o)

add_custom_command(
        OUTPUT ${BPF_OBJECT}
        COMMAND ${CLANG}
        -O2 -target bpf -D__TARGET_ARCH_x86 -g
        -I${LIBBPF_INCLUDE_DIRS}
        -I/usr/include/x86_64-linux-gnu
        -I/usr/include/bpf
        -c ${BPF_SOURCE}
        -o ${BPF_OBJECT}
        DEPENDS ${BPF_SOURCE}
        COMMENT "Compiling eBPF program with BTF support"
)

add_custom_target(bpf_program DEPENDS ${BPF_OBJECT})

# ============================================================================
# PROTOBUF COMPILATION
# ============================================================================

set(PROTO_SOURCE ${CMAKE_SOURCE_DIR}/../protobuf/network_security.proto)
set(PROTO_BINARY_DIR ${CMAKE_BINARY_DIR}/proto)
set(PROTO_SOURCES ${PROTO_BINARY_DIR}/network_security.pb.cc)
set(PROTO_HEADERS ${PROTO_BINARY_DIR}/network_security.pb.h)

file(MAKE_DIRECTORY ${PROTO_BINARY_DIR})

add_custom_command(
        OUTPUT ${PROTO_SOURCES} ${PROTO_HEADERS}
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        --cpp_out=${PROTO_BINARY_DIR}
        --proto_path=${CMAKE_SOURCE_DIR}/../protobuf
        ${PROTO_SOURCE}
        DEPENDS ${PROTO_SOURCE}
        COMMENT "Compiling protobuf from shared schema"
)

add_custom_target(proto_compilation DEPENDS ${PROTO_SOURCES} ${PROTO_HEADERS})

# ============================================================================
# SOURCE FILES CONFIGURATION
# ============================================================================

set(SNIFFER_SOURCES
        src/userspace/main.cpp
        src/userspace/config_manager.cpp
        src/userspace/ebpf_loader.cpp
        src/userspace/ring_consumer.cpp
        src/userspace/zmq_pool_manager.cpp
        src/userspace/thread_manager.cpp
        src/userspace/compression_handler.cpp
        src/userspace/etcd_client.cpp
        src/userspace/flow_manager.cpp
        src/userspace/feature_extractor.cpp
        src/userspace/time_window_manager.cpp
        proto/network_security.pb.cc
)

# Feature-specific sources
set(SNIFFER_FEATURE_SOURCES "")

# Compression handler (always included - mandatory)
list(APPEND SNIFFER_FEATURE_SOURCES src/userspace/compression_handler.cpp)

# ZMQ Pool manager (always included for performance)
list(APPEND SNIFFER_FEATURE_SOURCES src/userspace/zmq_pool_manager.cpp)

# etcd client (optional)
if(CURL_LIBRARY)
    list(APPEND SNIFFER_FEATURE_SOURCES src/userspace/etcd_client.cpp)
endif()

# Combine all sources
# set(SNIFFER_SOURCES ${SNIFFER_CORE_SOURCES} ${SNIFFER_FEATURE_SOURCES})

# ============================================================================
# EXECUTABLE TARGET
# ============================================================================

add_executable(sniffer ${SNIFFER_SOURCES})

add_dependencies(sniffer bpf_program proto_compilation)

# ============================================================================
# LIBRARIES CONFIGURATION
# ============================================================================

# Core libraries (always linked)
set(SNIFFER_CORE_LIBRARIES
        ${LIBBPF_LIBRARIES}
        ${ZMQ_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        ${Protobuf_LIBRARIES}
        ${LZ4_LIBRARIES}
        ${ZSTD_LIBRARIES}
        Threads::Threads
        elf
        z
)

# Optional libraries
set(SNIFFER_OPTIONAL_LIBRARIES "")

if(SNAPPY_FOUND)
    list(APPEND SNIFFER_OPTIONAL_LIBRARIES ${SNAPPY_LIBRARIES})
endif()

if(CURL_LIBRARY)
    list(APPEND SNIFFER_OPTIONAL_LIBRARIES ${CURL_LIBRARY})
endif()

if(NUMA_LIBRARY)
    list(APPEND SNIFFER_OPTIONAL_LIBRARIES ${NUMA_LIBRARY})
endif()

# Link all libraries
target_link_libraries(sniffer ${SNIFFER_CORE_LIBRARIES} ${SNIFFER_OPTIONAL_LIBRARIES})

# ============================================================================
# LINK DIRECTORIES
# ============================================================================

set(SNIFFER_LINK_DIRECTORIES
        ${LIBBPF_LIBRARY_DIRS}
        ${ZMQ_LIBRARY_DIRS}
        ${JSONCPP_LIBRARY_DIRS}
        ${LZ4_LIBRARY_DIRS}
        ${ZSTD_LIBRARY_DIRS}
)

if(SNAPPY_FOUND)
    list(APPEND SNIFFER_LINK_DIRECTORIES ${SNAPPY_LIBRARY_DIRS})
endif()

target_link_directories(sniffer PRIVATE ${SNIFFER_LINK_DIRECTORIES})

# ============================================================================
# COMPILE OPTIONS
# ============================================================================

set(SNIFFER_COMPILE_OPTIONS
        ${LIBBPF_CFLAGS_OTHER}
        ${ZMQ_CFLAGS_OTHER}
        ${JSONCPP_CFLAGS_OTHER}
        ${LZ4_CFLAGS_OTHER}
        ${ZSTD_CFLAGS_OTHER}
)

if(SNAPPY_FOUND)
    list(APPEND SNIFFER_COMPILE_OPTIONS ${SNAPPY_CFLAGS_OTHER})
endif()

target_compile_options(sniffer PRIVATE ${SNIFFER_COMPILE_OPTIONS})

# ============================================================================
# COMPILE DEFINITIONS
# ============================================================================

# Core feature definitions
target_compile_definitions(sniffer PRIVATE
        COMPRESSION_SUPPORT     # Always enabled
        HAVE_LZ4               # Always available
        HAVE_ZSTD              # Always available
)

# Optional feature definitions
if(SNAPPY_FOUND)
    target_compile_definitions(sniffer PRIVATE HAVE_SNAPPY)
endif()

if(CURL_LIBRARY)
    target_compile_definitions(sniffer PRIVATE ETCD_SUPPORT)
endif()

if(NUMA_LIBRARY)
    target_compile_definitions(sniffer PRIVATE NUMA_SUPPORT)
endif()

# ============================================================================
# PERFORMANCE OPTIMIZATIONS
# ============================================================================

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(sniffer PRIVATE NDEBUG)

    # Enable specific CPU optimizations if supported
    include(CheckCXXCompilerFlag)

    # Check for AVX2 support
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(sniffer PRIVATE -mavx2)
    endif()

    # Enable fast math for statistical calculations
    check_cxx_compiler_flag("-ffast-math" COMPILER_SUPPORTS_FAST_MATH)
    if(COMPILER_SUPPORTS_FAST_MATH)
        target_compile_options(sniffer PRIVATE -ffast-math)
    endif()
endif()

# ============================================================================
# DIRECTORY STRUCTURE SETUP
# ============================================================================

# Create necessary directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/config)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/config/features)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/logs)

# Copy configuration files
configure_file(
        ${CMAKE_SOURCE_DIR}/config/sniffer-proposal.json
        ${CMAKE_BINARY_DIR}/config/sniffer.json
        COPYONLY
)

# Create feature group placeholder files
set(FEATURE_GROUPS ddos_83 ransomware_83 rf_23 internal_4)
foreach(GROUP ${FEATURE_GROUPS})
    file(WRITE ${CMAKE_BINARY_DIR}/config/features/${GROUP}_features.json "[]")
endforeach()

# ============================================================================
# INSTALL RULES
# ============================================================================

install(TARGETS sniffer DESTINATION bin)
install(FILES ${BPF_OBJECT} DESTINATION lib/sniffer)
install(FILES ${CMAKE_BINARY_DIR}/config/sniffer.json DESTINATION etc/sniffer)

# ============================================================================
# BUILD SUMMARY
# ============================================================================

message(STATUS "")
message(STATUS "=== ⚡ Enhanced Sniffer Configuration ===")
message(STATUS "📋 Build Info:")
message(STATUS "   Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "   C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "   Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "   LTO enabled: ${CMAKE_INTERPROCEDURAL_OPTIMIZATION}")

message(STATUS "")
message(STATUS "🔧 Core Dependencies:")
message(STATUS "   libbpf: ${LIBBPF_VERSION}")
message(STATUS "   ZeroMQ: ${ZMQ_VERSION}")
message(STATUS "   jsoncpp: ${JSONCPP_VERSION}")
message(STATUS "   Protobuf: ${Protobuf_VERSION}")

message(STATUS "")
message(STATUS "🗜️ Compression Support (MANDATORY):")
message(STATUS "   ✅ LZ4: ${LZ4_VERSION} (required)")
message(STATUS "   ✅ Zstandard: ${ZSTD_VERSION} (required)")
if(SNAPPY_FOUND)
    message(STATUS "   ✅ Snappy: available (optional)")
else()
    message(STATUS "   ⚪ Snappy: not available (optional)")
endif()

message(STATUS "")
message(STATUS "🚀 Optional Features:")
if(CURL_LIBRARY)
    message(STATUS "   ✅ etcd client: enabled")
else()
    message(STATUS "   ⚪ etcd client: disabled (libcurl not found)")
endif()

if(NUMA_LIBRARY)
    message(STATUS "   ✅ NUMA optimization: enabled")
else()
    message(STATUS "   ⚪ NUMA optimization: disabled (libnuma not found)")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(COMPILER_SUPPORTS_AVX2)
        message(STATUS "   ✅ AVX2 optimizations: enabled")
    else()
        message(STATUS "   ⚪ AVX2 optimizations: not supported")
    endif()

    if(COMPILER_SUPPORTS_FAST_MATH)
        message(STATUS "   ✅ Fast math: enabled")
    else()
        message(STATUS "   ⚪ Fast math: not supported")
    endif()
endif()

message(STATUS "")
message(STATUS "📦 Build Artifacts:")
message(STATUS "   Binary: ${CMAKE_BINARY_DIR}/sniffer")
message(STATUS "   eBPF program: ${BPF_OBJECT}")
message(STATUS "   Configuration: ${CMAKE_BINARY_DIR}/config/sniffer.json")

message(STATUS "")
message(STATUS "🎯 Sniffer Capabilities:")
message(STATUS "   ✅ Multi-threading support")
message(STATUS "   ✅ eBPF/XDP high-performance packet capture")
message(STATUS "   ✅ Mandatory LZ4/Zstd compression")
message(STATUS "   ✅ Protobuf serialization")
message(STATUS "   ✅ ZeroMQ communication")
message(STATUS "   🔐 Encryption ready (via etcd tokens)")
message(STATUS "========================================")
message(STATUS "")