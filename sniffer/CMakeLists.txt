# sniffer/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(cpp_evolutionary_sniffer VERSION 3.1.0 LANGUAGES C CXX)

# C++20 requirement
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -mtune=native)
        # Enable LTO for maximum performance
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -mtune=native)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

# Threading support
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# Find required programs
find_program(CLANG clang REQUIRED)
find_program(LLVM_STRIP llvm-strip REQUIRED)
find_program(BPFTOOL bpftool REQUIRED)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)

# Find core eBPF dependencies
pkg_check_modules(LIBBPF REQUIRED libbpf>=0.8)

# Find ZeroMQ
pkg_check_modules(ZMQ REQUIRED libzmq>=4.3)

# Find JSON library
pkg_check_modules(JSONCPP REQUIRED jsoncpp>=1.9)

# Find compression libraries
pkg_check_modules(LZ4 REQUIRED liblz4>=1.8)
pkg_check_modules(ZSTD libzstd>=1.4)

# Optional: Find additional compression libraries
pkg_check_modules(SNAPPY libsnappy)

# Optional: Find etcd client libraries
find_library(CURL_LIBRARY curl)

# Optional: Find NUMA library for NUMA-aware allocation
find_library(NUMA_LIBRARY numa)

# Configuration status
message(STATUS "")
message(STATUS "=== Enhanced Sniffer Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# Core dependencies
message(STATUS "Core dependencies:")
message(STATUS "  libbpf version: ${LIBBPF_VERSION}")
message(STATUS "  ZeroMQ version: ${ZMQ_VERSION}")
message(STATUS "  jsoncpp version: ${JSONCPP_VERSION}")
message(STATUS "  Protobuf version: ${Protobuf_VERSION}")

# Compression libraries
message(STATUS "Compression support:")
message(STATUS "  LZ4: ${LZ4_VERSION} (required)")
if(ZSTD_FOUND)
    message(STATUS "  Zstandard: ${ZSTD_VERSION} (optional)")
else()
    message(STATUS "  Zstandard: NOT FOUND (optional)")
endif()
if(SNAPPY_FOUND)
    message(STATUS "  Snappy: found (optional)")
else()
    message(STATUS "  Snappy: NOT FOUND (optional)")
endif()

# Optional libraries
message(STATUS "Optional libraries:")
if(CURL_LIBRARY)
    message(STATUS "  libcurl: found (for etcd client)")
else()
    message(STATUS "  libcurl: NOT FOUND (etcd client disabled)")
endif()
if(NUMA_LIBRARY)
    message(STATUS "  libnuma: found (NUMA-aware allocation)")
else()
    message(STATUS "  libnuma: NOT FOUND (no NUMA optimization)")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_BINARY_DIR}/proto)
include_directories(${LIBBPF_INCLUDE_DIRS})
include_directories(${ZMQ_INCLUDE_DIRS})
include_directories(${JSONCPP_INCLUDE_DIRS})
include_directories(${LZ4_INCLUDE_DIRS})

if(ZSTD_FOUND)
    include_directories(${ZSTD_INCLUDE_DIRS})
endif()

# eBPF compilation with proper BTF support
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

# Protobuf compilation - using shared schema
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

# Enhanced sniffer executable with new components
set(SNIFFER_SOURCES
        src/userspace/main.cpp
        src/userspace/config_manager.cpp
        src/userspace/ebpf_loader.cpp
        src/userspace/ring_consumer.cpp
        ${PROTO_SOURCES}
)

# Optional sources based on available libraries
set(OPTIONAL_SOURCES "")

# Add compression handler if we have compression libraries
if(LZ4_FOUND OR ZSTD_FOUND)
    list(APPEND OPTIONAL_SOURCES src/userspace/compression_handler.cpp)
    add_definitions(-DCOMPRESSION_SUPPORT)
endif()

# Add etcd client if we have curl
if(CURL_LIBRARY)
    list(APPEND OPTIONAL_SOURCES src/userspace/etcd_client.cpp)
    add_definitions(-DETCD_SUPPORT)
endif()

# Add thread manager for advanced threading
list(APPEND OPTIONAL_SOURCES src/userspace/thread_manager.cpp)

# Add ZMQ pool manager for advanced ZeroMQ features
list(APPEND OPTIONAL_SOURCES src/userspace/zmq_pool_manager.cpp)

# Combine all sources
list(APPEND SNIFFER_SOURCES ${OPTIONAL_SOURCES})

add_executable(sniffer ${SNIFFER_SOURCES})

add_dependencies(sniffer bpf_program proto_compilation)

# Base libraries
set(SNIFFER_LIBRARIES
        ${LIBBPF_LIBRARIES}
        ${ZMQ_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        ${Protobuf_LIBRARIES}
        ${LZ4_LIBRARIES}
        Threads::Threads
        elf
        z
)

# Optional libraries
if(ZSTD_FOUND)
    list(APPEND SNIFFER_LIBRARIES ${ZSTD_LIBRARIES})
endif()

if(SNAPPY_FOUND)
    list(APPEND SNIFFER_LIBRARIES ${SNAPPY_LIBRARIES})
endif()

if(CURL_LIBRARY)
    list(APPEND SNIFFER_LIBRARIES ${CURL_LIBRARY})
endif()

if(NUMA_LIBRARY)
    list(APPEND SNIFFER_LIBRARIES ${NUMA_LIBRARY})
    add_definitions(-DNUMA_SUPPORT)
endif()

target_link_libraries(sniffer ${SNIFFER_LIBRARIES})

# Link directories
set(LINK_DIRECTORIES
        ${LIBBPF_LIBRARY_DIRS}
        ${ZMQ_LIBRARY_DIRS}
        ${JSONCPP_LIBRARY_DIRS}
        ${LZ4_LIBRARY_DIRS}
)

if(ZSTD_FOUND)
    list(APPEND LINK_DIRECTORIES ${ZSTD_LIBRARY_DIRS})
endif()

target_link_directories(sniffer PRIVATE ${LINK_DIRECTORIES})

# Compile options
set(COMPILE_OPTIONS
        ${LIBBPF_CFLAGS_OTHER}
        ${ZMQ_CFLAGS_OTHER}
        ${JSONCPP_CFLAGS_OTHER}
        ${LZ4_CFLAGS_OTHER}
)

if(ZSTD_FOUND)
    list(APPEND COMPILE_OPTIONS ${ZSTD_CFLAGS_OTHER})
endif()

target_compile_options(sniffer PRIVATE ${COMPILE_OPTIONS})

# Feature definitions based on available libraries
if(LZ4_FOUND)
    target_compile_definitions(sniffer PRIVATE HAVE_LZ4)
endif()

if(ZSTD_FOUND)
    target_compile_definitions(sniffer PRIVATE HAVE_ZSTD)
endif()

if(SNAPPY_FOUND)
    target_compile_definitions(sniffer PRIVATE HAVE_SNAPPY)
endif()

# Performance optimization flags for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(sniffer PRIVATE NDEBUG)

    # Enable specific CPU optimizations if supported
    include(CheckCXXCompilerFlag)

    # Check for AVX2 support
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(sniffer PRIVATE -mavx2)
        message(STATUS "AVX2 optimizations enabled")
    endif()

    # Enable fast math for statistical calculations
    check_cxx_compiler_flag("-ffast-math" COMPILER_SUPPORTS_FAST_MATH)
    if(COMPILER_SUPPORTS_FAST_MATH)
        target_compile_options(sniffer PRIVATE -ffast-math)
        message(STATUS "Fast math optimizations enabled")
    endif()
endif()

# Create config directory structure
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/config)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/config/features)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/logs)

# Copy configuration files
configure_file(
        ${CMAKE_SOURCE_DIR}/config/sniffer-proposal.json
        ${CMAKE_BINARY_DIR}/config/sniffer.json
        COPYONLY
)

# Install rules (for future packaging)
install(TARGETS sniffer DESTINATION bin)
install(FILES ${BPF_OBJECT} DESTINATION lib/sniffer)
install(FILES ${CMAKE_BINARY_DIR}/config/sniffer.json DESTINATION etc/sniffer)

# Create feature group placeholder files
set(FEATURE_GROUPS ddos_83 ransomware_83 rf_23 internal_4)
foreach(GROUP ${FEATURE_GROUPS})
    file(WRITE ${CMAKE_BINARY_DIR}/config/features/${GROUP}_features.json "[]")
endforeach()

# Summary with capabilities
message(STATUS "")
message(STATUS "=== Build Configuration Summary ===")
message(STATUS "Sniffer capabilities:")
message(STATUS "  Multi-threading: YES")
message(STATUS "  eBPF/XDP: YES")
message(STATUS "  LZ4 compression: ${LZ4_FOUND}")
message(STATUS "  Zstd compression: ${ZSTD_FOUND}")
message(STATUS "  Snappy compression: ${SNAPPY_FOUND}")
message(STATUS "  etcd client: ${CURL_LIBRARY}")
message(STATUS "  NUMA optimization: ${NUMA_LIBRARY}")
message(STATUS "  Profile-guided optimization: ${CMAKE_INTERPROCEDURAL_OPTIMIZATION}")
message(STATUS "")
message(STATUS "Build artifacts:")
message(STATUS "  Binary: ${CMAKE_BINARY_DIR}/sniffer")
message(STATUS "  eBPF program: ${BPF_OBJECT}")
message(STATUS "  Config: ${CMAKE_BINARY_DIR}/config/sniffer.json")
message(STATUS "====================================")
message(STATUS "")