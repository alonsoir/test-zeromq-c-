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
        add_compile_options(-O3 -march=native)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native)
    endif()
endif()

# Find required programs
find_program(CLANG clang REQUIRED)
find_program(LLVM_STRIP llvm-strip REQUIRED)
find_program(BPFTOOL bpftool REQUIRED)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)

# Find libbpf
pkg_check_modules(LIBBPF REQUIRED libbpf>=0.8)

# Find ZeroMQ
pkg_check_modules(ZMQ REQUIRED libzmq>=4.3)

# Find jsoncpp
pkg_check_modules(JSONCPP REQUIRED jsoncpp>=1.9)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_BINARY_DIR}/proto)
include_directories(${LIBBPF_INCLUDE_DIRS})
include_directories(${ZMQ_INCLUDE_DIRS})
include_directories(${JSONCPP_INCLUDE_DIRS})

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

# Protobuf compilation
set(PROTO_SOURCE ${CMAKE_SOURCE_DIR}/src/proto/network_security_v31.proto)
set(PROTO_BINARY_DIR ${CMAKE_BINARY_DIR}/proto)
set(PROTO_SOURCES ${PROTO_BINARY_DIR}/network_security_v31.pb.cc)
set(PROTO_HEADERS ${PROTO_BINARY_DIR}/network_security_v31.pb.h)

file(MAKE_DIRECTORY ${PROTO_BINARY_DIR})

add_custom_command(
        OUTPUT ${PROTO_SOURCES} ${PROTO_HEADERS}
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        --cpp_out=${PROTO_BINARY_DIR}
        --proto_path=${CMAKE_SOURCE_DIR}/src/proto
        ${PROTO_SOURCE}
        DEPENDS ${PROTO_SOURCE}
        COMMENT "Compiling protobuf"
)

add_custom_target(proto_compilation DEPENDS ${PROTO_SOURCES} ${PROTO_HEADERS})

# Main sniffer executable - now with eBPF loader and ring consumer
set(SNIFFER_SOURCES
        src/userspace/main.cpp
        src/userspace/config_manager.cpp
        src/userspace/ebpf_loader.cpp
        src/userspace/ring_consumer.cpp
        ${PROTO_SOURCES}
)

add_executable(sniffer ${SNIFFER_SOURCES})

add_dependencies(sniffer bpf_program proto_compilation)

target_link_libraries(sniffer
        ${LIBBPF_LIBRARIES}
        ${ZMQ_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        ${Protobuf_LIBRARIES}
        Threads::Threads
        elf
        z
)

target_link_directories(sniffer PRIVATE
        ${LIBBPF_LIBRARY_DIRS}
        ${ZMQ_LIBRARY_DIRS}
        ${JSONCPP_LIBRARY_DIRS}
)

# Use target_compile_options instead of target_compile_definitions for CFLAGS_OTHER
target_compile_options(sniffer PRIVATE
        ${LIBBPF_CFLAGS_OTHER}
        ${ZMQ_CFLAGS_OTHER}
        ${JSONCPP_CFLAGS_OTHER}
)

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Clang: ${CLANG}")
message(STATUS "  LLVM strip: ${LLVM_STRIP}")
message(STATUS "  bpftool: ${BPFTOOL}")
message(STATUS "  libbpf version: ${LIBBPF_VERSION}")
message(STATUS "  ZeroMQ version: ${ZMQ_VERSION}")
message(STATUS "  jsoncpp version: ${JSONCPP_VERSION}")
message(STATUS "  Protobuf version: ${Protobuf_VERSION}")
message(STATUS "")