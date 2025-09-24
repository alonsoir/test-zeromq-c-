cmake_minimum_required(VERSION 3.16)
project(service3 LANGUAGES CXX)

# C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Protobuf REQUIRED)

# Find ZeroMQ
pkg_check_modules(ZMQ REQUIRED libzmq>=4.3)

# Find jsoncpp
pkg_check_modules(JSONCPP REQUIRED jsoncpp>=1.9)

# Use pre-compiled protobuf files
set(PROTO_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/../protobuf/network_security.pb.cc")
set(PROTO_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/../protobuf/network_security.pb.h")

# Create executable
add_executable(service3
        src/main_simple_fixed.cpp
        src/config_manager.cpp
)

# Link libraries
target_link_libraries(service3
        ${ZMQ_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        ${Protobuf_LIBRARIES}
        pthread
)

# Include directories for dependencies
target_include_directories(service3 PRIVATE
        ${ZMQ_INCLUDE_DIRS}
        ${JSONCPP_INCLUDE_DIRS}
        ${Protobuf_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/../protobuf
)

# Compiler flags
target_compile_options(service3 PRIVATE
        ${ZMQ_CFLAGS_OTHER}
        ${JSONCPP_CFLAGS_OTHER}
)

# Set output directory
set_target_properties(service3 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
)