cmake_minimum_required(VERSION 3.20)
project(firewall-acl-agent VERSION 1.0.0 LANGUAGES CXX)

# C++20 required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address")

# Find dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq>=4.3)
pkg_check_modules(JSONCPP REQUIRED jsoncpp>=1.9)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${ZMQ_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    src/core/acl_manager.cpp
    src/core/iptables_wrapper.cpp
    src/core/rule_generator.cpp
    src/rules/temporal_acl.cpp
    src/rules/rate_limiter.cpp
    src/rules/whitelist_manager.cpp
    src/api/zmq_subscriber.cpp
)

# Main executable
add_executable(firewall-agent ${SOURCES})

target_link_libraries(firewall-agent
    ${ZMQ_LIBRARIES}
    ${JSONCPP_LIBRARIES}
    pthread
    iptc  # libiptc for iptables C API
)

# Copy config to build directory
file(COPY ${CMAKE_SOURCE_DIR}/config DESTINATION ${CMAKE_BINARY_DIR})

# Tests
enable_testing()
add_subdirectory(tests)

# Installation
install(TARGETS firewall-agent DESTINATION bin)
install(DIRECTORY config/ DESTINATION etc/ml-defender/firewall)
install(DIRECTORY scripts/ DESTINATION share/ml-defender/firewall/scripts
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ)

message(STATUS "")
message(STATUS "=== Firewall ACL Agent Configuration ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "=======================================")
message(STATUS "")
