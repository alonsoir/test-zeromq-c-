cmake_minimum_required(VERSION 3.16)
project(crypto_transport VERSION 1.0.0 LANGUAGES CXX)

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
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

# Find dependencies
find_package(PkgConfig REQUIRED)

# libsodium (ChaCha20-Poly1305)
pkg_check_modules(LIBSODIUM REQUIRED libsodium)
if(NOT LIBSODIUM_FOUND)
    message(FATAL_ERROR "libsodium not found. Install: sudo apt-get install libsodium-dev")
endif()

# LZ4 (compression)
pkg_check_modules(LZ4 REQUIRED liblz4)
if(NOT LZ4_FOUND)
    message(FATAL_ERROR "liblz4 not found. Install: sudo apt-get install liblz4-dev")
endif()

# Sources
set(SOURCES
        src/crypto.cpp
        src/compression.cpp
        src/utils.cpp
)

# Create shared library
add_library(crypto_transport SHARED ${SOURCES})

# Include directories
target_include_directories(crypto_transport
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${LIBSODIUM_INCLUDE_DIRS}
        ${LZ4_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(crypto_transport
        PRIVATE
        ${LIBSODIUM_LIBRARIES}
        ${LZ4_LIBRARIES}
)

# Set library properties
set_target_properties(crypto_transport PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        PUBLIC_HEADER "include/crypto_transport/crypto.hpp;include/crypto_transport/compression.hpp;include/crypto_transport/utils.hpp"
)

# Install rules
install(TARGETS crypto_transport
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include/crypto_transport
)

# Tests (optional)
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "========================================")
message(STATUS "crypto-transport Library Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "libsodium: ${LIBSODIUM_VERSION}")
message(STATUS "LZ4: ${LZ4_VERSION}")
message(STATUS "========================================")