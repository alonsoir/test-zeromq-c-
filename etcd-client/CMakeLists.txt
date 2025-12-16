cmake_minimum_required(VERSION 3.16)
project(etcd_client VERSION 1.0.0 LANGUAGES CXX)

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

# Try to find libsodium (ChaCha20)
pkg_check_modules(LIBSODIUM libsodium)

# Find LZ4
pkg_check_modules(LZ4 REQUIRED liblz4)
if(NOT LZ4_FOUND)
    message(FATAL_ERROR "liblz4 not found. Install: sudo apt-get install liblz4-dev")
endif()

# nlohmann/json (header-only, should be available)
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found via find_package, will use system headers")
endif()

# Sources (base)
set(SOURCES
    src/etcd_client.cpp
    src/config_loader.cpp
    src/http_client.cpp
    src/component_registration.cpp
    src/compression_lz4.cpp
)

# Add crypto source based on availability
if(LIBSODIUM_FOUND)
    list(APPEND SOURCES src/crypto_chacha20.cpp)
    add_definitions(-DHAVE_LIBSODIUM)
    message(STATUS "✅ Using ChaCha20-Poly1305 (libsodium)")
else()
    list(APPEND SOURCES src/crypto_aes256.cpp)
    message(STATUS "⚠️  libsodium not found, falling back to AES256 (Crypto++)")
    message(STATUS "    Install libsodium: sudo apt-get install libsodium-dev")
endif()

# Create shared library
add_library(etcd_client SHARED ${SOURCES})

# Include directories
target_include_directories(etcd_client
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${LZ4_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(etcd_client
    PRIVATE
        ${LZ4_LIBRARIES}
)

# Add libsodium if available
if(LIBSODIUM_FOUND)
    target_include_directories(etcd_client PRIVATE ${LIBSODIUM_INCLUDE_DIRS})
    target_link_libraries(etcd_client PRIVATE ${LIBSODIUM_LIBRARIES})
else()
    # Fallback to Crypto++
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(CRYPTOPP REQUIRED libcrypto++)
    target_include_directories(etcd_client PRIVATE ${CRYPTOPP_INCLUDE_DIRS})
    target_link_libraries(etcd_client PRIVATE ${CRYPTOPP_LIBRARIES})
endif()

# Add nlohmann_json if found
if(nlohmann_json_FOUND)
    target_link_libraries(etcd_client PRIVATE nlohmann_json::nlohmann_json)
endif()

# Set library properties
set_target_properties(etcd_client PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER include/etcd_client/etcd_client.hpp
)

# Install rules
install(TARGETS etcd_client
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/etcd_client
)

# Install all headers
install(DIRECTORY include/etcd_client
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Tests (optional)
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "========================================")
message(STATUS "etcd-client Library Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Encryption: ${LIBSODIUM_FOUND}")
if(LIBSODIUM_FOUND)
    message(STATUS "  Algorithm: ChaCha20-Poly1305 (libsodium)")
else()
    message(STATUS "  Algorithm: AES256-GCM (Crypto++)")
endif()
message(STATUS "Compression: LZ4")
message(STATUS "========================================")
