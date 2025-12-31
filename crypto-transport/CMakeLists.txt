cmake_minimum_required(VERSION 3.16)
project(crypto_transport VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# C++20 STANDARD
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# BUILD TYPE
# ============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================================
# COMPILER FLAGS
# ============================================================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# ============================================================================
# FIND DEPENDENCIES
# ============================================================================
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

# ============================================================================
# SOURCES
# ============================================================================
set(SOURCES
        src/crypto.cpp
        src/compression.cpp
        src/utils.cpp
)

# Public headers that will be installed
set(PUBLIC_HEADERS
        include/crypto_transport/crypto.hpp
        include/crypto_transport/compression.hpp
        include/crypto_transport/utils.hpp
        include/crypto_transport/crypto_manager.hpp  # ← DAY 27: ADDED
)

# ============================================================================
# LIBRARY TARGET
# ============================================================================
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

# Library properties
set_target_properties(crypto_transport PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# INSTALLATION
# ============================================================================
install(TARGETS crypto_transport
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

# Install headers individually to ensure crypto_manager.hpp is included
install(FILES ${PUBLIC_HEADERS}
        DESTINATION include/crypto_transport
)

# Alternative: Install entire directory (if you prefer)
# install(DIRECTORY include/crypto_transport
#     DESTINATION include
#     FILES_MATCHING PATTERN "*.hpp"
# )

# ============================================================================
# TESTS (optional)
# ============================================================================
option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)

    message(STATUS "========================================")
    message(STATUS "crypto-transport Tests Configuration")
    message(STATUS "========================================")
    message(STATUS "Test framework: Google Test")
    message(STATUS "Tests: test_crypto, test_compression, test_integration")
    message(STATUS "========================================")
endif()

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================
message(STATUS "========================================")
message(STATUS "crypto-transport Library Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "libsodium: ${LIBSODIUM_VERSION}")
message(STATUS "LZ4: ${LZ4_VERSION}")
message(STATUS "========================================")
message(STATUS "Public Headers:")
foreach(header ${PUBLIC_HEADERS})
    message(STATUS "  - ${header}")
endforeach()
message(STATUS "========================================")
message(STATUS "Install destinations:")
message(STATUS "  Library: ${CMAKE_INSTALL_PREFIX}/lib")
message(STATUS "  Headers: ${CMAKE_INSTALL_PREFIX}/include/crypto_transport")
message(STATUS "========================================")