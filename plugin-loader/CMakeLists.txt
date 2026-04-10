cmake_minimum_required(VERSION 3.16)
project(plugin_loader VERSION 1.0.0 LANGUAGES CXX C)

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
# ⚠️  DO NOT set CMAKE_BUILD_TYPE, optimization or sanitizer flags here

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CXX Flags (from Makefile): ${CMAKE_CXX_FLAGS}")

# ============================================================================
# SOURCES
# ============================================================================
set(SOURCES
    src/plugin_loader.cpp
)

set(PUBLIC_HEADERS
    include/plugin_loader/plugin_api.h
    include/plugin_loader/plugin_loader.hpp
)


# ============================================================================
# LIBSODIUM (ADR-025 — Ed25519 plugin verification)
# NO_DEFAULT_PATH: use only libsodium compiled from source (DAY 96-97)
# ============================================================================
find_library(SODIUM_LIB sodium
    PATHS /usr/local/lib /usr/lib
    NO_DEFAULT_PATH
)
if(NOT SODIUM_LIB)
    message(FATAL_ERROR "libsodium not found — run: sudo bash tools/provision.sh full")
endif()
find_path(SODIUM_INCLUDE sodium.h
    PATHS /usr/local/include /usr/include
    NO_DEFAULT_PATH
)
message(STATUS "libsodium: ${SODIUM_LIB}")

# ============================================================================
# MLD_PLUGIN_PUBKEY_HEX (ADR-025 D7)
# Ed25519 public key hardcoded in binary — NOT a secret, correct by design
# Source: /etc/ml-defender/plugins/plugin_signing.pk (raw 32 bytes, hex)
# To update: provision.sh --reset then recompile
# ============================================================================
if(NOT DEFINED MLD_PLUGIN_PUBKEY_HEX OR MLD_PLUGIN_PUBKEY_HEX STREQUAL "")
    set(MLD_PLUGIN_PUBKEY_HEX "b824bcd7a14f6e19a0d8c9be86110828060e600723d12e118dccc95c862c8468")
    message(STATUS "MLD_PLUGIN_PUBKEY_HEX: using default dev key")
else()
    message(STATUS "MLD_PLUGIN_PUBKEY_HEX: injected via CMake -D")
endif()

# ============================================================================
# LIBRARY TARGET
# ============================================================================
add_library(plugin_loader SHARED ${SOURCES})

target_include_directories(plugin_loader
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${SODIUM_INCLUDE}
)

# dl for dlopen/dlsym
target_link_libraries(plugin_loader
    PRIVATE
        ${CMAKE_DL_LIBS}
        ${SODIUM_LIB}
)

target_compile_definitions(plugin_loader
    PRIVATE
        MLD_PLUGIN_PUBKEY_HEX="${MLD_PLUGIN_PUBKEY_HEX}"
)
target_compile_options(plugin_loader
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

set_target_properties(plugin_loader PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# INSTALLATION
# ============================================================================
install(TARGETS plugin_loader
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${PUBLIC_HEADERS}
    DESTINATION include/plugin_loader
)

# ============================================================================
# TESTS (optional)
# ============================================================================
option(BUILD_TESTS "Build tests" OFF)  # tests/ se añade en DAY 94
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================
message(STATUS "========================================")
message(STATUS "plugin-loader Library Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Phase: 2 — Ed25519 plugin verification (ADR-025)")
message(STATUS "  Auth: PHASE 2 (ADR-013, seed-client DAY 95-96)")
message(STATUS "Public Headers:")
foreach(header ${PUBLIC_HEADERS})
    message(STATUS "  - ${header}")
endforeach()
message(STATUS "========================================")
