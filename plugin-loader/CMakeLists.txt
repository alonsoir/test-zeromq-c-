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
# LIBRARY TARGET
# ============================================================================
add_library(plugin_loader SHARED ${SOURCES})

target_include_directories(plugin_loader
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# dl for dlopen/dlsym
target_link_libraries(plugin_loader
    PRIVATE
        ${CMAKE_DL_LIBS}
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
message(STATUS "Phase: 1 — no crypto, no seed-client")
message(STATUS "  Auth: PHASE 2 (ADR-013, seed-client DAY 95-96)")
message(STATUS "Public Headers:")
foreach(header ${PUBLIC_HEADERS})
    message(STATUS "  - ${header}")
endforeach()
message(STATUS "========================================")
