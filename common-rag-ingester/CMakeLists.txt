cmake_minimum_required(VERSION 3.16)
project(common-rag-ingester VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address")

# Find FAISS manually (no pkg-config available)
find_path(FAISS_INCLUDE_DIR
        NAMES faiss/IndexFlat.h
        PATHS /usr/local/include /usr/include
        DOC "FAISS include directory"
)

find_library(FAISS_LIBRARY
        NAMES faiss
        PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
        DOC "FAISS library"
)

if(NOT FAISS_INCLUDE_DIR OR NOT FAISS_LIBRARY)
    message(FATAL_ERROR "FAISS not found. Please install FAISS library.")
endif()

message(STATUS "Found FAISS: ${FAISS_LIBRARY}")
message(STATUS "FAISS include: ${FAISS_INCLUDE_DIR}")

# Library: common-rag-ingester
add_library(common-rag-ingester SHARED
        src/dimensionality_reducer.cpp
)

target_include_directories(common-rag-ingester
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${FAISS_INCLUDE_DIR}
)

target_link_libraries(common-rag-ingester
        PUBLIC
        ${FAISS_LIBRARY}
        pthread
)

# Set library properties
set_target_properties(common-rag-ingester PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        PUBLIC_HEADER include/dimensionality_reducer.hpp
)

# Installation rules
include(GNUInstallDirs)

install(TARGETS common-rag-ingester
        EXPORT common-rag-ingester-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ml_defender/rag
)

# Export targets
install(EXPORT common-rag-ingester-targets
        FILE common-rag-ingester-targets.cmake
        NAMESPACE ml_defender::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/common-rag-ingester
)

# Create config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/common-rag-ingester-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/common-rag-ingester-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/common-rag-ingester
)

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/common-rag-ingester-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/common-rag-ingester-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/common-rag-ingester-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/common-rag-ingester
)

# Build summary
message(STATUS "")
message(STATUS "=== common-rag-ingester Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "FAISS Include: ${FAISS_INCLUDE_DIR}")
message(STATUS "FAISS Library: ${FAISS_LIBRARY}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================")
message(STATUS "")