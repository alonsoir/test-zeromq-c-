cmake_minimum_required(VERSION 3.20)
project(ml-detector-tricapa VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# C++20 STANDARD
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# BUILD OPTIONS
# ============================================================================
option(BUILD_TESTS "Build unit and integration tests" ON)
option(ENABLE_PROFILING "Enable profiling support" OFF)
option(ENABLE_SIMD "Enable SIMD optimizations (AVX2)" ON)
option(ENABLE_LTO "Enable Link Time Optimization" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)

# ============================================================================
# BUILD TYPE DEFAULT
# ============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")

# ============================================================================
# FIND DEPENDENCIES
# ============================================================================

# PkgConfig for system libraries
find_package(PkgConfig REQUIRED)

# ZeroMQ
pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq)
message(STATUS "Found ZeroMQ: ${ZMQ_VERSION}")

# Protobuf
find_package(Protobuf REQUIRED)
message(STATUS "Found Protobuf: ${Protobuf_VERSION}")

# ONNX Runtime
find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND)
    # Fallback: buscar manualmente
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            PATHS
            /usr/local/include/onnxruntime
            /usr/include/onnxruntime
            ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime/include
    )
    find_library(ONNXRUNTIME_LIB onnxruntime
            PATHS
            /usr/local/lib
            /usr/lib
            ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime/lib
    )
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
        message(STATUS "Found ONNX Runtime (manual): ${ONNXRUNTIME_LIB}")
        add_library(onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime PROPERTIES
                IMPORTED_LOCATION ${ONNXRUNTIME_LIB}
                INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_INCLUDE_DIR}
        )
    else()
        message(FATAL_ERROR "ONNX Runtime not found. Install from: https://github.com/microsoft/onnxruntime/releases")
    endif()
endif()

# nlohmann/json
find_package(nlohmann_json 3.11.0 QUIET)
if(NOT nlohmann_json_FOUND)
    # Fallback: header-only
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
            PATHS
            /usr/include
            /usr/local/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/json/include
    )
    if(NLOHMANN_JSON_INCLUDE_DIR)
        message(STATUS "Found nlohmann/json (header-only): ${NLOHMANN_JSON_INCLUDE_DIR}")
        add_library(nlohmann_json INTERFACE)
        target_include_directories(nlohmann_json INTERFACE ${NLOHMANN_JSON_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "nlohmann/json not found. Install: sudo apt-get install nlohmann-json3-dev")
    endif()
else()
    message(STATUS "Found nlohmann/json: ${nlohmann_json_VERSION}")
endif()

# spdlog
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    pkg_check_modules(SPDLOG REQUIRED IMPORTED_TARGET spdlog)
    message(STATUS "Found spdlog (pkg-config): ${SPDLOG_VERSION}")
else()
    message(STATUS "Found spdlog: ${spdlog_VERSION}")
endif()

# LZ4 (compression)
pkg_check_modules(LZ4 REQUIRED IMPORTED_TARGET liblz4)
message(STATUS "Found LZ4: ${LZ4_VERSION}")

# etcd-cpp-apiv3 (optional)
find_package(etcd-cpp-api QUIET)
if(etcd-cpp-api_FOUND)
    message(STATUS "Found etcd-cpp-api: ${etcd-cpp-api_VERSION}")
    set(HAVE_ETCD_CPP_API TRUE)
else()
    message(STATUS "etcd-cpp-api not found - ETCD integration will be disabled")
    set(HAVE_ETCD_CPP_API FALSE)
endif()

# Threads
find_package(Threads REQUIRED)

# ============================================================================
# PROTOBUF INTEGRATION (SHARED FROM ROOT)
# ============================================================================
set(PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../protobuf")
set(PROTO_FILE "${PROTO_DIR}/network_security.proto")

# Usar los archivos .pb.cc y .pb.h ya generados
if(EXISTS "${PROTO_DIR}/network_security.pb.cc")
    message(STATUS "Using pre-generated protobuf files from: ${PROTO_DIR}")
    set(PROTO_SOURCES "${PROTO_DIR}/network_security.pb.cc")
    set(PROTO_HEADERS "${PROTO_DIR}/network_security.pb.h")
else()
    message(STATUS "Generating protobuf files from: ${PROTO_FILE}")

    # Generar si no existen
    execute_process(
            COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --cpp_out=${PROTO_DIR}
            --proto_path=${PROTO_DIR}
            ${PROTO_FILE}
            RESULT_VARIABLE PROTOC_RESULT
    )

    if(NOT PROTOC_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to generate protobuf files")
    endif()

    set(PROTO_SOURCES "${PROTO_DIR}/network_security.pb.cc")
    set(PROTO_HEADERS "${PROTO_DIR}/network_security.pb.h")
endif()

# ============================================================================
# SOURCE FILES
# ============================================================================
set(SOURCES
        src/main.cpp
        src/ml_detector.cpp
        src/classifier_tricapa.cpp
        src/feature_extractor.cpp
    src/zmq_handler.cpp
        src/model_loader.cpp
        src/onnx_model.cpp
    src/feature_extractor.cpp
    src/zmq_handler.cpp
        src/zmq_handler.cpp
        src/config_loader.cpp
        src/logger.cpp
        src/stats_collector.cpp
        ${PROTO_SOURCES}
)

set(HEADERS
        include/ml_detector.hpp
        include/classifier_tricapa.hpp
        include/feature_extractor.hpp
        include/model_loader.hpp
        include/onnx_model.hpp
        include/zmq_handler.hpp
        include/config_loader.hpp
        include/logger.hpp
        include/stats_collector.hpp
        include/concurrent_queue.hpp
        ${PROTO_HEADERS}
)

# ============================================================================
# MAIN EXECUTABLE
# ============================================================================
add_executable(ml-detector ${SOURCES} ${HEADERS})

target_include_directories(ml-detector PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PROTO_DIR}
)

# ============================================================================
# LINK LIBRARIES
# ============================================================================
target_link_libraries(ml-detector PRIVATE
        PkgConfig::ZMQ
        ${Protobuf_LIBRARIES}
        onnxruntime
        PkgConfig::LZ4
        Threads::Threads
)

# nlohmann/json
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(ml-detector PRIVATE nlohmann_json::nlohmann_json)
elseif(TARGET nlohmann_json)
    target_link_libraries(ml-detector PRIVATE nlohmann_json)
endif()

# spdlog
if(TARGET spdlog::spdlog)
    target_link_libraries(ml-detector PRIVATE spdlog::spdlog)
elseif(TARGET PkgConfig::SPDLOG)
    target_link_libraries(ml-detector PRIVATE PkgConfig::SPDLOG)
endif()

# etcd (optional)
if(HAVE_ETCD_CPP_API)
    target_link_libraries(ml-detector PRIVATE etcd-cpp-api)
    target_compile_definitions(ml-detector PRIVATE HAVE_ETCD_CPP_API)
endif()

# ============================================================================
# COMPILER FLAGS
# ============================================================================

# Warnings
target_compile_options(ml-detector PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wno-unused-parameter
)

# Release optimizations
target_compile_options(ml-detector PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-DNDEBUG>
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:Release>:-funroll-loops>
)

# SIMD optimizations (AVX2)
if(ENABLE_SIMD)
    target_compile_options(ml-detector PRIVATE
            $<$<CONFIG:Release>:-march=native>
            $<$<CONFIG:Release>:-mavx2>
    )
    message(STATUS "SIMD optimizations enabled (AVX2)")
endif()

# Debug options
target_compile_options(ml-detector PRIVATE
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
)

# Link Time Optimization
if(ENABLE_LTO)
    set_property(TARGET ml-detector PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "Link Time Optimization enabled")
endif()

# Sanitizers
if(ENABLE_ASAN)
    target_compile_options(ml-detector PRIVATE -fsanitize=address)
    target_link_options(ml-detector PRIVATE -fsanitize=address)
    message(STATUS "AddressSanitizer enabled")
endif()

if(ENABLE_TSAN)
    target_compile_options(ml-detector PRIVATE -fsanitize=thread)
    target_link_options(ml-detector PRIVATE -fsanitize=thread)
    message(STATUS "ThreadSanitizer enabled")
endif()

# Position Independent Code
set_property(TARGET ml-detector PROPERTY POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# TESTS
# ============================================================================
if(BUILD_TESTS)
    enable_testing()

    # Google Test (si está disponible)
    find_package(GTest QUIET)

    if(GTest_FOUND)
        message(STATUS "Found GTest: ${GTest_VERSION}")

        # Test sources
        set(TEST_SOURCES
                tests/unit/test_classifier.cpp
                tests/unit/test_feature_extractor.cpp
                tests/unit/test_model_loader.cpp
        )

        foreach(test_src ${TEST_SOURCES})
            get_filename_component(test_name ${test_src} NAME_WE)
            add_executable(${test_name} ${test_src})

            target_include_directories(${test_name} PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${PROTO_DIR}
            )

            target_link_libraries(${test_name} PRIVATE
                    GTest::GTest
                    GTest::Main
                    # Mismas dependencias que el ejecutable principal
                    PkgConfig::ZMQ
                    ${Protobuf_LIBRARIES}
                    onnxruntime
                    PkgConfig::LZ4
                    Threads::Threads
            )

            add_test(NAME ${test_name} COMMAND ${test_name})
        endforeach()

        message(STATUS "Tests enabled")
    else()
        message(STATUS "GTest not found - tests disabled")
    endif()
endif()

# ============================================================================
# INSTALLATION
# ============================================================================
install(TARGETS ml-detector
        RUNTIME DESTINATION bin
        COMPONENT runtime
)

install(DIRECTORY config/
        DESTINATION etc/ml-detector
        COMPONENT config
)

install(DIRECTORY models/
        DESTINATION share/ml-detector/models
        COMPONENT models
        PATTERN "*.onnx" EXCLUDE
        PATTERN "*.joblib" EXCLUDE
)

install(FILES README.md
        DESTINATION share/doc/ml-detector
        COMPONENT documentation
)

# ============================================================================
# SUMMARY
# ============================================================================
message(STATUS "")
message(STATUS "======================================")
message(STATUS "ML Detector Tricapa - Configuration")
message(STATUS "======================================")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  ZeroMQ:          ${ZMQ_VERSION}")
message(STATUS "  Protobuf:        ${Protobuf_VERSION}")
message(STATUS "  ONNX Runtime:    Found")
message(STATUS "  nlohmann/json:   Found")
message(STATUS "  spdlog:          Found")
message(STATUS "  LZ4:             ${LZ4_VERSION}")
message(STATUS "  etcd-cpp-api:    ${HAVE_ETCD_CPP_API}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  Build tests:     ${BUILD_TESTS}")
message(STATUS "  SIMD (AVX2):     ${ENABLE_SIMD}")
message(STATUS "  LTO:             ${ENABLE_LTO}")
message(STATUS "  ASAN:            ${ENABLE_ASAN}")
message(STATUS "  TSAN:            ${ENABLE_TSAN}")
message(STATUS "")
message(STATUS "Protobuf:")
message(STATUS "  Proto dir:       ${PROTO_DIR}")
message(STATUS "  Proto file:      ${PROTO_FILE}")
message(STATUS "  Generated:       ${PROTO_SOURCES}")
message(STATUS "======================================")
message(STATUS "")