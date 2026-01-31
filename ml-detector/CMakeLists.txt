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

# Debug flags
# DISABLED FOR MAKEFILE CONTROL: set(CMAKE_BUILD_TYPE Debug)
# DISABLED FOR MAKEFILE CONTROL: set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer")

# ============================================================================
# FIND DEPENDENCIES
# ============================================================================

# PkgConfig for system libraries
find_package(PkgConfig REQUIRED)

# nlohmann_json
find_package(nlohmann_json REQUIRED)
message(STATUS "Found nlohmann_json")

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

# etcd-cpp-apiv3 (optional)
find_package(etcd-cpp-api QUIET)
if(etcd-cpp-api_FOUND)
    message(STATUS "Found etcd-cpp-api: ${etcd-cpp-api_VERSION}")
    set(HAVE_ETCD_CPP_API TRUE)
else()
    message(STATUS "etcd-cpp-api not found - ETCD integration will be disabled")
    set(HAVE_ETCD_CPP_API FALSE)
endif()

# ============================================================================
# ETCD-CLIENT LIBRARY (Custom)
# ============================================================================
set(ETCD_CLIENT_LIB /vagrant/etcd-client/build/libetcd_client.so)
set(ETCD_CLIENT_INCLUDE_DIR /vagrant/etcd-client/include)

if(EXISTS ${ETCD_CLIENT_LIB})
    message(STATUS "✅ Found etcd-client library: ${ETCD_CLIENT_LIB}")
    include_directories(${ETCD_CLIENT_INCLUDE_DIR})
    set(HAVE_ETCD_CLIENT TRUE)
else()
    message(FATAL_ERROR "❌ etcd-client library not found - required for ml-detector")
endif()

# ============================================================================
# CRYPTO-TRANSPORT LIBRARY (Custom) - DAY 27 FIX
# ============================================================================
# Busca primero en /usr/local (instalación system-wide), luego en build local
find_library(CRYPTO_TRANSPORT_LIB
        NAMES crypto_transport
        PATHS
        /usr/local/lib
        /vagrant/crypto-transport/build
        NO_DEFAULT_PATH
)

find_path(CRYPTO_TRANSPORT_INCLUDE_DIR
        NAMES crypto_transport/crypto_manager.hpp
        PATHS
        /usr/local/include
        /vagrant/crypto-transport/include
        NO_DEFAULT_PATH
)

if(CRYPTO_TRANSPORT_LIB AND CRYPTO_TRANSPORT_INCLUDE_DIR)
    message(STATUS "✅ Found crypto-transport library: ${CRYPTO_TRANSPORT_LIB}")
    message(STATUS "✅ Found crypto-transport headers: ${CRYPTO_TRANSPORT_INCLUDE_DIR}")
    include_directories(${CRYPTO_TRANSPORT_INCLUDE_DIR})
    set(HAVE_CRYPTO_TRANSPORT TRUE)
else()
    message(FATAL_ERROR "❌ crypto-transport not found - required for ml-detector
    Library: ${CRYPTO_TRANSPORT_LIB}
    Headers: ${CRYPTO_TRANSPORT_INCLUDE_DIR}

    Fix: Run 'make crypto-transport-build' from project root")
endif()

# Threads
find_package(Threads REQUIRED)

# ============================================================================
# PROTOBUF UNIFICADO - VERIFICACIÓN Y CONFIGURACIÓN
# ============================================================================

# Verificar que los archivos protobuf unificados existen
if(NOT EXISTS "${CMAKE_BINARY_DIR}/proto/network_security.pb.cc")
    message(FATAL_ERROR "❌ Protobuf unificado no encontrado. Ejecuta: cd /vagrant/protobuf && ./generate.sh")
else()
    message(STATUS "✅ Protobuf unificado encontrado: ${CMAKE_BINARY_DIR}/proto/network_security.pb.cc")
endif()

set(PROTO_SOURCES ${CMAKE_BINARY_DIR}/proto/network_security.pb.cc)
set(PROTO_HEADERS ${CMAKE_BINARY_DIR}/proto/network_security.pb.h)

# ============================================================================
# SOURCE FILES
# ============================================================================
set(SOURCES
        src/main.cpp
        src/ml_detector.cpp
        src/classifier_tricapa.cpp
        src/feature_extractor.cpp
        src/rag_logger.cpp
        src/contract_validator.cpp
        src/zmq_handler.cpp
        src/model_loader.cpp
        src/onnx_model.cpp
        src/config_loader.cpp
        src/logger.cpp
        src/stats_collector.cpp
        src/ransomware_detector.cpp
        src/ddos_detector.cpp
        src/traffic_detector.cpp
        src/internal_detector.cpp
        src/etcd_client.cpp
        ${PROTO_SOURCES}
)

# ML Defender - Ransomware Detector Library
add_library(ransomware_detector
        src/ransomware_detector.cpp
)

target_include_directories(ransomware_detector
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(ransomware_detector
        PUBLIC cxx_std_20
)

target_compile_options(ransomware_detector PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -O3
        -march=native
        -flto
        $<$<CONFIG:Release>:-DNDEBUG>
)

# Optional: Install headers
install(DIRECTORY include/ml_defender
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp"
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
        include/etcd_client.hpp
        ${PROTO_HEADERS}
)

# ============================================================================
# INCLUDE DIRECTORIES - ACTUALIZADO PARA PROTOBUF UNIFICADO y reason_codes
# ============================================================================
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/proto  # Archivos protobuf unificados
        /vagrant/common/include # Se incluye reason_codes.hpp
)

# ============================================================================
# MODELS SYMLINK (Single Source of Truth)
# ============================================================================
message(STATUS "")
message(STATUS "🔗 Setting up models symlink...")
message(STATUS "   Source: ${CMAKE_SOURCE_DIR}/models")
message(STATUS "   Target: ${CMAKE_BINARY_DIR}/models")

# Remove existing symlink/directory if it exists
if(EXISTS ${CMAKE_BINARY_DIR}/models)
    file(REMOVE ${CMAKE_BINARY_DIR}/models)
endif()

# Create symlink
execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_SOURCE_DIR}/models
        ${CMAKE_BINARY_DIR}/models
        RESULT_VARIABLE SYMLINK_RESULT
)

if(SYMLINK_RESULT EQUAL 0)
    message(STATUS "✅ Models symlink created successfully")
    message(STATUS "   Config will use: models/production/")
    message(STATUS "   Points to:       ../models/production/")
else()
    message(WARNING "⚠️  Failed to create models symlink")
    message(WARNING "   Manual workaround: cp -r ${CMAKE_SOURCE_DIR}/models ${CMAKE_BINARY_DIR}/")
endif()

# ============================================================================
# CONFIG SYMLINK (Single Source of Truth)
# ============================================================================
message(STATUS "")
message(STATUS "🔗 Setting up config symlink...")
message(STATUS "   Source: ${CMAKE_SOURCE_DIR}/config")
message(STATUS "   Target: ${CMAKE_BINARY_DIR}/config")

# Remove existing symlink/directory if it exists
if(EXISTS ${CMAKE_BINARY_DIR}/config)
    file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/config)
endif()

# Create symlink
execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_SOURCE_DIR}/config
        ${CMAKE_BINARY_DIR}/config
        RESULT_VARIABLE CONFIG_SYMLINK_RESULT
)

if(CONFIG_SYMLINK_RESULT EQUAL 0)
    message(STATUS "✅ Config symlink created successfully")
else()
    message(WARNING "⚠️  Failed to create config symlink")
endif()

message(STATUS "")

# ============================================================================
# MAIN EXECUTABLE
# ============================================================================
add_executable(ml-detector ${SOURCES} ${HEADERS})

target_include_directories(ml-detector PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/proto  # Archivos protobuf unificados
)

# ============================================================================
# LINK LIBRARIES
# ============================================================================
target_link_libraries(ml-detector PRIVATE
        PkgConfig::ZMQ
        ${Protobuf_LIBRARIES}
        onnxruntime
        Threads::Threads
        ransomware_detector
        nlohmann_json::nlohmann_json
        ${ETCD_CLIENT_LIB}
        ${CRYPTO_TRANSPORT_LIB}
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
                    ${CMAKE_BINARY_DIR}/proto  # Archivos protobuf unificados
            )

            target_link_libraries(${test_name} PRIVATE
                    GTest::GTest
                    GTest::Main
                    PkgConfig::ZMQ
                    ${Protobuf_LIBRARIES}
                    onnxruntime
                    Threads::Threads
            )

            add_test(NAME ${test_name} COMMAND ${test_name})
        endforeach()

        # ============================================================================
        # TEST_DETECTORS - FIXED: Ahora incluye los archivos .cpp de detectores
        # ============================================================================
        add_executable(test_detectors
                tests/unit/test_detectors.cpp
                src/ddos_detector.cpp
                src/traffic_detector.cpp
                src/internal_detector.cpp
        )

        target_include_directories(test_detectors PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_BINARY_DIR}/proto
        )

        target_link_libraries(test_detectors PRIVATE
                GTest::GTest
                GTest::Main
                PkgConfig::ZMQ
                ${Protobuf_LIBRARIES}
                onnxruntime
                Threads::Threads
        )

        add_test(NAME test_detectors COMMAND test_detectors)

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
# RANSOMWARE DETECTOR TESTS
# ============================================================================

if(BUILD_TESTS)
    # Unit Tests - Solo el detector aislado
    add_executable(test_ransomware_detector_unit
            tests/unit/test_ransomware_detector.cpp
    )

    target_link_libraries(test_ransomware_detector_unit PRIVATE
            ransomware_detector
    )

    add_test(NAME RansomwareDetectorUnit
            COMMAND test_ransomware_detector_unit
    )

    # Unit Tests - DDoS, Traffic, Internal Detectors
    add_executable(test_detectors_unit
            tests/unit/test_detectors.cpp
            src/ddos_detector.cpp
            src/traffic_detector.cpp
            src/internal_detector.cpp
    )

    target_include_directories(test_detectors_unit PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(test_detectors_unit PRIVATE
    )

    add_test(NAME DetectorsUnit
            COMMAND test_detectors_unit
    )
endif()

# ============================================================================
# SUMMARY - ACTUALIZADO
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
message(STATUS "  etcd-client:     ${HAVE_ETCD_CLIENT}")
message(STATUS "  crypto-transport: ${HAVE_CRYPTO_TRANSPORT}")
message(STATUS "  etcd-cpp-api:    ${HAVE_ETCD_CPP_API}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  Build tests:     ${BUILD_TESTS}")
message(STATUS "  SIMD (AVX2):     ${ENABLE_SIMD}")
message(STATUS "  LTO:             ${ENABLE_LTO}")
message(STATUS "  ASAN:            ${ENABLE_ASAN}")
message(STATUS "  TSAN:            ${ENABLE_TSAN}")
message(STATUS "")
message(STATUS "🎯 Single Source of Truth:")
message(STATUS "  Models:          ${CMAKE_SOURCE_DIR}/models → build/models (symlink)")
message(STATUS "  Config:          ${CMAKE_SOURCE_DIR}/config → build/config (symlink)")
message(STATUS "  Protobuf:        ${CMAKE_BINARY_DIR}/proto/network_security.pb.cc (unificado)")
message(STATUS "======================================")
message(STATUS "")