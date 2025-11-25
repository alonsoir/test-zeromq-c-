cmake_minimum_required(VERSION 3.20)
project(rag-security-system VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# CONFIGURACIÓN BÁSICA
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party")

# ============================================================================
# LLAMA.CPP - CONFIGURACIÓN
# ============================================================================
set(LLAMA_CPP_DIR "${THIRD_PARTY_DIR}/llama.cpp")
set(LLAMA_BUILD_DIR "${LLAMA_CPP_DIR}/build")

if(EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(STATUS "Found llama.cpp: ${LLAMA_CPP_DIR}")

    if(EXISTS "${LLAMA_BUILD_DIR}/bin/libllama.so")
        set(LLAMA_LIB_PATH "${LLAMA_BUILD_DIR}/bin/libllama.so")
        set(HAVE_LLAMA_CPP TRUE)
        message(STATUS "✅ Found llama shared library: ${LLAMA_LIB_PATH}")
    else()
        set(HAVE_LLAMA_CPP FALSE)
        message(WARNING "❌ llama shared library not found")
    endif()

    set(LLAMA_INCLUDE_DIRS
            ${LLAMA_CPP_DIR}/include
            ${LLAMA_CPP_DIR}/ggml/include
            ${LLAMA_CPP_DIR}
            ${LLAMA_CPP_DIR}/src
            ${LLAMA_CPP_DIR}/common
    )
else()
    message(WARNING "llama.cpp not found at: ${LLAMA_CPP_DIR}")
    set(HAVE_LLAMA_CPP FALSE)
endif()

# ============================================================================
# DEPENDENCIAS EXTERNAS - BUSCAR EN SISTEMA
# ============================================================================
message(STATUS "🔍 Buscando dependencias del sistema...")

# 1. HTTPLIB (header-only)
find_path(HTTPLIB_INCLUDE_DIR httplib.h
        PATHS /usr/include /usr/local/include
        REQUIRED
)

# 2. NLOHMANN JSON (header-only)
find_path(JSON_INCLUDE_DIR nlohmann/json.hpp
        PATHS /usr/include /usr/local/include
        REQUIRED
)

# 3. Threads
find_package(Threads REQUIRED)

# ============================================================================
# ARCHIVOS FUENTE
# ============================================================================
set(SOURCES
        src/main.cpp
        src/etcd_client.cpp
        src/whitelist_manager.cpp
        src/llama_integration_real.cpp
        src/config_manager.cpp
        src/rag_command_manager.cpp
)

# ============================================================================
# EJECUTABLE PRINCIPAL
# ============================================================================
add_executable(rag-security ${SOURCES})

# ============================================================================
# CONFIGURACIÓN DE INCLUSIONES
# ============================================================================
target_include_directories(rag-security PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${HTTPLIB_INCLUDE_DIR}
        ${JSON_INCLUDE_DIR}
)

# ============================================================================
# LINKING
# ============================================================================
target_link_libraries(rag-security PRIVATE Threads::Threads)

if(HAVE_LLAMA_CPP AND LLAMA_LIB_PATH AND LLAMA_INCLUDE_DIRS)
    target_link_libraries(rag-security PRIVATE "${LLAMA_LIB_PATH}")
    target_include_directories(rag-security PRIVATE ${LLAMA_INCLUDE_DIRS})
    target_compile_definitions(rag-security PRIVATE HAVE_LLAMA_CPP)
    message(STATUS "✅ llama.cpp integration enabled")
else()
    message(STATUS "❌ llama.cpp not available - using simulation mode")
    target_compile_definitions(rag-security PRIVATE LLAMA_SIMULATION_MODE)
endif()

# ============================================================================
# CONFIGURACIÓN DE COMPILACIÓN
# ============================================================================
target_compile_options(rag-security PRIVATE
        -Wall
        -Wextra
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(rag-security PRIVATE -g -O0)
else()
    target_compile_options(rag-security PRIVATE -O2)
endif()

# ============================================================================
# VERIFICACIÓN FINAL
# ============================================================================
message(STATUS "🎯 Build configuration:")
message(STATUS "   - httplib: ${HTTPLIB_INCLUDE_DIR}")
message(STATUS "   - nlohmann/json: ${JSON_INCLUDE_DIR}")
message(STATUS "   - llama.cpp: ${HAVE_LLAMA_CPP}")
message(STATUS "   - threads: OK")