cmake_minimum_required(VERSION 3.20)
project(rag-security-system VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# CONFIGURACIÓN BÁSICA
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================================
# BÚSQUEDA DE DEPENDENCIAS (DEBE IR ANTES DE TARGETS)
# ============================================================================
find_package(PkgConfig REQUIRED)

# ZeroMQ
pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq)
message(STATUS "Found ZeroMQ: ${ZMQ_VERSION}")

# Protobuf - Búsqueda más robusta
find_package(Protobuf QUIET)

if(Protobuf_FOUND)
    message(STATUS "Found Protobuf via find_package: ${Protobuf_VERSION}")
else()
    # Búsqueda manual alternativa
    find_path(PROTOBUF_INCLUDE_DIR google/protobuf/message.h
            PATHS
            /usr/include
            /usr/local/include
            /opt/homebrew/include  # Para macOS con Homebrew
    )

    find_library(PROTOBUF_LIBRARY NAMES protobuf
            PATHS
            /usr/lib
            /usr/lib/x86_64-linux-gnu
            /usr/local/lib
            /opt/homebrew/lib  # Para macOS con Homebrew
    )

    if(PROTOBUF_INCLUDE_DIR AND PROTOBUF_LIBRARY)
        message(STATUS "Found Protobuf manually: ${PROTOBUF_LIBRARY}")
        set(Protobuf_FOUND TRUE)
        set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIR})
        set(Protobuf_LIBRARIES ${PROTOBUF_LIBRARY})
    else()
        message(FATAL_ERROR "Protobuf not found. Install: sudo apt install libprotobuf-dev OR brew install protobuf")
    endif()
endif()

# Threads
find_package(Threads REQUIRED)

# nlohmann/json
find_package(nlohmann_json 3.11.0 QUIET)
if(NOT nlohmann_json_FOUND)
    # Búsqueda manual
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
            PATHS
            /usr/include
            /usr/local/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/json/include
    )
    if(NLOHMANN_JSON_INCLUDE_DIR)
        message(STATUS "Found nlohmann/json (header-only): ${NLOHMANN_JSON_INCLUDE_DIR}")
        add_library(nlohmann_json INTERFACE IMPORTED)
        set_target_properties(nlohmann_json PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${NLOHMANN_JSON_INCLUDE_DIR}"
        )
    else()
        message(FATAL_ERROR "nlohmann/json not found. Install: sudo apt-get install nlohmann-json3-dev")
    endif()
else()
    message(STATUS "Found nlohmann/json: ${nlohmann_json_VERSION}")
endif()

# etcd-cpp-apiv3
find_package(etcd-cpp-api QUIET)
if(etcd-cpp-api_FOUND)
    message(STATUS "Found etcd-cpp-api: ${etcd-cpp-api_VERSION}")
    set(HAVE_ETCD_CPP_API TRUE)
else()
    # Búsqueda manual de etcd
    find_path(ETCD_INCLUDE_DIR etcd/Client.hpp
            PATHS /usr/include /usr/local/include
    )
    find_library(ETCD_LIBRARY etcd-cpp-api
            PATHS /usr/lib /usr/local/lib
    )
    if(ETCD_INCLUDE_DIR AND ETCD_LIBRARY)
        message(STATUS "Found etcd-cpp-api (manual): ${ETCD_LIBRARY}")
        set(HAVE_ETCD_CPP_API TRUE)
        add_library(etcd-cpp-api INTERFACE IMPORTED)
        set_target_properties(etcd-cpp-api PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${ETCD_INCLUDE_DIR}"
                INTERFACE_LINK_LIBRARIES "${ETCD_LIBRARY}"
        )
    else()
        message(WARNING "etcd-cpp-api not found - etcd features disabled")
        set(HAVE_ETCD_CPP_API FALSE)
    endif()
endif()

# ============================================================================
# LLAMA.CPP INTEGRATION
# ============================================================================
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp")
set(LLAMA_BUILD_DIR "${LLAMA_CPP_DIR}/build")

if(EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(STATUS "Found llama.cpp: ${LLAMA_CPP_DIR}")

    # Incluir directorios de headers
    set(LLAMA_INCLUDE_DIRS
            ${LLAMA_CPP_DIR}/include
            ${LLAMA_CPP_DIR}/src
            ${LLAMA_CPP_DIR}/ggml/include
            ${LLAMA_CPP_DIR}/ggml/src
    )

    # Verificar que existe la librería
    if(EXISTS "${LLAMA_BUILD_DIR}/src/libllama.a")
        message(STATUS "✅ Found llama library: ${LLAMA_BUILD_DIR}/src/libllama.a")
        set(HAVE_LLAMA_CPP TRUE)

        # Crear target importado
        add_library(llama STATIC IMPORTED)
        set_target_properties(llama PROPERTIES
                IMPORTED_LOCATION "${LLAMA_BUILD_DIR}/src/libllama.a"
                INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_INCLUDE_DIRS}"
        )

        # También necesitamos las librerías de ggml
        if(EXISTS "${LLAMA_BUILD_DIR}/ggml/src/libggml.a")
            add_library(ggml STATIC IMPORTED)
            set_target_properties(ggml PROPERTIES
                    IMPORTED_LOCATION "${LLAMA_BUILD_DIR}/ggml/src/libggml.a"
            )
        endif()

    else()
        message(WARNING "❌ llama library not found at expected location")
        set(HAVE_LLAMA_CPP FALSE)
    endif()

else()
    message(WARNING "llama.cpp not found at: ${LLAMA_CPP_DIR}")
    set(HAVE_LLAMA_CPP FALSE)
endif()

# ============================================================================
# ARCHIVOS PROTOBUF
# ============================================================================
set(PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../protobuf")

if(EXISTS "${PROTO_DIR}/network_security.pb.cc" AND EXISTS "${PROTO_DIR}/network_security.pb.h")
    message(STATUS "✅ Using pre-generated protobuf files from: ${PROTO_DIR}")
    set(PROTO_SOURCES "${PROTO_DIR}/network_security.pb.cc")
    set(PROTO_HEADERS "${PROTO_DIR}/network_security.pb.h")
else()
    message(FATAL_ERROR "❌ Protobuf files not found at: ${PROTO_DIR}")
endif()

# ============================================================================
# FUENTES
# ============================================================================
set(SOURCES
        src/main.cpp
        src/rag_command_system.cpp
        src/whitelist_parser.cpp
        src/security_context.cpp
        src/query_validator.cpp
        src/response_generator.cpp
        # NUEVOS MANEJADORES
        src/config_manager.cpp
        src/whitelist_manager.cpp
        ${PROTO_SOURCES}
)

# ============================================================================
# EJECUTABLE PRINCIPAL (DEBE IR DESPUÉS DE DEFINIR FUENTES)
# ============================================================================
add_executable(rag-security ${SOURCES})

# Directorios de inclusión
target_include_directories(rag-security PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PROTO_DIR}
        ${Protobuf_INCLUDE_DIRS}
)

# ============================================================================
# LINKING (DEBE IR DESPUÉS DE ADD_EXECUTABLE)
# ============================================================================
target_link_libraries(rag-security PRIVATE
        PkgConfig::ZMQ
        ${Protobuf_LIBRARIES}
        Threads::Threads
)

# nlohmann/json
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(rag-security PRIVATE nlohmann_json::nlohmann_json)
elseif(TARGET nlohmann_json)
    target_link_libraries(rag-security PRIVATE nlohmann_json)
endif()

# llama.cpp (si está disponible)
if(HAVE_LLAMA_CPP)
    if(TARGET llama)
        target_link_libraries(rag-security PRIVATE llama ggml)
        target_include_directories(rag-security PRIVATE ${LLAMA_INCLUDE_DIRS})
        target_compile_definitions(rag-security PRIVATE HAVE_LLAMA_CPP)
        message(STATUS "✅ llama.cpp integration enabled")
    endif()
endif()

# etcd (opcional)
if(HAVE_ETCD_CPP_API)
    if(TARGET etcd-cpp-api)
        target_link_libraries(rag-security PRIVATE etcd-cpp-api)
        target_compile_definitions(rag-security PRIVATE HAVE_ETCD_CPP_API)
    endif()
endif()

# ============================================================================
# FLAGS DE COMPILACIÓN
# ============================================================================
target_compile_options(rag-security PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(rag-security PRIVATE -g -O0)
else()
    target_compile_options(rag-security PRIVATE -O2)
endif()

# ============================================================================
# VERIFICACIÓN FINAL
# ============================================================================
message(STATUS "")
message(STATUS "🎯 BUILD SUMMARY")
message(STATUS "======================================")
message(STATUS "Target:            rag-security")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "ZeroMQ:            ${ZMQ_VERSION}")
message(STATUS "Protobuf:          Found")
message(STATUS "llama.cpp:         ${HAVE_LLAMA_CPP}")
message(STATUS "etcd-cpp-api:      ${HAVE_ETCD_CPP_API}")
message(STATUS "======================================")
message(STATUS "")