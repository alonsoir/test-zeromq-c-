cmake_minimum_required(VERSION 3.16)
project(etcd-server)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Buscar dependencias
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

pkg_check_modules(CRYPTOPP REQUIRED libcrypto++)

# Find libsodium for ChaCha20
pkg_check_modules(LIBSODIUM REQUIRED libsodium)
if(LIBSODIUM_FOUND)
    message(STATUS "✅ libsodium found - Using ChaCha20-Poly1305")
else()
    message(FATAL_ERROR "libsodium not found - Install: sudo apt-get install libsodium-dev")
endif()

# Find LZ4 for compression
pkg_check_modules(LZ4 REQUIRED liblz4)
if(NOT LZ4_FOUND)
    message(FATAL_ERROR "liblz4 not found. Install: sudo apt-get install liblz4-dev")
endif()
message(STATUS "✅ LZ4 found - Compression enabled")

# Directorios de inclusión
include_directories(include)
include_directories(/usr/local/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Fuentes del servidor
set(SOURCES
        src/main.cpp
        src/etcd_server.cpp
        src/component_registry.cpp
        src/crypto_manager.cpp
        src/compression_lz4.cpp
)

# Ejecutable principal
add_executable(etcd-server ${SOURCES})

# Enlazar dependencias
# Enlazar Crypto++
target_link_libraries(etcd-server
        PRIVATE
        Threads::Threads
        ssl
        crypto
        ${CRYPTOPP_LIBRARIES}
        ${LIBSODIUM_LIBRARIES}
        ${LZ4_LIBRARIES}
)

# Incluir directorios de Crypto++
target_include_directories(etcd-server
                            PRIVATE
                            ${CRYPTOPP_INCLUDE_DIRS}
                            ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${LIBSODIUM_INCLUDE_DIRS}
                            ${LZ4_INCLUDE_DIRS})

# Opciones de compilación
target_compile_options(etcd-server PRIVATE -Wall -Wextra -O2)