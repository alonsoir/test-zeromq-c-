cmake_minimum_required(VERSION 3.16)
project(etcd-server)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Buscar dependencias
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

pkg_check_modules(CRYPTOPP REQUIRED libcrypto++)

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
)

# Incluir directorios de Crypto++
target_include_directories(etcd-server PRIVATE ${CRYPTOPP_INCLUDE_DIRS})

# Opciones de compilación
target_compile_options(etcd-server PRIVATE -Wall -Wextra -O2)