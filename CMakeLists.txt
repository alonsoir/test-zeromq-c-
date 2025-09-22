cmake_minimum_required(VERSION 3.16)
project(DDOSPipeline VERSION 1.0.0 LANGUAGES CXX)

# Configuración del compilador
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Configuración de build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Buscar dependencias
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)

# ZeroMQ
pkg_check_modules(ZMQ REQUIRED libzmq)

# JsonCpp
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

# Threads
find_package(Threads REQUIRED)

# Generar archivos protobuf
set(PROTO_FILES protobuf/network_security.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# Directorio para archivos generados
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Archivos comunes
set(COMMON_SOURCES
        common/EtcdServiceRegistry.cpp
        ${PROTO_SRCS}
)

# Service1 executable
add_executable(service1_exe
        service1/main.cpp
        ${COMMON_SOURCES}
)

target_link_libraries(service1_exe
        ${Protobuf_LIBRARIES}
        ${ZMQ_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        etcd-cpp-api
        Threads::Threads
)

target_include_directories(service1_exe PRIVATE
        ${Protobuf_INCLUDE_DIRS}
        ${ZMQ_INCLUDE_DIRS}
        ${JSONCPP_INCLUDE_DIRS}
)

# Service2 executable
add_executable(service2_exe
        service2/main.cpp
        ${COMMON_SOURCES}
)

target_link_libraries(service2_exe
        ${Protobuf_LIBRARIES}
        ${ZMQ_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        etcd-cpp-api
        Threads::Threads
)

target_include_directories(service2_exe PRIVATE
        ${Protobuf_INCLUDE_DIRS}
        ${ZMQ_INCLUDE_DIRS}
        ${JSONCPP_INCLUDE_DIRS}
)

# Configurar propiedades de los ejecutables
set_target_properties(service1_exe service2_exe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Instalar ejecutables
install(TARGETS service1_exe service2_exe
        RUNTIME DESTINATION bin
)

# Target para tests
enable_testing()

add_custom_target(check
        COMMAND echo "Verificando compilación..."
        COMMAND test -f ${CMAKE_BINARY_DIR}/bin/service1_exe && echo "✅ service1_exe OK" || echo "❌ service1_exe faltante"
        COMMAND test -f ${CMAKE_BINARY_DIR}/bin/service2_exe && echo "✅ service2_exe OK" || echo "❌ service2_exe faltante"
        DEPENDS service1_exe service2_exe
)

# Target para limpiar todo
add_custom_target(clean-all
        COMMAND ${CMAKE_BUILD_TOOL} clean
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/bin
)

# Información de debug
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Protobuf version: ${Protobuf_VERSION}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

# Crear directorio de salida
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)