cmake_minimum_required(VERSION 3.16)
project(plugin_hello VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# plugin-loader headers (plugin_api.h)
find_path(PLUGIN_LOADER_INCLUDE_DIR
    NAMES plugin_loader/plugin_api.h
    PATHS
        ${CMAKE_SOURCE_DIR}/../../plugin-loader/include
        /usr/local/include
        /usr/include
)

if(NOT PLUGIN_LOADER_INCLUDE_DIR)
    message(FATAL_ERROR "plugin_api.h not found. Build plugin-loader first.")
endif()

add_library(plugin_hello SHARED hello_plugin.cpp)

target_include_directories(plugin_hello PRIVATE ${PLUGIN_LOADER_INCLUDE_DIR})

set_target_properties(plugin_hello PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "plugin_hello"
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(plugin_hello PRIVATE -Wall -Wextra)

message(STATUS "plugin_hello: will build libplugin_hello.so")
