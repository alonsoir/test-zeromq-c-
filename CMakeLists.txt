cmake_minimum_required(VERSION 3.20)
project(ZeroMQ_POC)

set(CMAKE_CXX_STANDARD 20)

# Buscar ZeroMQ
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

# Service 1 (PULL)
add_executable(service1 service1/main.cpp)
target_link_libraries(service1 ${ZMQ_LIBRARIES})
target_include_directories(service1 PRIVATE ${ZMQ_INCLUDE_DIRS})
target_compile_options(service1 PRIVATE ${ZMQ_CFLAGS_OTHER})

# Service 2 (PUSH)
add_executable(service2 service2/main.cpp)
target_link_libraries(service2 ${ZMQ_LIBRARIES})
target_include_directories(service2 PRIVATE ${ZMQ_INCLUDE_DIRS})
target_compile_options(service2 PRIVATE ${ZMQ_CFLAGS_OTHER})