troubleshooting main.cpp firewall-acl-agent.txt
Estamos trabajando en el main.cpp del firewall-acl-agent

Esta es la salida del compilador:

[ 85%] Building CXX object _deps/googletest-build/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o
/vagrant/firewall-acl-agent/src/main.cpp:49:5: error: ‘BatchConfig’ does not name a type
49 |     BatchConfig batch;
|     ^~~~~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:52:5: error: ‘ZMQConfig’ does not name a type; did you mean ‘Config’?
52 |     ZMQConfig zmq;
|     ^~~~~~~~~
|     Config
/vagrant/firewall-acl-agent/src/main.cpp: In function ‘bool load_config(const std::string&, Config&)’:
/vagrant/firewall-acl-agent/src/main.cpp:91:26: error: ‘struct mldefender::firewall::IPSetConfig’ has no member named ‘set_name’
91 |             config.ipset.set_name = ipset.get("set_name", "ml_defender_blacklist").asString();
|                          ^~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:92:26: error: ‘struct mldefender::firewall::IPSetConfig’ has no member named ‘set_type’
92 |             config.ipset.set_type = ipset.get("set_type", "hash:ip").asString();
|                          ^~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:93:26: error: ‘struct mldefender::firewall::IPSetConfig’ has no member named ‘hash_size’; did you mean ‘hashsize’?
93 |             config.ipset.hash_size = ipset.get("hash_size", 4096).asUInt();
|                          ^~~~~~~~~
|                          hashsize
/vagrant/firewall-acl-agent/src/main.cpp:94:26: error: ‘struct mldefender::firewall::IPSetConfig’ has no member named ‘max_elements’
94 |             config.ipset.max_elements = ipset.get("max_elements", 1000000).asUInt();
|                          ^~~~~~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:95:26: error: ‘struct mldefender::firewall::IPSetConfig’ has no member named ‘default_timeout’
95 |             config.ipset.default_timeout = ipset.get("timeout", 3600).asUInt();
|                          ^~~~~~~~~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:101:20: error: ‘struct Config’ has no member named ‘batch’
101 |             config.batch.batch_size_threshold = batch.get("batch_size_threshold", 1000).asUInt();
|                    ^~~~~
/vagrant/firewall-acl-agent/src/main.cpp:102:20: error: ‘struct Config’ has no member named ‘batch’
102 |             config.batch.batch_time_threshold_ms = batch.get("batch_time_threshold_ms", 100).asUInt();
|                    ^~~~~
/vagrant/firewall-acl-agent/src/main.cpp:103:20: error: ‘struct Config’ has no member named ‘batch’
103 |             config.batch.max_pending_ips = batch.get("max_pending_ips", 10000).asUInt();
|                    ^~~~~
/vagrant/firewall-acl-agent/src/main.cpp:104:20: error: ‘struct Config’ has no member named ‘batch’
104 |             config.batch.min_confidence = batch.get("min_confidence", 0.5).asFloat();
|                    ^~~~~
/vagrant/firewall-acl-agent/src/main.cpp:105:20: error: ‘struct Config’ has no member named ‘batch’
105 |             config.batch.enable_dedup = batch.get("enable_dedup", true).asBool();
|                    ^~~~~
/vagrant/firewall-acl-agent/src/main.cpp:111:20: error: ‘struct Config’ has no member named ‘zmq’
111 |             config.zmq.endpoint = zmq.get("endpoint", "tcp://localhost:5555").asString();
|                    ^~~
/vagrant/firewall-acl-agent/src/main.cpp:112:20: error: ‘struct Config’ has no member named ‘zmq’
112 |             config.zmq.topic = zmq.get("topic", "").asString();
|                    ^~~
/vagrant/firewall-acl-agent/src/main.cpp:113:20: error: ‘struct Config’ has no member named ‘zmq’
113 |             config.zmq.recv_timeout_ms = zmq.get("recv_timeout_ms", 1000).asInt();
|                    ^~~
/vagrant/firewall-acl-agent/src/main.cpp:114:20: error: ‘struct Config’ has no member named ‘zmq’
114 |             config.zmq.linger_ms = zmq.get("linger_ms", 1000).asInt();
|                    ^~~
/vagrant/firewall-acl-agent/src/main.cpp:115:20: error: ‘struct Config’ has no member named ‘zmq’
115 |             config.zmq.reconnect_interval_ms = zmq.get("reconnect_interval_ms", 1000).asInt();
|                    ^~~
/vagrant/firewall-acl-agent/src/main.cpp:116:20: error: ‘struct Config’ has no member named ‘zmq’
116 |             config.zmq.max_reconnect_interval_ms = zmq.get("max_reconnect_interval_ms", 30000).asInt();
|                    ^~~
/vagrant/firewall-acl-agent/src/main.cpp:117:20: error: ‘struct Config’ has no member named ‘zmq’
117 |             config.zmq.reconnect_backoff_multiplier = zmq.get("reconnect_backoff_multiplier", 2.0).asDouble();
|                    ^~~
/vagrant/firewall-acl-agent/src/main.cpp:118:20: error: ‘struct Config’ has no member named ‘zmq’
118 |             config.zmq.rcvhwm = zmq.get("rcvhwm", 1000).asInt();
|                    ^~~
/vagrant/firewall-acl-agent/src/main.cpp:119:20: error: ‘struct Config’ has no member named ‘zmq’
119 |             config.zmq.enable_stats = zmq.get("enable_stats", true).asBool();
|                    ^~~
/vagrant/firewall-acl-agent/src/main.cpp:120:20: error: ‘struct Config’ has no member named ‘zmq’
120 |             config.zmq.stats_interval_sec = zmq.get("stats_interval_sec", 60).asInt();
|                    ^~~
/vagrant/firewall-acl-agent/src/main.cpp: In function ‘Config create_default_config()’:
/vagrant/firewall-acl-agent/src/main.cpp:162:18: error: ‘struct mldefender::firewall::IPSetConfig’ has no member named ‘set_name’
162 |     config.ipset.set_name = "ml_defender_blacklist";
|                  ^~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:163:18: error: ‘struct mldefender::firewall::IPSetConfig’ has no member named ‘set_type’
163 |     config.ipset.set_type = "hash:ip";
|                  ^~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:164:18: error: ‘struct mldefender::firewall::IPSetConfig’ has no member named ‘hash_size’; did you mean ‘hashsize’?
164 |     config.ipset.hash_size = 4096;
|                  ^~~~~~~~~
|                  hashsize
/vagrant/firewall-acl-agent/src/main.cpp:165:18: error: ‘struct mldefender::firewall::IPSetConfig’ has no member named ‘max_elements’
165 |     config.ipset.max_elements = 1000000;
|                  ^~~~~~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:166:18: error: ‘struct mldefender::firewall::IPSetConfig’ has no member named ‘default_timeout’
166 |     config.ipset.default_timeout = 3600;
|                  ^~~~~~~~~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:169:12: error: ‘struct Config’ has no member named ‘batch’
169 |     config.batch.batch_size_threshold = 1000;
|            ^~~~~
/vagrant/firewall-acl-agent/src/main.cpp:170:12: error: ‘struct Config’ has no member named ‘batch’
170 |     config.batch.batch_time_threshold_ms = 100;
|            ^~~~~
/vagrant/firewall-acl-agent/src/main.cpp:171:12: error: ‘struct Config’ has no member named ‘batch’
171 |     config.batch.max_pending_ips = 10000;
|            ^~~~~
/vagrant/firewall-acl-agent/src/main.cpp:172:12: error: ‘struct Config’ has no member named ‘batch’
172 |     config.batch.min_confidence = 0.5f;
|            ^~~~~
/vagrant/firewall-acl-agent/src/main.cpp:173:12: error: ‘struct Config’ has no member named ‘batch’
173 |     config.batch.enable_dedup = true;
|            ^~~~~
/vagrant/firewall-acl-agent/src/main.cpp:176:12: error: ‘struct Config’ has no member named ‘zmq’
176 |     config.zmq.endpoint = "tcp://localhost:5555";
|            ^~~
/vagrant/firewall-acl-agent/src/main.cpp:177:12: error: ‘struct Config’ has no member named ‘zmq’
177 |     config.zmq.topic = "";
|            ^~~
/vagrant/firewall-acl-agent/src/main.cpp:178:12: error: ‘struct Config’ has no member named ‘zmq’
178 |     config.zmq.recv_timeout_ms = 1000;
|            ^~~
/vagrant/firewall-acl-agent/src/main.cpp:179:12: error: ‘struct Config’ has no member named ‘zmq’
179 |     config.zmq.linger_ms = 1000;
|            ^~~
/vagrant/firewall-acl-agent/src/main.cpp:180:12: error: ‘struct Config’ has no member named ‘zmq’
180 |     config.zmq.reconnect_interval_ms = 1000;
|            ^~~
/vagrant/firewall-acl-agent/src/main.cpp:181:12: error: ‘struct Config’ has no member named ‘zmq’
181 |     config.zmq.max_reconnect_interval_ms = 30000;
|            ^~~
/vagrant/firewall-acl-agent/src/main.cpp:182:12: error: ‘struct Config’ has no member named ‘zmq’
182 |     config.zmq.reconnect_backoff_multiplier = 2.0;
|            ^~~
/vagrant/firewall-acl-agent/src/main.cpp:183:12: error: ‘struct Config’ has no member named ‘zmq’
183 |     config.zmq.rcvhwm = 1000;
|            ^~~
/vagrant/firewall-acl-agent/src/main.cpp:184:12: error: ‘struct Config’ has no member named ‘zmq’
184 |     config.zmq.enable_stats = true;
|            ^~~
/vagrant/firewall-acl-agent/src/main.cpp:185:12: error: ‘struct Config’ has no member named ‘zmq’
185 |     config.zmq.stats_interval_sec = 60;
|            ^~~
/vagrant/firewall-acl-agent/src/main.cpp: In function ‘void export_metrics(const Config&, const mldefender::firewall::ZMQSubscriber&, const mldefender::firewall::BatchProcessor&)’:
/vagrant/firewall-acl-agent/src/main.cpp:261:40: error: ‘const class mldefender::firewall::BatchProcessor’ has no member named ‘get_stats’
261 |     const auto& proc_stats = processor.get_stats();
|                                        ^~~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp: In function ‘bool perform_health_checks(const Config&, mldefender::firewall::IPSetWrapper&, mldefender::firewall::IPTablesWrapper&, const mldefender::firewall::ZMQSubscriber&)’:
/vagrant/firewall-acl-agent/src/main.cpp:343:16: error: ‘class mldefender::firewall::IPSetWrapper’ has no member named ‘exists’; did you mean ‘set_exists’?
343 |     if (!ipset.exists(config.ipset.set_name)) {
|                ^~~~~~
|                set_exists
/vagrant/firewall-acl-agent/src/main.cpp:343:36: error: ‘const struct mldefender::firewall::IPSetConfig’ has no member named ‘set_name’
343 |     if (!ipset.exists(config.ipset.set_name)) {
|                                    ^~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:344:59: error: ‘const struct mldefender::firewall::IPSetConfig’ has no member named ‘set_name’
344 |         std::cerr << "[HEALTH] ✗ IPSet '" << config.ipset.set_name << "' does not exist!" << std::endl;
|                                                           ^~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:352:36: error: ‘const struct mldefender::firewall::IPSetConfig’ has no member named ‘set_name’
352 |         if (rule.find(config.ipset.set_name) != std::string::npos) {
|                                    ^~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:359:71: error: ‘const struct mldefender::firewall::IPSetConfig’ has no member named ‘set_name’
359 |         std::cerr << "[HEALTH] ✗ IPTables rule for '" << config.ipset.set_name << "' not found!" << std::endl;
|                                                                       ^~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp: In function ‘int main(int, char**)’:
/vagrant/firewall-acl-agent/src/main.cpp:437:40: error: no matching function for call to ‘mldefender::firewall::IPSetWrapper::IPSetWrapper(mldefender::firewall::IPSetConfig&)’
437 |         IPSetWrapper ipset(config.ipset);
|                                        ^
In file included from /vagrant/firewall-acl-agent/src/main.cpp:15:
/vagrant/firewall-acl-agent/include/firewall/ipset_wrapper.hpp:184:5: note: candidate: ‘mldefender::firewall::IPSetWrapper::IPSetWrapper()’
184 |     IPSetWrapper();
|     ^~~~~~~~~~~~
/vagrant/firewall-acl-agent/include/firewall/ipset_wrapper.hpp:184:5: note:   candidate expects 0 arguments, 1 provided
/vagrant/firewall-acl-agent/src/main.cpp:440:20: error: ‘class mldefender::firewall::IPSetWrapper’ has no member named ‘exists’; did you mean ‘set_exists’?
440 |         if (!ipset.exists(config.ipset.set_name)) {
|                    ^~~~~~
|                    set_exists
/vagrant/firewall-acl-agent/src/main.cpp:440:40: error: ‘struct mldefender::firewall::IPSetConfig’ has no member named ‘set_name’
440 |         if (!ipset.exists(config.ipset.set_name)) {
|                                        ^~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:441:68: error: ‘struct mldefender::firewall::IPSetConfig’ has no member named ‘set_name’
441 |             std::cerr << "[INIT] Creating ipset '" << config.ipset.set_name << "'..." << std::endl;
|                                                                    ^~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:442:48: error: ‘struct mldefender::firewall::IPSetConfig’ has no member named ‘set_name’
442 |             if (!ipset.create_set(config.ipset.set_name)) {
|                                                ^~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:457:53: error: ‘struct mldefender::firewall::IPSetConfig’ has no member named ‘set_name’
457 |         if (!iptables.setup_base_rules(config.ipset.set_name)) {
|                                                     ^~~~~~~~
/vagrant/firewall-acl-agent/src/main.cpp:465:48: error: ‘struct Config’ has no member named ‘batch’
465 |         BatchProcessor processor(ipset, config.batch);
|                                                ^~~~~
/vagrant/firewall-acl-agent/src/main.cpp:471:52: error: ‘struct Config’ has no member named ‘zmq’
471 |         ZMQSubscriber subscriber(processor, config.zmq);
|                                                    ^~~
[ 90%] Linking CXX static library ../../../lib/libgmock_main.a
make[2]: *** [CMakeFiles/firewall-acl-agent.dir/build.make:76: CMakeFiles/firewall-acl-agent.dir/src/main.cpp.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:195: CMakeFiles/firewall-acl-agent.dir/all] Error 2
make[1]: *** Se espera a que terminen otras tareas....
[ 90%] Built target gmock_main
[ 95%] Linking CXX executable firewall_tests
lto-wrapper: warning: using serial compilation of 8 LTRANS jobs
lto-wrapper: note: see the ‘-flto’ option documentation for more information
[ 95%] Built target firewall_tests
make: *** [Makefile:146: all] Error 2

Este es el CMakelists.txt

#===----------------------------------------------------------------------===//
# ML Defender - Firewall ACL Agent
# CMakeLists.txt - Build Configuration
#
# Target: Ultra-high performance packet DROP agent
# Performance: 1M+ packets/sec on commodity hardware
#
# Dependencies:
#   - libipset (kernel interface)
#   - iptables (static rules setup)
#   - ZMQ (detection stream from ml-detector)
#   - Protobuf (message parsing)
#   - Boost (lock-free queues)
#   - jsoncpp (configuration)
#
# Via Appia Quality: Methodical build system
#===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.20)

project(firewall-acl-agent
VERSION 1.0.0
DESCRIPTION "ML Defender - High-Performance Firewall ACL Agent"
LANGUAGES CXX
)

#===----------------------------------------------------------------------===//
# C++ Standard and Compiler Settings
#===----------------------------------------------------------------------===//

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags for extreme performance
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address,undefined")

# Warning flags
add_compile_options(
-Wall
-Wextra
-Wpedantic
-Werror
-Wno-unused-parameter
)

#===----------------------------------------------------------------------===//
# Find Required Dependencies
#===----------------------------------------------------------------------===//

find_package(PkgConfig REQUIRED)

# ZeroMQ - High-performance messaging
pkg_check_modules(ZMQ REQUIRED libzmq)
if(NOT ZMQ_FOUND)
message(FATAL_ERROR "ZeroMQ not found. Install: sudo apt-get install libzmq3-dev")
endif()

# NOTE: We do NOT need libipset-dev - we use system ipset commands
# This is simpler, more maintainable, and benefits from ipset optimizations

# Protobuf - Message serialization
find_package(Protobuf REQUIRED)
if(NOT Protobuf_FOUND)
message(FATAL_ERROR "Protobuf not found. Install: sudo apt-get install libprotobuf-dev protobuf-compiler")
endif()

# Boost - Lock-free data structures
find_package(Boost 1.71 REQUIRED COMPONENTS
system
thread
filesystem
)
if(NOT Boost_FOUND)
message(FATAL_ERROR "Boost not found. Install: sudo apt-get install libboost-all-dev")
endif()

# jsoncpp - Configuration parsing
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
if(NOT JSONCPP_FOUND)
message(FATAL_ERROR "jsoncpp not found. Install: sudo apt-get install libjsoncpp-dev")
endif()

# Threads
find_package(Threads REQUIRED)

#===----------------------------------------------------------------------===//
# Protobuf Generation - Shared Schema
#===----------------------------------------------------------------------===//

# Path to shared protobuf schema
# TODO: Update this path to point to the real shared protobuf once project structure is set
# Real path should be: ../../protobuf/network_security.proto (shared between all components)
# For now, using local protobuf directory for development
set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../protobuf)
set(PROTO_FILE ${PROTO_DIR}/network_security.proto)

# Verify proto file exists
if(NOT EXISTS ${PROTO_FILE})
message(FATAL_ERROR
"Protobuf schema not found: ${PROTO_FILE}\n"
"Expected structure: ml-defender/protobuf/network_security.proto"
)
endif()

# Generate C++ files from .proto (into build directory)
protobuf_generate_cpp(PROTO_SOURCES PROTO_HEADERS ${PROTO_FILE})

message(STATUS "Protobuf schema: ${PROTO_FILE}")
message(STATUS "Generated sources: ${PROTO_SOURCES}")
message(STATUS "Generated headers: ${PROTO_HEADERS}")

# Create protobuf library
add_library(firewall_proto STATIC
${PROTO_SOURCES}
${PROTO_HEADERS}
)

target_link_libraries(firewall_proto
PUBLIC
${Protobuf_LIBRARIES}
)

target_include_directories(firewall_proto
PUBLIC
${CMAKE_CURRENT_BINARY_DIR}  # Generated protobuf headers
${PROTO_DIR}                 # Original .proto location
)

#===----------------------------------------------------------------------===//
# Include Directories
#===----------------------------------------------------------------------===//

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/include
${CMAKE_CURRENT_BINARY_DIR}      # Generated protobuf headers

        ${ZMQ_INCLUDE_DIRS}
        ${Protobuf_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${JSONCPP_INCLUDE_DIRS}
)

#===----------------------------------------------------------------------===//
# Source Files (Only files that exist)
#===----------------------------------------------------------------------===//

# Core components (created so far)
set(FIREWALL_CORE_SOURCES
src/core/ipset_wrapper.cpp
src/core/iptables_wrapper.cpp
src/core/batch_processor.cpp
# src/core/acl_intelligence.cpp     # TODO: Next to implement
)

# API layer (to be created)
set(FIREWALL_API_SOURCES
src/api/zmq_subscriber.cpp
)

# Utilities (to be created)
set(FIREWALL_UTIL_SOURCES
# src/utils/config_loader.cpp       # TODO: Next to implement
# src/utils/logger.cpp              # TODO: Next to implement
# src/utils/metrics.cpp             # TODO: Next to implement
)

# Main executable (to be created)
set(FIREWALL_MAIN_SOURCE
src/main.cpp
)

#===----------------------------------------------------------------------===//
# Core Library (for testing) - Only with existing sources
#===----------------------------------------------------------------------===//

add_library(firewall_core STATIC
${FIREWALL_CORE_SOURCES}
${FIREWALL_API_SOURCES}   # Uncomment when implemented
# ${FIREWALL_UTIL_SOURCES}  # Uncomment when implemented
)

target_link_libraries(firewall_core
PUBLIC
firewall_proto

        ${ZMQ_LIBRARIES}
        ${Boost_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        Threads::Threads
)

target_include_directories(firewall_core
PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}/include
${CMAKE_CURRENT_BINARY_DIR}  # For generated protobuf headers

        ${ZMQ_INCLUDE_DIRS}
        ${JSONCPP_INCLUDE_DIRS}
)

#===----------------------------------------------------------------------===//
# Main Executable (TODO: Uncomment when main.cpp is ready)
#===----------------------------------------------------------------------===//

add_executable(firewall-acl-agent
${FIREWALL_MAIN_SOURCE}
)
#
target_link_libraries(firewall-acl-agent
PRIVATE
firewall_core
)

# NOTE: Executable will be created once we have src/main.cpp
message(STATUS "⚠️  Main executable disabled - waiting for src/main.cpp")
message(STATUS "   Current focus: Core library and unit tests")

#===----------------------------------------------------------------------===//
# Installation (Disabled until executable is ready)
#===----------------------------------------------------------------------===//

# TODO: Uncomment when firewall-acl-agent executable exists
install(TARGETS firewall-acl-agent
RUNTIME DESTINATION bin
)
#
install(DIRECTORY config/
DESTINATION etc/ml-defender/firewall
FILES_MATCHING PATTERN "*.json"
)
#
install(FILES systemd/firewall-acl-agent.service
DESTINATION /etc/systemd/system/
OPTIONAL
)

message(STATUS "📦 Installation targets disabled - waiting for main executable")

#===----------------------------------------------------------------------===//
# Testing
#===----------------------------------------------------------------------===//

option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
enable_testing()

    # Google Test
    find_package(GTest)
    if(NOT GTest_FOUND)
        message(STATUS "GTest not found, fetching from GitHub...")
        include(FetchContent)
        FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG release-1.12.1
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    # Test sources (only existing tests)
    set(TEST_SOURCES
            tests/unit/test_ipset_wrapper.cpp
            # tests/test_batch_processor.cpp     # TODO: Create when batch_processor is ready
            # tests/test_acl_intelligence.cpp    # TODO: Create when acl_intelligence is ready
    )

    add_executable(firewall_tests
            ${TEST_SOURCES}
    )

    target_link_libraries(firewall_tests
            PRIVATE
            firewall_core
            GTest::gtest
            GTest::gtest_main
    )

    # Discover tests
    include(GoogleTest)
    gtest_discover_tests(firewall_tests)

    message(STATUS "✅ Unit tests enabled")
    message(STATUS "   Run: sudo ./firewall_tests  (requires root for ipset operations)")
endif()

#===----------------------------------------------------------------------===//
# Benchmarks
#===----------------------------------------------------------------------===//

option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)

if(BUILD_BENCHMARKS)
# Google Benchmark
find_package(benchmark)
if(NOT benchmark_FOUND)
message(STATUS "Google Benchmark not found, fetching...")
include(FetchContent)
FetchContent_Declare(
benchmark
GIT_REPOSITORY https://github.com/google/benchmark.git
GIT_TAG v1.8.3
)
FetchContent_MakeAvailable(benchmark)
endif()

    add_executable(firewall_bench
            benchmarks/bench_batch_operations.cpp
            benchmarks/bench_queue_throughput.cpp
    )

    target_link_libraries(firewall_bench
            PRIVATE
            firewall_core
            benchmark::benchmark
            benchmark::benchmark_main
    )

    message(STATUS "✅ Benchmarks enabled")
    message(STATUS "   Run: sudo ./firewall_bench  (requires root)")
endif()

#===----------------------------------------------------------------------===//
# Documentation
#===----------------------------------------------------------------------===//

option(BUILD_DOCS "Build documentation with Doxygen" OFF)

if(BUILD_DOCS)
find_package(Doxygen)
if(DOXYGEN_FOUND)
set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

        add_custom_target(docs
                COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Generating API documentation with Doxygen"
                VERBATIM
        )

        message(STATUS "✅ Documentation enabled: make docs")
    else()
        message(WARNING "Doxygen not found, documentation disabled")
    endif()
endif()

#===----------------------------------------------------------------------===//
# Performance Profiling Support
#===----------------------------------------------------------------------===//

option(ENABLE_PROFILING "Enable profiling with gprof/perf" OFF)

if(ENABLE_PROFILING)
add_compile_options(-pg -fno-omit-frame-pointer)
add_link_options(-pg)
message(STATUS "✅ Profiling enabled (-pg)")
endif()

#===----------------------------------------------------------------------===//
# Sanitizers (Development)
#===----------------------------------------------------------------------===//

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

if(ENABLE_ASAN)
add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
add_link_options(-fsanitize=address)
message(STATUS "✅ AddressSanitizer enabled")
endif()

if(ENABLE_TSAN)
add_compile_options(-fsanitize=thread)
add_link_options(-fsanitize=thread)
message(STATUS "✅ ThreadSanitizer enabled")
endif()

if(ENABLE_UBSAN)
add_compile_options(-fsanitize=undefined)
add_link_options(-fsanitize=undefined)
message(STATUS "✅ UndefinedBehaviorSanitizer enabled")
endif()

#===----------------------------------------------------------------------===//
# Build Summary
#===----------------------------------------------------------------------===//

message(STATUS "")
message(STATUS "╔════════════════════════════════════════════════════════╗")
message(STATUS "║  ML Defender - Firewall ACL Agent Configuration       ║")
message(STATUS "╚════════════════════════════════════════════════════════╝")
message(STATUS "")
message(STATUS "Version:           ${PROJECT_VERSION}")
message(STATUS "C++ Standard:      C++${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  ZeroMQ:          ${ZMQ_VERSION}")
message(STATUS "  Protobuf:        ${Protobuf_VERSION}")
message(STATUS "  Boost:           ${Boost_VERSION}")
message(STATUS "  jsoncpp:         ${JSONCPP_VERSION}")
message(STATUS "  NOTE: Using system ipset commands (no libipset dependency)")
message(STATUS "")
message(STATUS "Optional Features:")
message(STATUS "  Tests:           ${BUILD_TESTS}")
message(STATUS "  Benchmarks:      ${BUILD_BENCHMARKS}")
message(STATUS "  Documentation:   ${BUILD_DOCS}")
message(STATUS "  Profiling:       ${ENABLE_PROFILING}")
message(STATUS "")
message(STATUS "⚡ Performance Target: 1M+ packets/sec DROP rate")
message(STATUS "🎯 Design Philosophy: Via Appia Quality")
message(STATUS "")
message(STATUS "Build Commands:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake -DCMAKE_BUILD_TYPE=Release ..")
message(STATUS "  make -j$(nproc)")
message(STATUS "  sudo ./firewall-acl-agent -c ../config/firewall.json")
message(STATUS "")
message(STATUS "╚════════════════════════════════════════════════════════╝")
message(STATUS "")

Este es el firewall.json actual:

{
"_comment": "ML Defender - Firewall ACL Agent Configuration",
"_version": "1.0.0",

"ipset": {
"set_name": "ml_defender_blacklist",
"set_type": "hash:ip",
"hash_size": 4096,
"max_elements": 1000000,
"timeout": 3600,
"comment": "ML Defender blocked IPs"
},

"iptables": {
"chain_name": "ML_DEFENDER_INPUT",
"default_policy": "ACCEPT",
"log_blocked": false,
"log_prefix": "ML_DEFENDER_DROP: "
},

"batch_processor": {
"batch_size_threshold": 1000,
"batch_time_threshold_ms": 100,
"max_pending_ips": 10000,
"min_confidence": 0.5,
"enable_dedup": true,
"flush_on_shutdown": true
},

"zmq": {
"endpoint": "tcp://localhost:5555",
"topic": "",
"recv_timeout_ms": 1000,
"linger_ms": 1000,
"reconnect_interval_ms": 1000,
"max_reconnect_interval_ms": 30000,
"reconnect_backoff_multiplier": 2.0,
"rcvhwm": 1000,
"enable_stats": true,
"stats_interval_sec": 60
},

"daemon": {
"daemonize": false,
"pid_file": "/var/run/firewall-acl-agent.pid",
"user": "root",
"group": "root"
},

"logging": {
"level": "info",
"console": true,
"syslog": false,
"file": "",
"max_file_size_mb": 100
},

"metrics": {
"enable_export": true,
"export_interval_sec": 60,
"export_format": "json",
"export_file": "/var/log/ml-defender/firewall-metrics.json"
},

"health_check": {
"enable": true,
"check_interval_sec": 30,
"ipset_health_check": true,
"iptables_health_check": true,
"zmq_connection_check": true
}
}