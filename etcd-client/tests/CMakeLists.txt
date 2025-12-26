# etcd-client/tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# Find crypto-transport library
find_library(CRYPTO_TRANSPORT_LIB crypto_transport PATHS /usr/local/lib REQUIRED)

# Test: Compression
add_executable(test_compression test_compression.cpp)
target_link_libraries(test_compression
        PRIVATE
        ${CRYPTO_TRANSPORT_LIB}
)
target_include_directories(test_compression
        PRIVATE
        /usr/local/include
)

# Test: Encryption
add_executable(test_encryption test_encryption.cpp)
target_link_libraries(test_encryption
        PRIVATE
        ${CRYPTO_TRANSPORT_LIB}
)
target_include_directories(test_encryption
        PRIVATE
        /usr/local/include
)

# Test: Complete Pipeline (Compression + Encryption)
add_executable(test_pipeline test_pipeline.cpp)
target_link_libraries(test_pipeline
        PRIVATE
        ${CRYPTO_TRANSPORT_LIB}
)
target_include_directories(test_pipeline
        PRIVATE
        /usr/local/include
)

# Test: Integration test (uses etcd_client library)
add_executable(test_put_config_integration test_put_config_integration.cpp)
target_link_libraries(test_put_config_integration
        PRIVATE
        etcd_client
)
target_include_directories(test_put_config_integration
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Add tests to CTest
add_test(NAME compression COMMAND test_compression)
add_test(NAME encryption COMMAND test_encryption)
add_test(NAME pipeline COMMAND test_pipeline)

# Print test info
message(STATUS "========================================")
message(STATUS "etcd-client Tests Configuration")
message(STATUS "========================================")
message(STATUS "Tests configured:")
message(STATUS "  - test_compression (crypto-transport)")
message(STATUS "  - test_encryption (crypto-transport)")
message(STATUS "  - test_pipeline (crypto-transport)")
message(STATUS "  - test_put_config_integration (etcd_client)")
message(STATUS "========================================")