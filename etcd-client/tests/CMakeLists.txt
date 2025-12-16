# etcd-client/tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# Test: Compression
add_executable(test_compression test_compression.cpp)
target_link_libraries(test_compression
    PRIVATE
        ${LZ4_LIBRARIES}
)
target_include_directories(test_compression
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${LZ4_INCLUDE_DIRS}
)

# Add compression source directly (for testing internal functions)
target_sources(test_compression PRIVATE
    ${CMAKE_SOURCE_DIR}/src/compression_lz4.cpp
)

# Test: Encryption
add_executable(test_encryption test_encryption.cpp)

if(LIBSODIUM_FOUND)
    target_compile_definitions(test_encryption PRIVATE HAVE_LIBSODIUM)
    target_link_libraries(test_encryption
        PRIVATE
            ${LIBSODIUM_LIBRARIES}
    )
    target_include_directories(test_encryption
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${LIBSODIUM_INCLUDE_DIRS}
    )
    
    # Add crypto source directly
    target_sources(test_encryption PRIVATE
        ${CMAKE_SOURCE_DIR}/src/crypto_chacha20.cpp
    )
endif()

# Add tests to CTest
add_test(NAME compression COMMAND test_compression)
add_test(NAME encryption COMMAND test_encryption)

# Print test info
message(STATUS "Tests configured:")
message(STATUS "  - test_compression (LZ4)")
if(LIBSODIUM_FOUND)
    message(STATUS "  - test_encryption (ChaCha20-Poly1305)")
else()
    message(STATUS "  - test_encryption (SKIPPED - libsodium not found)")
endif()

# Test: Complete Pipeline (Compression + Encryption)
add_executable(test_pipeline test_pipeline.cpp)

if(LIBSODIUM_FOUND)
    target_compile_definitions(test_pipeline PRIVATE HAVE_LIBSODIUM)
    target_link_libraries(test_pipeline
        PRIVATE
            ${LIBSODIUM_LIBRARIES}
            ${LZ4_LIBRARIES}
    )
    target_include_directories(test_pipeline
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${LIBSODIUM_INCLUDE_DIRS}
            ${LZ4_INCLUDE_DIRS}
    )
    
    # Add both sources
    target_sources(test_pipeline PRIVATE
        ${CMAKE_SOURCE_DIR}/src/crypto_chacha20.cpp
        ${CMAKE_SOURCE_DIR}/src/compression_lz4.cpp
    )
    
    # Add to CTest
    add_test(NAME pipeline COMMAND test_pipeline)
    
    message(STATUS "  - test_pipeline (Complete pipeline)")
endif()
