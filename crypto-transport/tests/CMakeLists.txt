# crypto-transport/tests/CMakeLists.txt

# Find Google Test (optional, fallback to manual compilation)
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "GTest not found, using manual test compilation")
endif()

# Helper macro to add a test executable
macro(add_crypto_transport_test test_name)
    add_executable(${test_name} ${test_name}.cpp)

    target_link_libraries(${test_name}
            PRIVATE
            crypto_transport
    )

    target_include_directories(${test_name}
            PRIVATE
            ${CMAKE_SOURCE_DIR}/include
    )

    if(GTest_FOUND)
        target_link_libraries(${test_name} PRIVATE GTest::GTest GTest::Main)
    endif()

    add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

# Add individual test executables
add_crypto_transport_test(test_crypto)
add_crypto_transport_test(test_compression)
add_crypto_transport_test(test_integration)

# Summary
message(STATUS "========================================")
message(STATUS "crypto-transport Tests Configuration")
message(STATUS "========================================")
if(GTest_FOUND)
    message(STATUS "Test framework: Google Test")
else()
    message(STATUS "Test framework: Manual (no GTest)")
endif()
message(STATUS "Tests: test_crypto, test_compression, test_integration")
message(STATUS "========================================")