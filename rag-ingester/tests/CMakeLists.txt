# Test executable for config parser
add_executable(test_config_parser 
    test_config_parser.cpp
    ../src/common/config_parser.cpp  # ← AGREGADO: source del config_parser
)

target_include_directories(test_config_parser 
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_config_parser
    PRIVATE
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

# Add test
add_test(NAME test_config_parser COMMAND test_config_parser)
