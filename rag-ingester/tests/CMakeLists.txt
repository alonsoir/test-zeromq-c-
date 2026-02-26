# tests/CMakeLists.txt
# RAG Ingester - Test Suite
# Day 36: config_parser + FileWatcher tests
# Day 67: CsvFileWatcher + CsvEventLoader tests
# Day 69: CsvDirWatcher + FirewallCsvEventLoader tests
#
# ⚠️  All test targets defined HERE only.
#     The root CMakeLists.txt does: enable_testing() + add_subdirectory(tests)
#     Do NOT define any test_* target in the root CMakeLists.txt.

# ============================================================================
# Test: config_parser (Day 36)
# ============================================================================
add_executable(test_config_parser
        test_config_parser.cpp
        ../src/common/config_parser.cpp
)
target_include_directories(test_config_parser
        PRIVATE ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_config_parser
        PRIVATE
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)
add_test(NAME test_config_parser COMMAND test_config_parser)

# ============================================================================
# Test: FileWatcher (Day 36) — inotify IN_CLOSE_WRITE for *.pb files
# ============================================================================
add_executable(test_file_watcher
        test_file_watcher.cpp
)
target_include_directories(test_file_watcher
        PRIVATE ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_file_watcher
        PRIVATE
        file_watcher
)
add_test(NAME test_file_watcher COMMAND test_file_watcher)

# ============================================================================
# Test: CsvFileWatcher (Day 67) — inotify IN_MODIFY + offset for CSV tail
# ============================================================================
add_executable(test_csv_file_watcher
        test_csv_file_watcher.cpp
)
target_include_directories(test_csv_file_watcher
        PRIVATE ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_csv_file_watcher
        PRIVATE
        csv_file_watcher
        Threads::Threads
)
add_test(NAME test_csv_file_watcher COMMAND test_csv_file_watcher)

# ============================================================================
# Test: CsvEventLoader (Day 67) — 127-column CSV parsing + HMAC-SHA256
# ============================================================================
add_executable(test_csv_event_loader
        test_csv_event_loader.cpp
)
target_include_directories(test_csv_event_loader
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${OPENSSL_INCLUDE_DIR}
)
target_link_libraries(test_csv_event_loader
        PRIVATE
        csv_event_loader
        OpenSSL::Crypto
)
add_test(NAME test_csv_event_loader COMMAND test_csv_event_loader)

# ============================================================================
# Test: CsvDirWatcher (Day 69) — inotify on directory, daily rotation
# ============================================================================
add_executable(test_csv_dir_watcher
        test_csv_dir_watcher.cpp
)
target_include_directories(test_csv_dir_watcher
        PRIVATE ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_csv_dir_watcher
        PRIVATE
        csv_dir_watcher
        Threads::Threads
)
add_test(NAME test_csv_dir_watcher COMMAND test_csv_dir_watcher)

# ============================================================================
# Test: FirewallCsvEventLoader (Day 69) — 7-column firewall CSV + HMAC-SHA256
# ============================================================================
add_executable(test_firewall_csv_event_loader
        test_firewall_csv_event_loader.cpp
)
target_include_directories(test_firewall_csv_event_loader
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${OPENSSL_INCLUDE_DIR}
)
target_link_libraries(test_firewall_csv_event_loader
        PRIVATE
        firewall_csv_event_loader
        OpenSSL::Crypto
)
add_test(NAME test_firewall_csv_event_loader COMMAND test_firewall_csv_event_loader)