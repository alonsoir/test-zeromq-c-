// rag_logger.hpp - RAG Event Logger Header
// Day 16 - RACE CONDITION FIX Applied
// Authors: Alonso Isidoro Roman + Claude (Anthropic)
// Changes: Moved rotation check inside critical section

#pragma once

#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <fstream>
#include <chrono>

// Logging
#include <spdlog/spdlog.h>

// JSON
#include <nlohmann/json.hpp>

// Protobuf
#include "network_security.pb.h"
#include <crypto_transport/crypto_manager.hpp>
namespace ml_defender {

// ============================================================================
// Configuration Structure
// ============================================================================

struct RAGLoggerConfig {
    std::string base_path = "/vagrant/logs/rag";
    std::string deployment_id = "ml-defender-dev";
    std::string node_id = "detector-01";

    // Thresholds
    double min_score_to_log = 0.70;
    double min_divergence_to_log = 0.30;

    // Performance settings
    size_t max_events_per_file = 10000;
    size_t max_file_size_mb = 100;

    // Features
    bool save_protobuf_artifacts = true;
    bool save_json_artifacts = true;
};

// ============================================================================
// ML Context Structure
// ============================================================================

struct MLContext {
    // System state
    uint64_t events_processed_total = 0;
    uint64_t events_in_last_minute = 0;
    double memory_usage_mb = 0.0;
    double cpu_usage_percent = 0.0;
    uint64_t uptime_seconds = 0;

    // Detection context
    std::string attack_family;
    std::string level_1_label;
    std::string level_2_category;
    std::string level_3_subcategory;
    double level_1_confidence = 0.0;

    // Temporal window
    std::chrono::system_clock::time_point window_start;
    std::chrono::system_clock::time_point window_end;
    uint64_t events_in_window = 0;

    // Investigation priority
    std::string investigation_priority = "LOW";
};

// ============================================================================
// RAG Logger Class
// ============================================================================

class RAGLogger {
public:
    RAGLogger(const RAGLoggerConfig& config,
          std::shared_ptr<spdlog::logger> logger,
          std::shared_ptr<crypto::CryptoManager> crypto_manager);  // NUEVO
    ~RAGLogger();

    // Prevent copying
    RAGLogger(const RAGLogger&) = delete;
    RAGLogger& operator=(const RAGLogger&) = delete;

    // Main interface
    bool log_event(const protobuf::NetworkSecurityEvent& event,
                   const MLContext& context);

    void flush();
    void rotate_logs();
    nlohmann::json get_statistics() const;

private:
    // Configuration
    RAGLoggerConfig config_;
    std::shared_ptr<spdlog::logger> logger_;

    // File handling
    std::ofstream current_log_;
    std::string current_log_path_;
    std::string current_date_;
    std::atomic<uint64_t> events_in_current_file_{0};

    // Statistics
    std::atomic<uint64_t> events_logged_{0};
    std::atomic<uint64_t> events_skipped_{0};
    std::atomic<uint64_t> divergent_events_{0};

    // Timing
    std::chrono::system_clock::time_point start_time_;

    // Thread safety
    mutable std::mutex mutex_;

    // Core functions
    bool should_log_event(const protobuf::NetworkSecurityEvent& event) const;
    nlohmann::json build_json_record(const protobuf::NetworkSecurityEvent& event,
                                     const MLContext& context) const;
    bool write_jsonl(const nlohmann::json& record);
    void save_artifacts(const protobuf::NetworkSecurityEvent& event,
                       const nlohmann::json& json_record);

    // FIX: New functions that assume mutex is already held
    void check_rotation_locked();
    void rotate_logs_locked();

    // Utilities
    void ensure_directories() const;
    std::string get_date_string() const;
    std::string get_current_log_path() const;
    std::string get_iso8601_timestamp() const;

    // Static utilities
    static std::string calculate_sha256(const std::string& data);

    std::shared_ptr<crypto::CryptoManager> crypto_manager_;
};

// ============================================================================
// Factory Function
// ============================================================================

    std::unique_ptr<RAGLogger> create_rag_logger_from_config(
        const std::string& config_path,
        std::shared_ptr<spdlog::logger> logger,
        std::shared_ptr<crypto::CryptoManager> crypto_manager);  // NUEVO

} // namespace ml_defender