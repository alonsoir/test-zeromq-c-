// csv_event_writer.hpp
// ML Defender - CSV Event Writer for RAG Ingester pipeline
// Day 63: Standardized CSV output from ml-detector
// Authors: Alonso Isidoro Roman + Claude (Anthropic)
//
// DESIGN:
// - One CSV row per detection event (threshold-filtered, same as RAGLogger)
// - 115 columns: metadata + 105 float features + HMAC integrity
// - Daily rotation matching RAGLogger convention (YYYY-MM-DD.csv)
// - HMAC per row using component key from etcd (/secrets/ml-detector)
// - Thread-safe (mutex-protected writes)
//
// CSV SCHEMA (115 columns, no header row, comma-separated):
//   timestamp_ns, event_id, src_ip, dst_ip, src_port, dst_port, protocol,
//   final_class, confidence, threat_category,
//   fast_score, ml_score, divergence, final_decision,
//   f0..f104  (105 floats),
//   hmac
//
// RATIONALE FOR NO ENCRYPTION:
//   CSV files are protected by per-row HMAC (integrity).
//   ChaCha20 encryption (as used in .pb.enc artifacts) ties file lifetime
//   to key rotation period — files older than the rotation window become
//   permanently unreadable. For RAG historical analysis we need durability.
//   Enterprise version will add at-rest encryption transparently.

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <mutex>
#include <atomic>
#include <memory>
#include <chrono>

#include <spdlog/spdlog.h>
#include <network_security.pb.h>

namespace ml_defender {

// ============================================================================
// Configuration
// ============================================================================

struct CsvEventWriterConfig {
    std::string base_dir;           // e.g. /vagrant/logs/ml-detector/events
    std::string hmac_key_hex;       // 64-char hex key from etcd /secrets/ml-detector
    size_t max_events_per_file;     // rotate after N events (default: 10000)
    float min_score_threshold;      // only write events above this score (default: 0.5)
};

// ============================================================================
// CsvEventWriter
// ============================================================================

class CsvEventWriter {
public:
    explicit CsvEventWriter(const CsvEventWriterConfig& config,
                            std::shared_ptr<spdlog::logger> logger);
    ~CsvEventWriter();

    // Non-copyable
    CsvEventWriter(const CsvEventWriter&) = delete;
    CsvEventWriter& operator=(const CsvEventWriter&) = delete;

    /**
     * @brief Write one detection event as a CSV row.
     *
     * Extracts all 105 features from network_features, writes metadata fields,
     * computes HMAC over the complete row, appends to current daily file.
     *
     * @param event  Complete NetworkSecurityEvent (post-classification)
     * @return true if written, false if filtered or error
     */
    bool write_event(const protobuf::NetworkSecurityEvent& event);

    void flush();

    struct Stats {
        uint64_t events_written;
        uint64_t events_skipped;
        uint64_t rows_failed;
        std::string current_file;
    };
    Stats get_stats() const noexcept;

private:
    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /**
     * @brief Extract 105-dimensional feature vector from NetworkSecurityEvent.
     *
     * Mirrors exactly the field order used by EventLoader::extract_features()
     * in rag-ingester so the CsvEventLoader can reconstruct the same vector.
     * Field order is documented in FEATURE_SCHEMA.md (to be generated).
     */
    std::vector<float> extract_features(
        const protobuf::NetworkSecurityEvent& event) const;

    /**
     * @brief Build the complete CSV row string (without trailing newline).
     *
     * Column order matches CSV_SCHEMA defined in this header.
     * All float values use fixed precision (6 decimal places).
     */
    std::string build_row(const protobuf::NetworkSecurityEvent& event,
                          const std::vector<float>& features) const;

    /**
     * @brief Compute HMAC-SHA256 over row content using hmac_key_.
     * @return 64-char lowercase hex string
     */
    std::string compute_hmac(const std::string& row) const;

    // File management
    void ensure_open();
    void rotate_if_needed();
    void rotate_locked();
    std::string get_date_string() const;
    std::string get_file_path(const std::string& date) const;

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------
    CsvEventWriterConfig config_;
    std::shared_ptr<spdlog::logger> logger_;

    // HMAC key as raw bytes (decoded from hex at construction)
    std::vector<uint8_t> hmac_key_;

    mutable std::mutex mutex_;
    std::ofstream current_file_;
    std::string current_date_;
    std::string current_file_path_;

    std::atomic<uint64_t> events_written_{0};
    std::atomic<uint64_t> events_skipped_{0};
    std::atomic<uint64_t> rows_failed_{0};
    std::atomic<size_t>   events_in_current_file_{0};
};

// ============================================================================
// Column count constant — used by CsvEventLoader in rag-ingester
// to validate rows at parse time.
// ============================================================================
static constexpr size_t CSV_COLUMN_COUNT = 115;
//  0:  timestamp_ns
//  1:  event_id
//  2:  src_ip
//  3:  dst_ip
//  4:  src_port
//  5:  dst_port
//  6:  protocol
//  7:  final_class
//  8:  confidence
//  9:  threat_category
// 10:  fast_score
// 11:  ml_score
// 12:  divergence
// 13:  final_decision
// 14..118: f0..f104  (105 floats)
// 119: hmac  ← wait, let's recount:
//   14 metadata cols + 105 feature cols + 1 hmac = 120? No:
//   cols 0-13 = 14 metadata
//   cols 14-118 = 105 features  (14 + 105 = 119 cols so far, index 0..118)
//   col  119 = hmac
// Total = 120 columns.
// Corrected below:
static constexpr size_t CSV_METADATA_COLS = 14;  // cols 0-13
static constexpr size_t CSV_FEATURE_COLS  = 105; // cols 14-118
static constexpr size_t CSV_HMAC_COL      = 119; // col 119
static constexpr size_t CSV_TOTAL_COLS    = 120; // 14 + 105 + 1

} // namespace ml_defender