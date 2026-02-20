// csv_event_writer.cpp
// ML Defender - CSV Event Writer Implementation
// Day 63
// Authors: Alonso Isidoro Roman + Claude (Anthropic)

#include "csv_event_writer.hpp"

#include <filesystem>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cstring>

#include <openssl/hmac.h>
#include <openssl/evp.h>

namespace fs = std::filesystem;
namespace ml_defender {

// ============================================================================
// Construction / Destruction
// ============================================================================

CsvEventWriter::CsvEventWriter(const CsvEventWriterConfig& config,
                               std::shared_ptr<spdlog::logger> logger)
    : config_(config)
    , logger_(logger)
{
    // Validate config
    if (config_.base_dir.empty()) {
        throw std::invalid_argument("[CsvEventWriter] base_dir cannot be empty");
    }
    if (config_.hmac_key_hex.size() != 64) {
        throw std::invalid_argument(
            "[CsvEventWriter] hmac_key_hex must be 64 hex chars (32 bytes), got: "
            + std::to_string(config_.hmac_key_hex.size()));
    }
    if (config_.max_events_per_file == 0) {
        config_.max_events_per_file = 10000;
    }
    if (config_.min_score_threshold <= 0.0f) {
        config_.min_score_threshold = 0.5f;
    }

    // Decode HMAC key from hex
    hmac_key_.reserve(32);
    for (size_t i = 0; i < 64; i += 2) {
        uint8_t byte = static_cast<uint8_t>(
            std::stoul(config_.hmac_key_hex.substr(i, 2), nullptr, 16));
        hmac_key_.push_back(byte);
    }

    // Ensure output directory exists
    fs::create_directories(config_.base_dir);

    // Open today's file
    ensure_open();

    logger_->info("[CsvEventWriter] Initialized");
    logger_->info("  Output dir:    {}", config_.base_dir);
    logger_->info("  HMAC key:      {}...{} (32 bytes)",
        config_.hmac_key_hex.substr(0, 8),
        config_.hmac_key_hex.substr(56, 8));
    logger_->info("  Max events/file: {}", config_.max_events_per_file);
    logger_->info("  Min score:     {:.2f}", config_.min_score_threshold);
    logger_->info("  CSV columns:   {}", CSV_TOTAL_COLS);
    logger_->info("  Current file:  {}", current_file_path_);
}

CsvEventWriter::~CsvEventWriter() {
    flush();
    logger_->info("[CsvEventWriter] Shutdown — written={} skipped={} failed={}",
        events_written_.load(),
        events_skipped_.load(),
        rows_failed_.load());
}

// ============================================================================
// Public Interface
// ============================================================================

bool CsvEventWriter::write_event(const protobuf::NetworkSecurityEvent& event) {
    // ---- Filter ----
    if (event.overall_threat_score() < config_.min_score_threshold) {
        events_skipped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    if (!event.has_network_features()) {
        logger_->warn("[CsvEventWriter] Skipping event {} — missing network_features",
                      event.event_id());
        events_skipped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    try {
        // ---- Extract features ----
        auto features = extract_features(event);

        if (features.size() != CSV_FEATURE_COLS) {
            logger_->error("[CsvEventWriter] Feature count mismatch: got {} expected {}",
                          features.size(), CSV_FEATURE_COLS);
            rows_failed_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        // ---- Build row (without HMAC) ----
        std::string row = build_row(event, features);

        // ---- Compute HMAC over row content ----
        std::string hmac = compute_hmac(row);

        // ---- Final row = content + comma + hmac ----
        std::string full_row = row + "," + hmac;

        // ---- Write (thread-safe) ----
        {
            std::lock_guard<std::mutex> lock(mutex_);
            rotate_if_needed();
            current_file_ << full_row << "\n";
            current_file_.flush();
            events_in_current_file_.fetch_add(1, std::memory_order_relaxed);
        }

        events_written_.fetch_add(1, std::memory_order_relaxed);

        logger_->debug("[CsvEventWriter] Written event_id={} class={} score={:.4f}",
            event.event_id(),
            event.final_classification(),
            event.overall_threat_score());

        return true;

    } catch (const std::exception& e) {
        rows_failed_.fetch_add(1, std::memory_order_relaxed);
        logger_->error("[CsvEventWriter] Failed to write event {}: {}",
                       event.event_id(), e.what());
        return false;
    }
}

void CsvEventWriter::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (current_file_.is_open()) {
        current_file_.flush();
    }
}

CsvEventWriter::Stats CsvEventWriter::get_stats() const noexcept {
    return {
        events_written_.load(),
        events_skipped_.load(),
        rows_failed_.load(),
        current_file_path_
    };
}

// ============================================================================
// Feature Extraction
// Mirrors EventLoader::extract_features() in rag-ingester exactly.
// Field order is the contract between ml-detector and rag-ingester.
// DO NOT reorder without updating CsvEventLoader in rag-ingester.
// ============================================================================

std::vector<float> CsvEventWriter::extract_features(
    const protobuf::NetworkSecurityEvent& event) const
{
    const auto& net = event.network_features();
    std::vector<float> f;
    f.reserve(105);

    // ---- Part 1: Basic flow statistics (65 features) ----

    // Identification (4)
    f.push_back(static_cast<float>(net.protocol_number()));
    f.push_back(static_cast<float>(net.interface_mode()));
    f.push_back(static_cast<float>(net.source_ifindex()));
    f.push_back(static_cast<float>(net.flow_duration_microseconds() / 1e6));

    // Basic packet stats (4)
    f.push_back(static_cast<float>(net.total_forward_packets()));
    f.push_back(static_cast<float>(net.total_backward_packets()));
    f.push_back(static_cast<float>(net.total_forward_bytes()));
    f.push_back(static_cast<float>(net.total_backward_bytes()));

    // Forward packet length (4)
    f.push_back(static_cast<float>(net.forward_packet_length_max()));
    f.push_back(static_cast<float>(net.forward_packet_length_min()));
    f.push_back(static_cast<float>(net.forward_packet_length_mean()));
    f.push_back(static_cast<float>(net.forward_packet_length_std()));

    // Backward packet length (4)
    f.push_back(static_cast<float>(net.backward_packet_length_max()));
    f.push_back(static_cast<float>(net.backward_packet_length_min()));
    f.push_back(static_cast<float>(net.backward_packet_length_mean()));
    f.push_back(static_cast<float>(net.backward_packet_length_std()));

    // Speeds and ratios (8)
    f.push_back(static_cast<float>(net.flow_bytes_per_second()));
    f.push_back(static_cast<float>(net.flow_packets_per_second()));
    f.push_back(static_cast<float>(net.forward_packets_per_second()));
    f.push_back(static_cast<float>(net.backward_packets_per_second()));
    f.push_back(static_cast<float>(net.download_upload_ratio()));
    f.push_back(static_cast<float>(net.average_packet_size()));
    f.push_back(static_cast<float>(net.average_forward_segment_size()));
    f.push_back(static_cast<float>(net.average_backward_segment_size()));

    // Flow IAT (4)
    f.push_back(static_cast<float>(net.flow_inter_arrival_time_mean()));
    f.push_back(static_cast<float>(net.flow_inter_arrival_time_std()));
    f.push_back(static_cast<float>(net.flow_inter_arrival_time_max()));
    f.push_back(static_cast<float>(net.flow_inter_arrival_time_min()));

    // Forward IAT (5)
    f.push_back(static_cast<float>(net.forward_inter_arrival_time_total()));
    f.push_back(static_cast<float>(net.forward_inter_arrival_time_mean()));
    f.push_back(static_cast<float>(net.forward_inter_arrival_time_std()));
    f.push_back(static_cast<float>(net.forward_inter_arrival_time_max()));
    f.push_back(static_cast<float>(net.forward_inter_arrival_time_min()));

    // Backward IAT (5)
    f.push_back(static_cast<float>(net.backward_inter_arrival_time_total()));
    f.push_back(static_cast<float>(net.backward_inter_arrival_time_mean()));
    f.push_back(static_cast<float>(net.backward_inter_arrival_time_std()));
    f.push_back(static_cast<float>(net.backward_inter_arrival_time_max()));
    f.push_back(static_cast<float>(net.backward_inter_arrival_time_min()));

    // TCP flags (8)
    f.push_back(static_cast<float>(net.fin_flag_count()));
    f.push_back(static_cast<float>(net.syn_flag_count()));
    f.push_back(static_cast<float>(net.rst_flag_count()));
    f.push_back(static_cast<float>(net.psh_flag_count()));
    f.push_back(static_cast<float>(net.ack_flag_count()));
    f.push_back(static_cast<float>(net.urg_flag_count()));
    f.push_back(static_cast<float>(net.cwe_flag_count()));
    f.push_back(static_cast<float>(net.ece_flag_count()));

    // Directional flags (4)
    f.push_back(static_cast<float>(net.forward_psh_flags()));
    f.push_back(static_cast<float>(net.backward_psh_flags()));
    f.push_back(static_cast<float>(net.forward_urg_flags()));
    f.push_back(static_cast<float>(net.backward_urg_flags()));

    // Headers and bulk transfer (8)
    f.push_back(static_cast<float>(net.forward_header_length()));
    f.push_back(static_cast<float>(net.backward_header_length()));
    f.push_back(static_cast<float>(net.forward_average_bytes_bulk()));
    f.push_back(static_cast<float>(net.forward_average_packets_bulk()));
    f.push_back(static_cast<float>(net.forward_average_bulk_rate()));
    f.push_back(static_cast<float>(net.backward_average_bytes_bulk()));
    f.push_back(static_cast<float>(net.backward_average_packets_bulk()));
    f.push_back(static_cast<float>(net.backward_average_bulk_rate()));

    // Additional stats (5)
    f.push_back(static_cast<float>(net.minimum_packet_length()));
    f.push_back(static_cast<float>(net.maximum_packet_length()));
    f.push_back(static_cast<float>(net.packet_length_mean()));
    f.push_back(static_cast<float>(net.packet_length_std()));
    f.push_back(static_cast<float>(net.packet_length_variance()));

    // Active/idle (2)
    f.push_back(static_cast<float>(net.active_mean()));
    f.push_back(static_cast<float>(net.idle_mean()));

    // ---- Part 2: Embedded detector features (40 features) ----

    // DDoS embedded (10)
    if (net.has_ddos_embedded()) {
        const auto& d = net.ddos_embedded();
        f.push_back(d.syn_ack_ratio());
        f.push_back(d.packet_symmetry());
        f.push_back(d.source_ip_dispersion());
        f.push_back(d.protocol_anomaly_score());
        f.push_back(d.packet_size_entropy());
        f.push_back(d.traffic_amplification_factor());
        f.push_back(d.flow_completion_rate());
        f.push_back(d.geographical_concentration());
        f.push_back(d.traffic_escalation_rate());
        f.push_back(d.resource_saturation_score());
    } else {
        for (int i = 0; i < 10; i++) f.push_back(0.0f);
    }

    // Ransomware embedded (10)
    if (net.has_ransomware_embedded()) {
        const auto& r = net.ransomware_embedded();
        f.push_back(r.io_intensity());
        f.push_back(r.entropy());
        f.push_back(r.resource_usage());
        f.push_back(r.network_activity());
        f.push_back(r.file_operations());
        f.push_back(r.process_anomaly());
        f.push_back(r.temporal_pattern());
        f.push_back(r.access_frequency());
        f.push_back(r.data_volume());
        f.push_back(r.behavior_consistency());
    } else {
        for (int i = 0; i < 10; i++) f.push_back(0.0f);
    }

    // Traffic classification (10)
    if (net.has_traffic_classification()) {
        const auto& t = net.traffic_classification();
        f.push_back(t.packet_rate());
        f.push_back(t.connection_rate());
        f.push_back(t.tcp_udp_ratio());
        f.push_back(t.avg_packet_size());
        f.push_back(t.port_entropy());
        f.push_back(t.flow_duration_std());
        f.push_back(t.src_ip_entropy());
        f.push_back(t.dst_ip_concentration());
        f.push_back(t.protocol_variety());
        f.push_back(t.temporal_consistency());
    } else {
        for (int i = 0; i < 10; i++) f.push_back(0.0f);
    }

    // Internal anomaly (10)
    if (net.has_internal_anomaly()) {
        const auto& ia = net.internal_anomaly();
        f.push_back(ia.internal_connection_rate());
        f.push_back(ia.service_port_consistency());
        f.push_back(ia.protocol_regularity());
        f.push_back(ia.packet_size_consistency());
        f.push_back(ia.connection_duration_std());
        f.push_back(ia.lateral_movement_score());
        f.push_back(ia.service_discovery_patterns());
        f.push_back(ia.data_exfiltration_indicators());
        f.push_back(ia.temporal_anomaly_score());
        f.push_back(ia.access_pattern_entropy());
    } else {
        for (int i = 0; i < 10; i++) f.push_back(0.0f);
    }

    return f;  // exactly 105 floats
}

// ============================================================================
// Row Builder
// ============================================================================

std::string CsvEventWriter::build_row(
    const protobuf::NetworkSecurityEvent& event,
    const std::vector<float>& features) const
{
    const auto& net = event.network_features();

    // Timestamp in nanoseconds
    uint64_t timestamp_ns = 0;
    if (event.has_event_timestamp()) {
        timestamp_ns = static_cast<uint64_t>(event.event_timestamp().seconds())
                       * 1'000'000'000ULL
                       + static_cast<uint64_t>(event.event_timestamp().nanos());
    }

    // Final decision from provenance or fallback
    std::string final_decision = "UNKNOWN";
    if (event.has_provenance() && !event.provenance().final_decision().empty()) {
        final_decision = event.provenance().final_decision();
    }

    // Score divergence
    float divergence = 0.0f;
    if (event.has_provenance()) {
        divergence = event.provenance().discrepancy_score();
    } else if (event.has_decision_metadata()) {
        divergence = static_cast<float>(event.decision_metadata().score_divergence());
    }

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);

    // Cols 0-13: metadata
    ss << timestamp_ns                          << ","   // 0
       << event.event_id()                      << ","   // 1
       << net.source_ip()                       << ","   // 2
       << net.destination_ip()                  << ","   // 3
       << net.source_port()                     << ","   // 4
       << net.destination_port()                << ","   // 5
       << net.protocol_name()                   << ","   // 6
       << event.final_classification()          << ","   // 7
       << event.overall_threat_score()          << ","   // 8
       << event.threat_category()               << ","   // 9
       << event.fast_detector_score()           << ","   // 10
       << event.ml_detector_score()             << ","   // 11
       << divergence                            << ","   // 12
       << final_decision;                                // 13

    // Cols 14-118: f0..f104
    for (const float val : features) {
        ss << "," << val;
    }

    // col 119 (HMAC) will be appended by write_event() after this returns

    return ss.str();
}

// ============================================================================
// HMAC-SHA256
// ============================================================================

std::string CsvEventWriter::compute_hmac(const std::string& row) const {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len = 0;

    HMAC(EVP_sha256(),
         hmac_key_.data(),
         static_cast<int>(hmac_key_.size()),
         reinterpret_cast<const unsigned char*>(row.data()),
         row.size(),
         digest,
         &digest_len);

    std::ostringstream hex;
    hex << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_len; i++) {
        hex << std::setw(2) << static_cast<int>(digest[i]);
    }
    return hex.str();
}

// ============================================================================
// File Management
// ============================================================================

void CsvEventWriter::ensure_open() {
    current_date_ = get_date_string();
    current_file_path_ = get_file_path(current_date_);

    current_file_.open(current_file_path_, std::ios::app);
    if (!current_file_.is_open()) {
        throw std::runtime_error(
            "[CsvEventWriter] Cannot open file: " + current_file_path_);
    }
}

void CsvEventWriter::rotate_if_needed() {
    // PRECONDITION: mutex_ held by caller

    std::string today = get_date_string();

    bool date_changed = (today != current_date_);
    bool size_exceeded = (events_in_current_file_.load() >= config_.max_events_per_file);

    if (date_changed || size_exceeded) {
        if (date_changed) {
            logger_->info("[CsvEventWriter] Date changed → rotating to {}", today);
        } else {
            logger_->info("[CsvEventWriter] Max events ({}) reached → rotating",
                         config_.max_events_per_file);
        }
        rotate_locked();
    }
}

void CsvEventWriter::rotate_locked() {
    // PRECONDITION: mutex_ held by caller

    if (current_file_.is_open()) {
        current_file_.flush();
        current_file_.close();
    }

    current_date_ = get_date_string();
    current_file_path_ = get_file_path(current_date_);
    events_in_current_file_.store(0);

    current_file_.open(current_file_path_, std::ios::app);
    if (!current_file_.is_open()) {
        logger_->error("[CsvEventWriter] Failed to open rotated file: {}",
                       current_file_path_);
    } else {
        logger_->info("[CsvEventWriter] Rotated to: {}", current_file_path_);
    }
}

std::string CsvEventWriter::get_date_string() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&time_t, &tm);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d");
    return oss.str();
}

std::string CsvEventWriter::get_file_path(const std::string& date) const {
    return config_.base_dir + "/" + date + ".csv";
}

} // namespace ml_defender