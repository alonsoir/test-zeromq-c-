// main.h - Enhanced Sniffer v3.1
// FECHA: 26 de Septiembre de 2025
// DESCRIPCIÓN: Estructuras y declaraciones para strict JSON parsing

#pragma once

#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <memory>
#include <atomic>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <filesystem>
#include <fstream>

// JSON parsing
#include <json/json.h>

// ============================================================================
// MACROS DE VALIDACIÓN ESTRICTA JSON
// ============================================================================

#define REQUIRE_FIELD(json_obj, field_name, field_type) \
    do { \
        if (!json_obj.isMember(field_name)) { \
            throw std::runtime_error("CAMPO REQUERIDO FALTANTE: " + std::string(field_name) + " en sección " + #json_obj); \
        } \
        if (!json_obj[field_name].is##field_type()) { \
            throw std::runtime_error("TIPO INCORRECTO: " + std::string(field_name) + " debe ser " + #field_type + " en sección " + #json_obj); \
        } \
    } while(0)

#define REQUIRE_ARRAY(json_obj, field_name) \
    do { \
        if (!json_obj.isMember(field_name)) { \
            throw std::runtime_error("ARRAY REQUERIDO FALTANTE: " + std::string(field_name) + " en sección " + #json_obj); \
        } \
        if (!json_obj[field_name].isArray()) { \
            throw std::runtime_error("TIPO INCORRECTO: " + std::string(field_name) + " debe ser Array en sección " + #json_obj); \
        } \
    } while(0)

#define REQUIRE_OBJECT(json_obj, field_name) \
    do { \
        if (!json_obj.isMember(field_name)) { \
            throw std::runtime_error("OBJETO REQUERIDO FALTANTE: " + std::string(field_name) + " en sección " + #json_obj); \
        } \
        if (!json_obj[field_name].isObject()) { \
            throw std::runtime_error("TIPO INCORRECTO: " + std::string(field_name) + " debe ser Object en sección " + #json_obj); \
        } \
    } while(0)

// ============================================================================
// ESTRUCTURAS DE CONFIGURACIÓN
// ============================================================================

struct CommandLineArgs {
    bool verbose = false;
    bool help = false;
    std::string config_file = "../config/sniffer.json";
    std::string interface_override = "";
    std::string profile_override = "";
    bool dry_run = false;
    bool show_config_only = false;
};

struct FeatureGroup {
    std::string name;
    int count;
    std::string reference;
    std::string description;
    bool loaded = false;
};

struct StrictSnifferConfig {
    // Componente - TODOS REQUERIDOS
    std::string component_name;
    std::string component_version;
    std::string component_mode;
    std::string kernel_version_required;

    // Identificación - TODOS REQUERIDOS
    std::string node_id;
    std::string cluster_name;
    std::string active_profile;

    // Perfil aplicado - TODOS REQUERIDOS
    std::string capture_interface;
    bool promiscuous_mode;
    bool af_xdp_enabled;
    int worker_threads;
    int compression_level;

    // Captura - TODOS REQUERIDOS
    std::string kernel_interface;
    std::string user_interface;
    std::string capture_mode;
    std::vector<std::string> xdp_flags;
    std::string filter_expression;
    int buffer_size;
    int min_packet_size;
    int max_packet_size;
    std::vector<int> excluded_ports;
    std::vector<std::string> included_protocols;

    // AF_XDP - TODOS REQUERIDOS
    struct {
        bool enabled;
        int queue_id;
        int frame_size;
        int fill_ring_size;
        int comp_ring_size;
        int tx_ring_size;
        int rx_ring_size;
        int umem_size;
    } af_xdp;

    // Buffers - TODOS REQUERIDOS
    struct {
        int ring_buffer_entries;
        int user_processing_queue_depth;
        int protobuf_serialize_buffer_size;
        int zmq_send_buffer_size;
        int flow_state_buffer_entries;
        int statistics_buffer_entries;
        int batch_processing_size;
    } buffers;

    // Threading - TODOS REQUERIDOS
    struct {
        int ring_consumer_threads;
        int feature_processor_threads;
        int zmq_sender_threads;
        int statistics_collector_threads;
        int total_worker_threads;
        bool cpu_affinity_enabled;
        std::vector<int> ring_consumers_affinity;
        std::vector<int> processors_affinity;
        std::vector<int> zmq_senders_affinity;
        std::vector<int> statistics_affinity;
        std::string ring_consumers_priority;
        std::string processors_priority;
        std::string zmq_senders_priority;
    } threading;

    // Kernel space - TODOS REQUERIDOS
    struct {
        std::string ebpf_program;
        std::string xdp_mode;
        int ring_buffer_size;
        int max_flows_in_kernel;
        int flow_timeout_seconds;
        std::vector<std::string> kernel_features;
        int cpu_budget_us_per_packet;
        int max_instructions_per_program;
        int map_update_batch_size;
    } kernel_space;

    // User space - TODOS REQUERIDOS
    struct {
        int flow_table_size;
        int time_window_buffer_size;
        std::vector<std::string> user_features;
        std::string flow_eviction_policy;
        int max_memory_usage_mb;
        int gc_interval_seconds;
    } user_space;

    // Feature groups - TODOS REQUERIDOS
    std::unordered_map<std::string, FeatureGroup> feature_groups;

    // Time windows - TODOS REQUERIDOS
    struct {
        int flow_tracking_window_seconds;
        int statistics_collection_window_seconds;
        int feature_aggregation_window_seconds;
        int cleanup_interval_seconds;
        int max_flows_per_window;
        int window_overlap_seconds;
    } time_windows;

    // Network output - TODOS REQUERIDOS
    struct {
        std::string address;
        int port;
        std::string mode;
        std::string socket_type;
        int high_water_mark;
    } network_output;

    // ZMQ - TODOS REQUERIDOS
    struct {
        int worker_threads;
        int io_thread_pools;
        int push_sockets;
        std::string load_balancing;
        bool failover_enabled;
        int internal_queues;
        int queue_size;
        int queue_timeout_ms;
        std::string overflow_policy;
        int sndhwm;
        int linger_ms;
        int send_timeout_ms;
        int rcvhwm;
        int recv_timeout_ms;
        int tcp_keepalive;
        int sndbuf;
        int rcvbuf;
        int reconnect_interval_ms;
        int max_reconnect_attempts;
        bool batch_enabled;
        int batch_size;
        int batch_timeout_ms;
        int max_batches_queued;
    } zmq;

    // Compresión - TODOS REQUERIDOS
    struct {
        bool enabled;
        std::string algorithm;
        int level;
        int min_compress_size;
        double compression_ratio_threshold;
        bool adaptive_compression;
    } compression;

    // Encriptación - TODOS REQUERIDOS
    struct {
        bool enabled;
        bool etcd_token_required;
        std::string algorithm;
        int key_size;
        int key_rotation_hours;
        std::string fallback_mode;
        bool zmq_curve_enabled;
        std::string curve_server_key;
        std::string curve_secret_key;
    } encryption;

    // etcd - TODOS REQUERIDOS
    struct {
        bool enabled;
        std::vector<std::string> endpoints;
        int connection_timeout_ms;
        int retry_attempts;
        int retry_interval_ms;
        std::string crypto_token_path;
        std::string config_sync_path;
        bool required_for_encryption;
        std::string fallback_mode;
        int heartbeat_interval_seconds;
        int lease_ttl_seconds;
    } etcd;

    // Processing - TODOS REQUERIDOS
    struct {
        struct {
            int max_map_entries;
            int flow_hash_table_size;
            int counter_map_size;
            int stats_collection_interval_ms;
        } kernel_processing;

        struct {
            int internal_queue_size;
            int queue_timeout_seconds;
            int batch_size;
            bool lockfree_queues;
            bool numa_aware_allocation;
            bool memory_prefaulting;
        } user_processing;

        struct {
            bool enabled;
            bool extract_all_features;
            bool cache_flow_states;
            int flow_timeout_seconds;
            int max_concurrent_flows;
            std::string statistics_engine;
            bool simd_optimization;
        } feature_extraction;
    } processing;

    // Auto tuner - TODOS REQUERIDOS
    struct {
        bool enabled;
        bool calibration_on_startup;
        int benchmark_iterations;
        int recalibration_interval_hours;
        double safety_margin_factor;
        std::string feature_placement_strategy;
    } auto_tuner;

    // Monitoring - TODOS REQUERIDOS
    struct {
        int stats_interval_seconds;
        bool performance_tracking;
        bool ebpf_program_stats;
        bool map_utilization;
        bool instruction_count;
        bool processing_time_per_packet;
        bool feature_extraction_rate;
        bool queue_depth;
        bool memory_usage;
        bool thread_utilization;
        bool compression_ratio;
        bool zmq_throughput;
        double max_drop_rate_percent;
        double max_queue_usage_percent;
        int max_memory_usage_mb;
        double max_cpu_usage_percent;
        int max_processing_latency_us;
        double min_compression_ratio;
    } monitoring;

    // Protobuf - TODOS REQUERIDOS
    struct {
        std::string schema_version;
        bool validate_before_send;
        int max_event_size_bytes;
        int serialization_timeout_ms;
        int pool_size;
        bool reuse_objects;
    } protobuf;

    // Logging - TODOS REQUERIDOS
    struct {
        std::string level;
        std::string file;
        int max_file_size_mb;
        int backup_count;
        bool include_performance_logs;
        bool include_compression_stats;
        bool include_thread_stats;
    } logging;

    // Security - TODOS REQUERIDOS
    struct {
        bool input_validation_enabled;
        int max_packet_size_bytes;
        bool validate_ip_addresses;
        bool validate_ports;
        bool sanitize_inputs;
    } security;

    // Backpressure - TODOS REQUERIDOS
    struct {
        bool enabled;
        std::string ring_buffer_full_action;
        int max_drops_per_second;
        bool adaptive_rate_limiting;
        bool circuit_breaker_enabled;
        int failure_threshold;
        int recovery_timeout;
    } backpressure;
};

struct DetailedStats {
    // Estadísticas básicas
    std::atomic<uint64_t> packets_captured{0};
    std::atomic<uint64_t> packets_processed{0};
    std::atomic<uint64_t> packets_sent{0};
    std::atomic<uint64_t> bytes_captured{0};
    std::atomic<uint64_t> bytes_compressed{0};
    std::atomic<uint64_t> errors{0};
    std::atomic<uint64_t> drops{0};

    // Estadísticas kernel space
    std::atomic<uint64_t> kernel_packets_processed{0};
    std::atomic<uint64_t> kernel_map_updates{0};
    std::atomic<uint64_t> kernel_instructions_executed{0};

    // Estadísticas user space
    std::atomic<uint64_t> user_flows_tracked{0};
    std::atomic<uint64_t> user_features_extracted{0};
    std::atomic<uint64_t> user_memory_usage_mb{0};

    // Estadísticas ZMQ
    std::atomic<uint64_t> zmq_messages_sent{0};
    std::atomic<uint64_t> zmq_send_errors{0};
    std::atomic<uint64_t> zmq_reconnections{0};

    // Estadísticas compresión
    std::atomic<uint64_t> compression_operations{0};
    std::atomic<uint64_t> compression_savings_bytes{0};

    // Estadísticas etcd
    std::atomic<uint64_t> etcd_token_requests{0};
    std::atomic<uint64_t> etcd_connection_errors{0};

    std::chrono::steady_clock::time_point start_time{std::chrono::steady_clock::now()};

    // Métodos de incremento
    void incrementPacketsCaptured() { packets_captured++; }
    void incrementPacketsProcessed() { packets_processed++; }
    void incrementPacketsSent() { packets_sent++; }
    void addBytesCaptured(uint64_t bytes) { bytes_captured += bytes; }
    void addBytesCompressed(uint64_t bytes) { bytes_compressed += bytes; }
    void incrementErrors() { errors++; }
    void incrementDrops() { drops++; }
    void incrementKernelPacketsProcessed() { kernel_packets_processed++; }
    void incrementKernelMapUpdates() { kernel_map_updates++; }
    void addKernelInstructions(uint64_t count) { kernel_instructions_executed += count; }
    void setUserFlowsTracked(uint64_t count) { user_flows_tracked = count; }
    void incrementUserFeaturesExtracted() { user_features_extracted++; }
    void setUserMemoryUsage(uint64_t mb) { user_memory_usage_mb = mb; }
    void incrementZMQMessagesSent() { zmq_messages_sent++; }
    void incrementZMQSendErrors() { zmq_send_errors++; }
    void incrementZMQReconnections() { zmq_reconnections++; }
    void incrementCompressionOperations() { compression_operations++; }
    void addCompressionSavings(uint64_t bytes) { compression_savings_bytes += bytes; }
    void incrementEtcdTokenRequests() { etcd_token_requests++; }
    void incrementEtcdConnectionErrors() { etcd_connection_errors++; }

    // Getters
    uint64_t getPacketsCaptured() const { return packets_captured.load(); }
    uint64_t getPacketsProcessed() const { return packets_processed.load(); }
    uint64_t getPacketsSent() const { return packets_sent.load(); }
    uint64_t getBytesCaptured() const { return bytes_captured.load(); }
    uint64_t getBytesCompressed() const { return bytes_compressed.load(); }
    uint64_t getErrors() const { return errors.load(); }
    uint64_t getDrops() const { return drops.load(); }
    uint64_t getKernelPacketsProcessed() const { return kernel_packets_processed.load(); }
    uint64_t getKernelMapUpdates() const { return kernel_map_updates.load(); }
    uint64_t getKernelInstructionsExecuted() const { return kernel_instructions_executed.load(); }
    uint64_t getUserFlowsTracked() const { return user_flows_tracked.load(); }
    uint64_t getUserFeaturesExtracted() const { return user_features_extracted.load(); }
    uint64_t getUserMemoryUsage() const { return user_memory_usage_mb.load(); }
    uint64_t getZMQMessagesSent() const { return zmq_messages_sent.load(); }
    uint64_t getZMQSendErrors() const { return zmq_send_errors.load(); }
    uint64_t getZMQReconnections() const { return zmq_reconnections.load(); }
    uint64_t getCompressionOperations() const { return compression_operations.load(); }
    uint64_t getCompressionSavings() const { return compression_savings_bytes.load(); }
    uint64_t getEtcdTokenRequests() const { return etcd_token_requests.load(); }
    uint64_t getEtcdConnectionErrors() const { return etcd_connection_errors.load(); }

    uint64_t getUptime() const {
        return std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time).count();
    }

    void reset();
};

struct PacketInfo {
    std::string src_ip;
    std::string dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t protocol;
    size_t packet_size;
    std::chrono::high_resolution_clock::time_point timestamp;
    std::vector<double> extracted_features;
};

// ============================================================================
// DECLARACIONES DE FUNCIONES
// ============================================================================

// Signal handling
void signal_handler(int signum);

// Command line parsing
void parse_command_line(int argc, char* argv[], CommandLineArgs& args);
void print_help(const char* program_name);

// JSON configuration loading - STRICT MODE
bool strict_load_json_config(const std::string& config_path, StrictSnifferConfig& config, bool verbose);
void validate_component_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_profiles_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_capture_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_buffers_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_threading_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_kernel_space_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_user_space_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_feature_groups_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_time_windows_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_network_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_zmq_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_transport_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_etcd_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_processing_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_auto_tuner_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_monitoring_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_protobuf_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_logging_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_security_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);
void validate_backpressure_section(const Json::Value& root, StrictSnifferConfig& config, bool verbose);

// Configuration display
void print_complete_config(const StrictSnifferConfig& config, bool verbose);
void print_config_summary(const StrictSnifferConfig& config);

// etcd integration
bool initialize_etcd_connection(const StrictSnifferConfig& config, bool verbose);
std::string get_encryption_token_from_etcd(bool verbose);
void etcd_heartbeat_thread(const StrictSnifferConfig& config, DetailedStats& stats);

// Subsystem initialization
bool initialize_compression(const StrictSnifferConfig& config, bool verbose);
bool initialize_zmq_pool(const StrictSnifferConfig& config, bool verbose);
bool initialize_protobuf(const StrictSnifferConfig& config, bool verbose);

// Packet capture and processing
int create_advanced_raw_socket(const std::string& interface, const StrictSnifferConfig& config, bool verbose);
bool parse_advanced_packet(const uint8_t* packet_data, size_t packet_len, PacketInfo& info, const StrictSnifferConfig& config);
void extract_ml_features(PacketInfo& info, const StrictSnifferConfig& config);
void process_and_send_advanced_packet(const PacketInfo& packet_info, const StrictSnifferConfig& config, DetailedStats& stats);

// Threading functions
void packet_capture_thread(const StrictSnifferConfig& config, DetailedStats& stats);
void kernel_space_monitor_thread(const StrictSnifferConfig& config, DetailedStats& stats);
void user_space_processor_thread(const StrictSnifferConfig& config, DetailedStats& stats);
void zmq_sender_thread(const StrictSnifferConfig& config, DetailedStats& stats);
void detailed_stats_display_thread(const StrictSnifferConfig& config, DetailedStats& stats);

// Monitoring and alerts
void check_and_report_alerts(const StrictSnifferConfig& config, const DetailedStats& stats);
void log_performance_metrics(const StrictSnifferConfig& config, const DetailedStats& stats);

// Cleanup
void cleanup_subsystems();
void print_final_statistics(const StrictSnifferConfig& config, const DetailedStats& stats);

// ============================================================================
// DECLARACIONES EXTERN DE VARIABLES GLOBALES
// ============================================================================
extern std::atomic<bool> g_running;
extern StrictSnifferConfig g_config;
extern DetailedStats g_stats;
extern CommandLineArgs g_args;