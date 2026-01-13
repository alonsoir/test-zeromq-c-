// event_loader.hpp
// RAG Ingester - EventLoader Component
// Day 36: Decrypt + Decompress + Parse .pb event files
// Via Appia Quality - Robust event processing pipeline

#ifndef EVENT_LOADER_HPP
#define EVENT_LOADER_HPP

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace rag_ingester {


    // 🎯 ADR-002: Engine verdict structure
    struct EngineVerdict {
        std::string engine_name;        // "fast-path-sniffer", "random-forest", etc.
        std::string classification;     // "Benign", "Attack", etc.
        float confidence;               // 0.0 - 1.0
        std::string reason_code;        // "SIG_MATCH", "STAT_ANOMALY", etc.
        uint64_t timestamp_ns;          // When this engine decided
    };
    /**
     * @brief Structured event representation (after parsing)
     *
     * Contains all information extracted from protobuf NetworkEvent.
     * Updated with ADR-002
     */
    struct Event {
        // Metadata
        std::string event_id;           // Unique event identifier
        uint64_t timestamp_ns;          // Nanosecond timestamp

        // Classification (from ml-detector)
        std::string final_class;        // e.g., "ransomware", "ddos", "benign"
        float confidence;               // 0.0 - 1.0

        // Feature vector (83-dimensional or 11-dimensional)
        std::vector<float> features;    // Raw features from detector

        // Source information
        std::string source_detector;    // e.g., "ml-detector-default"
        std::string filepath;           // Original .pb file path

        // Flag for debugging
        bool is_partial;                // true if only 11/83 features present

        // 🎯 ADR-002: Multi-Engine Provenance (Day 37)
        std::vector<EngineVerdict> verdicts;  // All engine opinions
        float discrepancy_score;              // 0.0 (agree) - 1.0 (disagree)
        std::string final_decision;           // "ALLOW", "DROP", "ALERT"
    };

/**
 * @brief EventLoader - Decrypt, decompress, and parse .pb event files
 * 
 * Pipeline stages:
 * 1. Read encrypted .pb file from disk
 * 2. Decrypt using AES-256-GCM (crypto-transport)
 * 3. Decompress using gzip (crypto-transport)
 * 4. Parse protobuf (network_security.proto)
 * 5. Extract features into Event struct
 * 
 * Design Constraints:
 * - Zero-copy where possible (minimal allocations)
 * - Exception-safe (RAII for resources)
 * - Memory target: <10MB per event batch
 * - Supports both 83-feature and 11-feature events
 */
class EventLoader {
public:
    /**
     * @brief Construct EventLoader with crypto configuration
     * @param encryption_key_path Path to AES-256 key file
     * @throws std::runtime_error if key file not found or invalid
     */
    explicit EventLoader(const std::string& encryption_key_path);
    
    /**
     * @brief Destructor - cleanup crypto resources
     */
    ~EventLoader();
    
    // Non-copyable (holds crypto state)
    EventLoader(const EventLoader&) = delete;
    EventLoader& operator=(const EventLoader&) = delete;
    
    /**
     * @brief Load and process a single .pb event file
     * @param filepath Path to encrypted .pb file
     * @return Parsed Event structure
     * @throws std::runtime_error on decryption/parsing failure
     */
    Event load(const std::string& filepath);
    
    /**
     * @brief Load multiple .pb files in batch
     * @param filepaths Vector of .pb file paths
     * @return Vector of parsed Events
     * @throws std::runtime_error on any file failure
     */
    std::vector<Event> load_batch(const std::vector<std::string>& filepaths);
    
    /**
     * @brief Get statistics about loaded events
     */
    struct LoadStats {
        uint64_t total_loaded;
        uint64_t total_failed;
        uint64_t bytes_processed;
        uint64_t partial_feature_count;  // Events with only 11 features
    };
    
    LoadStats get_stats() const noexcept;

private:
    // Crypto components (forward-declared to avoid header pollution)
    class CryptoImpl;
    std::unique_ptr<CryptoImpl> crypto_;
    
    // Statistics
    LoadStats stats_;
    
    /**
     * @brief Read entire file into memory
     * @param path File path
     * @return Raw bytes
     * @throws std::runtime_error on I/O failure
     */
    std::vector<uint8_t> read_file(const std::string& path);
    
    /**
     * @brief Decrypt data using crypto-transport
     * @param encrypted Encrypted bytes
     * @return Decrypted bytes
     * @throws std::runtime_error on decryption failure
     */
    std::vector<uint8_t> decrypt(const std::vector<uint8_t>& encrypted);
    
    /**
     * @brief Decompress data using gzip
     * @param compressed Compressed bytes
     * @return Decompressed bytes
     * @throws std::runtime_error on decompression failure
     */
    std::vector<uint8_t> decompress(const std::vector<uint8_t>& compressed);
    
    /**
     * @brief Parse protobuf NetworkEvent message
     * @param data Protobuf-encoded bytes
     * @return Parsed Event
     * @throws std::runtime_error on parse failure
     */
    Event parse_protobuf(const std::vector<uint8_t>& data);
    
    /**
     * @brief Extract 83-dimensional feature vector from protobuf
     * @param proto_event Parsed protobuf message
     * @return Feature vector (may be 11-dim if partial)
     */
    std::vector<float> extract_features(const void* proto_event);
};

} // namespace rag_ingester

#endif // EVENT_LOADER_HPP
