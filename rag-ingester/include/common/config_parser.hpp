#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace rag_ingester {

struct EtcdConfig {
    std::vector<std::string> endpoints;
    int heartbeat_interval_sec;
    std::string partner_detector;
};

struct InputConfig {
    std::string source;
    std::string directory;
    std::string pattern;
    bool encrypted;
    bool compressed;
    bool delete_after_process;
};

struct ThreadingConfig {
    std::string mode;  // "single" or "parallel"
    int embedding_workers;
    int indexing_workers;
};

struct EmbedderConfig {
    bool enabled;
    std::string onnx_path;
    int input_dim;
    int output_dim;
    float benign_sample_rate;  // Only for attack embedder
};

struct PCAConfig {
    bool enabled;
    std::string chronos_model;
    std::string sbert_model;
    std::string attack_model;
};

struct FAISSConfig {
    std::string index_type;
    std::string metric;
    std::string persist_path;
    int checkpoint_interval_events;
};

struct HealthConfig {
    float cv_warning_threshold;
    float cv_critical_threshold;
    bool report_to_etcd;
};

struct Config {
    struct {
        std::string id;
        std::string location;
        std::string version;
        EtcdConfig etcd;
    } service;
    
    struct {
        InputConfig input;
        ThreadingConfig threading;
        
        struct {
            EmbedderConfig chronos;
            EmbedderConfig sbert;
            EmbedderConfig attack;
        } embedders;
        
        PCAConfig pca;
        FAISSConfig faiss;
        HealthConfig health;
    } ingester;
};

class ConfigParser {
public:
    static Config load(const std::string& config_path);
    static void validate(const Config& config);
    
private:
    static void from_json(const nlohmann::json& j, Config& config);
};

} // namespace rag_ingester
