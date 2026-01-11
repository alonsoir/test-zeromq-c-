#include "common/config_parser.hpp"
#include <fstream>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace rag_ingester {

Config ConfigParser::load(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }
    
    nlohmann::json j;
    file >> j;
    
    Config config;
    from_json(j, config);
    
    validate(config);
    
    spdlog::info("Configuration loaded from: {}", config_path);
    spdlog::info("  Service ID: {}", config.service.id);
    spdlog::info("  Location: {}", config.service.location);
    spdlog::info("  Threading mode: {}", config.ingester.threading.mode);
    
    return config;
}

void ConfigParser::from_json(const nlohmann::json& j, Config& config) {
    // Service
    config.service.id = j["service"]["id"];
    config.service.location = j["service"]["location"];
    config.service.version = j["service"]["version"];
    
    // Etcd
    config.service.etcd.endpoints = j["service"]["etcd"]["endpoints"].get<std::vector<std::string>>();
    config.service.etcd.heartbeat_interval_sec = j["service"]["etcd"]["heartbeat_interval_sec"];
    config.service.etcd.partner_detector = j["service"]["etcd"]["partner_detector"];
    
    // Input
    config.ingester.input.source = j["ingester"]["input"]["source"];
    config.ingester.input.directory = j["ingester"]["input"]["directory"];
    config.ingester.input.pattern = j["ingester"]["input"]["pattern"];
    config.ingester.input.encrypted = j["ingester"]["input"]["encrypted"];
    config.ingester.input.compressed = j["ingester"]["input"]["compressed"];
    config.ingester.input.delete_after_process = j["ingester"]["input"]["delete_after_process"];
    
    // Threading
    config.ingester.threading.mode = j["ingester"]["threading"]["mode"];
    config.ingester.threading.embedding_workers = j["ingester"]["threading"]["embedding_workers"];
    config.ingester.threading.indexing_workers = j["ingester"]["threading"]["indexing_workers"];
    
    // Embedders - Chronos
    config.ingester.embedders.chronos.enabled = j["ingester"]["embedders"]["chronos"]["enabled"];
    config.ingester.embedders.chronos.onnx_path = j["ingester"]["embedders"]["chronos"]["onnx_path"];
    config.ingester.embedders.chronos.input_dim = j["ingester"]["embedders"]["chronos"]["input_dim"];
    config.ingester.embedders.chronos.output_dim = j["ingester"]["embedders"]["chronos"]["output_dim"];
    
    // Embedders - SBERT
    config.ingester.embedders.sbert.enabled = j["ingester"]["embedders"]["sbert"]["enabled"];
    config.ingester.embedders.sbert.onnx_path = j["ingester"]["embedders"]["sbert"]["onnx_path"];
    config.ingester.embedders.sbert.input_dim = j["ingester"]["embedders"]["sbert"]["input_dim"];
    config.ingester.embedders.sbert.output_dim = j["ingester"]["embedders"]["sbert"]["output_dim"];
    
    // Embedders - Attack
    config.ingester.embedders.attack.enabled = j["ingester"]["embedders"]["attack"]["enabled"];
    config.ingester.embedders.attack.onnx_path = j["ingester"]["embedders"]["attack"]["onnx_path"];
    config.ingester.embedders.attack.input_dim = j["ingester"]["embedders"]["attack"]["input_dim"];
    config.ingester.embedders.attack.output_dim = j["ingester"]["embedders"]["attack"]["output_dim"];
    config.ingester.embedders.attack.benign_sample_rate = j["ingester"]["embedders"]["attack"]["benign_sample_rate"];
    
    // PCA
    config.ingester.pca.enabled = j["ingester"]["pca"]["enabled"];
    config.ingester.pca.chronos_model = j["ingester"]["pca"]["chronos_model"];
    config.ingester.pca.sbert_model = j["ingester"]["pca"]["sbert_model"];
    config.ingester.pca.attack_model = j["ingester"]["pca"]["attack_model"];
    
    // FAISS
    config.ingester.faiss.index_type = j["ingester"]["faiss"]["index_type"];
    config.ingester.faiss.metric = j["ingester"]["faiss"]["metric"];
    config.ingester.faiss.persist_path = j["ingester"]["faiss"]["persist_path"];
    config.ingester.faiss.checkpoint_interval_events = j["ingester"]["faiss"]["checkpoint_interval_events"];
    
    // Health
    config.ingester.health.cv_warning_threshold = j["ingester"]["health"]["cv_warning_threshold"];
    config.ingester.health.cv_critical_threshold = j["ingester"]["health"]["cv_critical_threshold"];
    config.ingester.health.report_to_etcd = j["ingester"]["health"]["report_to_etcd"];
}

void ConfigParser::validate(const Config& config) {
    // Validate threading mode
    if (config.ingester.threading.mode != "single" && 
        config.ingester.threading.mode != "parallel") {
        throw std::runtime_error("Invalid threading mode: " + config.ingester.threading.mode);
    }
    
    // Validate worker counts
    if (config.ingester.threading.embedding_workers < 1 || 
        config.ingester.threading.embedding_workers > 16) {
        throw std::runtime_error("Invalid embedding_workers count");
    }
    
    if (config.ingester.threading.indexing_workers < 1 || 
        config.ingester.threading.indexing_workers > 16) {
        throw std::runtime_error("Invalid indexing_workers count");
    }
    
    // Validate FAISS index type
    if (config.ingester.faiss.index_type != "Flat" && 
        config.ingester.faiss.index_type != "IVF") {
        throw std::runtime_error("Invalid FAISS index type: " + config.ingester.faiss.index_type);
    }
    
    // Validate CV thresholds
    if (config.ingester.health.cv_critical_threshold >= config.ingester.health.cv_warning_threshold) {
        throw std::runtime_error("CV critical threshold must be < warning threshold");
    }
    
    spdlog::info("✅ Configuration validation passed");
}

} // namespace rag_ingester
