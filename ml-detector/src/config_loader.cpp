#include "config_loader.hpp"
#include <fstream>
#include <stdexcept>

namespace ml_detector {

ConfigLoader::ConfigLoader(const std::string& config_path)
    : config_path_(config_path) {
}

DetectorConfig ConfigLoader::load() {
    // Por ahora, config hardcoded
    // TODO: Implementar lectura de YAML con nlohmann::json
    
    DetectorConfig config;
    
    // Defaults
    config.zmq_endpoint = "tcp://localhost:5556";
    config.zmq_bind_mode = true;
    config.num_threads = 4;
    config.batch_size = 32;
    config.inference_timeout_ms = 100;
    
    // Modelos ONNX
    config.level1_model_path = "/vagrant/ml-detector/models/production/level1/level1_attack_detector.onnx";
    config.level2_ddos_model_path = "";
    config.level2_ransomware_model_path = "";
    
    // Logging
    config.log_level = "info";
    config.log_file = "/tmp/ml-detector.log";
    
    return config;
}

} // namespace ml_detector
