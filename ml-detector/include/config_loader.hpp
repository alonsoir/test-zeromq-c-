#pragma once

#include <string>

namespace ml_detector {

/**
 * @brief Configuración del ML Detector
 */
struct DetectorConfig {
    // ZeroMQ
    std::string zmq_endpoint;
    bool zmq_bind_mode;
    
    // Threading
    int num_threads;
    
    // Inferencia
    int batch_size;
    int inference_timeout_ms;
    
    // Modelos ONNX
    std::string level1_model_path;
    std::string level2_ddos_model_path;
    std::string level2_ransomware_model_path;
    
    // Logging
    std::string log_level;
    std::string log_file;
};

/**
 * @brief Cargador de configuración
 */
class ConfigLoader {
public:
    explicit ConfigLoader(const std::string& config_path);
    
    DetectorConfig load();
    
private:
    std::string config_path_;
};

} // namespace ml_detector
