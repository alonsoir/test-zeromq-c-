#pragma once

#include <memory>
#include <atomic>
#include <thread>
#include <zmq.hpp>
#include <spdlog/spdlog.h>
#include "config_loader.hpp"
#include "onnx_model.hpp"
#include "feature_extractor.hpp"
#include "network_security.pb.h"

namespace ml_detector {

/**
 * @brief ZMQ Handler - Core del pipeline ML Detector
 * 
 * Flujo:
 * 1. PULL events del sniffer (protobuf serializado)
 * 2. Deserializar → NetworkSecurityEvent
 * 3. Extraer features (FeatureExtractor)
 * 4. Inference (ONNXModel)
 * 5. Enriquecer evento con ML predictions
 * 6. PUB evento enriquecido al analyzer
 * 
 * Sockets:
 * - Input:  PULL @ ipc:///tmp/sniffer.ipc
 * - Output: PUB  @ ipc:///tmp/ml-detector.ipc
 */
class ZMQHandler {
public:
    ZMQHandler(const DetectorConfig& config, 
               std::shared_ptr<ONNXModel> model,
               std::shared_ptr<FeatureExtractor> extractor);
    
    ~ZMQHandler();
    
    /**
     * @brief Inicia el handler (loop en thread separado)
     */
    void start();
    
    /**
     * @brief Detiene el handler gracefully
     */
    void stop();
    
    /**
     * @brief Verifica si está corriendo
     */
    bool is_running() const { return running_.load(); }
    
    /**
     * @brief Obtiene estadísticas
     */
    struct Stats {
        uint64_t events_received = 0;
        uint64_t events_processed = 0;
        uint64_t events_sent = 0;
        uint64_t deserialization_errors = 0;
        uint64_t feature_extraction_errors = 0;
        uint64_t inference_errors = 0;
        uint64_t attacks_detected = 0;
        double avg_processing_time_ms = 0.0;
    };
    
    Stats get_stats() const;
    void reset_stats();
    
private:
    // Main loop (corre en thread separado)
    void run();
    
    // Procesar un evento individual
    void process_event(const std::string& message);
    
    // Enviar evento enriquecido
    void send_enriched_event(const protobuf::NetworkSecurityEvent& event);
    
    // Configuración
    const DetectorConfig& config_;
    
    // ZMQ
    zmq::context_t context_;
    std::unique_ptr<zmq::socket_t> input_socket_;
    std::unique_ptr<zmq::socket_t> output_socket_;
    
    // ML Components
    std::shared_ptr<ONNXModel> model_;
    std::shared_ptr<FeatureExtractor> extractor_;
    
    // Threading
    std::atomic<bool> running_;
    std::unique_ptr<std::thread> worker_thread_;
    
    // Logging
    std::shared_ptr<spdlog::logger> logger_;
    
    // Stats
    mutable std::mutex stats_mutex_;
    Stats stats_;
    std::chrono::steady_clock::time_point last_stats_report_;
};

} // namespace ml_detector
