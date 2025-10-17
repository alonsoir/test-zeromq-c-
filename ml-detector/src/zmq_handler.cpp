#include "zmq_handler.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <chrono>
#include <sstream>

namespace ml_detector {

ZMQHandler::ZMQHandler(const DetectorConfig& config, 
                       std::shared_ptr<ONNXModel> model,
                       std::shared_ptr<FeatureExtractor> extractor)
    : config_(config)
    , context_(1)  // 1 IO thread
    , model_(model)
    , extractor_(extractor)
    , running_(false)
    , logger_(spdlog::get("ml-detector"))
    , last_stats_report_(std::chrono::steady_clock::now())
{
    if (!logger_) {
        logger_ = spdlog::stdout_color_mt("zmq-handler");
    }
    
    logger_->info("🔌 Initializing ZMQ Handler");
    
    // Crear sockets
    try {
        // Input socket (PULL)
        input_socket_ = std::make_unique<zmq::socket_t>(context_, zmq::socket_type::pull);
        
        // Configurar según config
        int hwm = config_.network.input_socket.high_water_mark;
        input_socket_->set(zmq::sockopt::rcvhwm, hwm);
        input_socket_->set(zmq::sockopt::linger, config_.zmq.connection_settings.linger_ms);
        
        // Connect o Bind según config
        if (config_.network.input_socket.mode == "connect") {
            logger_->info("   Input: PULL connect {}", config_.network.input_socket.endpoint);
            input_socket_->connect(config_.network.input_socket.endpoint);
        } else {
            logger_->info("   Input: PULL bind {}", config_.network.input_socket.endpoint);
            input_socket_->bind(config_.network.input_socket.endpoint);
        }
        
        // Output socket (PUB)
        output_socket_ = std::make_unique<zmq::socket_t>(context_, zmq::socket_type::pub);
        
        hwm = config_.network.output_socket.high_water_mark;
        output_socket_->set(zmq::sockopt::sndhwm, hwm);
        output_socket_->set(zmq::sockopt::linger, config_.zmq.connection_settings.linger_ms);
        
        // Connect o Bind según config
        if (config_.network.output_socket.mode == "connect") {
            logger_->info("   Output: PUB connect {}", config_.network.output_socket.endpoint);
            output_socket_->connect(config_.network.output_socket.endpoint);
        } else {
            logger_->info("   Output: PUB bind {}", config_.network.output_socket.endpoint);
            output_socket_->bind(config_.network.output_socket.endpoint);
        }
        
        logger_->info("✅ ZMQ sockets initialized successfully");
        
    } catch (const zmq::error_t& e) {
        logger_->error("❌ ZMQ initialization failed: {}", e.what());
        throw;
    }
}

ZMQHandler::~ZMQHandler() {
    stop();
}

void ZMQHandler::start() {
    if (running_.load()) {
        logger_->warn("ZMQ Handler already running");
        return;
    }
    
    logger_->info("🚀 Starting ZMQ Handler");
    running_.store(true);
    
    worker_thread_ = std::make_unique<std::thread>(&ZMQHandler::run, this);
    
    logger_->info("✅ ZMQ Handler started");
}

void ZMQHandler::stop() {
    if (!running_.load()) {
        return;
    }
    
    logger_->info("🛑 Stopping ZMQ Handler...");
    running_.store(false);
    
    if (worker_thread_ && worker_thread_->joinable()) {
        worker_thread_->join();
    }
    
    // Cerrar sockets
    if (input_socket_) {
        input_socket_->close();
    }
    if (output_socket_) {
        output_socket_->close();
    }
    
    logger_->info("✅ ZMQ Handler stopped");
}

void ZMQHandler::run() {
    logger_->info("📥 ZMQ Handler loop started");
    
    auto stats_interval = std::chrono::seconds(config_.monitoring.stats_interval_seconds);
    
    while (running_.load()) {
        try {
            // Recibir mensaje con timeout
            zmq::message_t message;
            auto result = input_socket_->recv(message, zmq::recv_flags::dontwait);
            
            if (result) {
                // Mensaje recibido
                std::string msg_data(static_cast<char*>(message.data()), message.size());
                
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.events_received++;
                }
                
                process_event(msg_data);
                
            } else {
                // No hay mensajes, dormir un poco
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            
            // Reportar stats periódicamente
            auto now = std::chrono::steady_clock::now();
            if (now - last_stats_report_ >= stats_interval) {
                auto stats = get_stats();
                logger_->info("📊 Stats: received={}, processed={}, sent={}, attacks={}, errors=(deser:{}, feat:{}, inf:{})",
                             stats.events_received, stats.events_processed, stats.events_sent,
                             stats.attacks_detected, stats.deserialization_errors,
                             stats.feature_extraction_errors, stats.inference_errors);
                last_stats_report_ = now;
            }
            
        } catch (const zmq::error_t& e) {
            if (e.num() == ETERM) {
                // Context terminated, salir
                break;
            }
            logger_->error("ZMQ error: {}", e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } catch (const std::exception& e) {
            logger_->error("Unexpected error in handler loop: {}", e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    logger_->info("📥 ZMQ Handler loop stopped");
}

void ZMQHandler::process_event(const std::string& message) {
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // 1. Deserializar protobuf
        protobuf::NetworkSecurityEvent event;
        if (!event.ParseFromString(message)) {
            logger_->error("Failed to deserialize protobuf message");
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.deserialization_errors++;
            return;
        }
        
        logger_->debug("📦 Event received: id={}", event.event_id());
        
        // 2. Extraer features
        std::vector<float> features;
        try {
            features = extractor_->extract_level1_features(event);
            
            // Validar features
            if (!extractor_->validate_features(features)) {
                logger_->error("Feature validation failed for event {}", event.event_id());
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.feature_extraction_errors++;
                return;
            }
            
        } catch (const std::exception& e) {
            logger_->error("Feature extraction failed: {}", e.what());
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.feature_extraction_errors++;
            return;
        }
        
        // 3. Inference
        int64_t label = -1;
        float confidence = 0.0f;
        
        try {
            auto [pred_label, pred_confidence] = model_->predict(features);
            label = pred_label;
            confidence = pred_confidence;
            
            logger_->debug("🤖 Prediction: label={} ({}), confidence={:.4f}", 
                          label, (label == 0 ? "BENIGN" : "ATTACK"), confidence);
            
        } catch (const std::exception& e) {
            logger_->error("Inference failed: {}", e.what());
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.inference_errors++;
            return;
        }
        
        // 4. Enriquecer evento con ML analysis
        auto* ml_analysis = event.mutable_ml_analysis();
        if (!ml_analysis) {
            ml_analysis = event.mutable_ml_analysis();
        }
        
        // Level 1 prediction
        auto* level1_pred = ml_analysis->mutable_level1_general_detection();
        level1_pred->set_model_name("level1_attack_detector");
        level1_pred->set_model_version("1.0.0");
        level1_pred->set_model_type(protobuf::ModelPrediction::RANDOM_FOREST_GENERAL);
        level1_pred->set_prediction_class(label == 0 ? "BENIGN" : "ATTACK");
        level1_pred->set_confidence_score(confidence);
        
        // Set attack detected flag
        ml_analysis->set_attack_detected_level1(label == 1);
        ml_analysis->set_level1_confidence(confidence);
        
        // Update overall event classification
        event.set_final_classification(label == 0 ? "BENIGN" : "MALICIOUS");
        event.set_overall_threat_score(label == 1 ? confidence : (1.0 - confidence));
        
        if (label == 1) {
            event.set_threat_category("ATTACK");
            
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.attacks_detected++;
        } else {
            event.set_threat_category("NORMAL");
        }
        
        // 5. Enviar evento enriquecido
        send_enriched_event(event);
        
        // 6. Stats
        auto end_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.events_processed++;
            
            // Running average
            stats_.avg_processing_time_ms = 
            (stats_.avg_processing_time_ms * static_cast<double>(stats_.events_processed - 1) + duration_ms) /
            static_cast<double>(stats_.events_processed);
        }
        
        if (label == 1) {
            logger_->info("🚨 ATTACK DETECTED: event={}, confidence={:.2f}%, processing_time={:.2f}ms",
                         event.event_id(), confidence * 100, duration_ms);
        } else {
            logger_->debug("✅ Benign traffic: event={}, confidence={:.2f}%, processing_time={:.2f}ms",
                          event.event_id(), confidence * 100, duration_ms);
        }
        
    } catch (const std::exception& e) {
        logger_->error("Failed to process event: {}", e.what());
    }
}

void ZMQHandler::send_enriched_event(const protobuf::NetworkSecurityEvent& event) {
    try {
        std::string serialized;
        if (!event.SerializeToString(&serialized)) {
            logger_->error("Failed to serialize enriched event {}", event.event_id());
            return;
        }
        
        zmq::message_t message(serialized.size());
        memcpy(message.data(), serialized.data(), serialized.size());
        
        auto result = output_socket_->send(message, zmq::send_flags::dontwait);
        
        if (result) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.events_sent++;
            
            logger_->debug("📤 Event sent: id={}, size={} bytes", 
                          event.event_id(), serialized.size());
        } else {
            logger_->warn("Failed to send event {} (queue full?)", event.event_id());
        }
        
    } catch (const zmq::error_t& e) {
        logger_->error("ZMQ send error: {}", e.what());
    }
}

ZMQHandler::Stats ZMQHandler::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void ZMQHandler::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = Stats{};
    last_stats_report_ = std::chrono::steady_clock::now();
    logger_->info("📊 Stats reset");
}

} // namespace ml_detector
