#include "zmq_handler.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <chrono>
#include <sstream>

namespace ml_detector {

    ZMQHandler::ZMQHandler(const DetectorConfig& config,
                       std::shared_ptr<ONNXModel> level1_model,
                       std::shared_ptr<FeatureExtractor> extractor,
                       std::shared_ptr<ONNXModel> level2_ddos_model,
                       std::shared_ptr<ml_defender::RansomwareDetector> ransomware_detector)  // ⬅️ AÑADIR parámetro
    : config_(config)
    , context_(1)  // 1 IO thread
    , level1_model_(level1_model)
    , level2_ddos_model_(level2_ddos_model)
    , ransomware_detector_(ransomware_detector)  // ⬅️ AÑADIR inicialización
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
            auto [pred_label, pred_confidence] = level1_model_->predict(features);
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
        // todo ojito, que esto es un hack para provocar que todo es un ataque y asi probar el modelo DDOS y Ransomware
        if (true || label == 1) {
            // Level 1 detectó ATTACK - activar Level 2 DDoS si disponible
            event.set_threat_category("ATTACK");

            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.attacks_detected++;
            }

            // ========== LEVEL 2: DDoS Binary Classification ==========
            if (level2_ddos_model_) {
                try {
                    logger_->debug("🔍 Running Level 2 DDoS classification...");

                    // SAFETY CHECK 1: Verificar que network_features existe
                    if (!event.has_network_features()) {
                        logger_->error("❌ Event {} missing network_features, skipping Level 2 DDoS",
                                      event.event_id());
                        // Continuar sin Level 2, no crashear
                        return;
                    }

                    const auto& nf = event.network_features();
                    logger_->debug("   Network features available for Level 2 extraction");

                    // SAFETY CHECK 2: Extraer features con manejo de errores
                    std::vector<float> features_l2;
                    try {
                        features_l2 = extractor_->extract_level2_ddos_features(nf);

                        // SAFETY CHECK 3: Validar tamaño de features
                        if (features_l2.size() != 8) {
                            logger_->error("❌ Level 2 DDoS feature extraction returned {} features, expected 8",
                                          features_l2.size());
                            return;
                        }

                        // Log features para debug (opcional, comentar en producción)
                        logger_->debug("   Level 2 DDoS Features (8):");
                        logger_->debug("     [0] Bwd Packet Length Max: {:.2f}", features_l2[0]);
                        logger_->debug("     [1] Flow Bytes/s:          {:.2f}", features_l2[1]);
                        logger_->debug("     [2] Fwd IAT Total:         {:.2f}", features_l2[2]);
                        logger_->debug("     [3] Bwd IAT Total:         {:.2f}", features_l2[3]);
                        logger_->debug("     [4] FIN Flag Count:        {:.2f}", features_l2[4]);
                        logger_->debug("     [5] Fwd PSH Flags:         {:.2f}", features_l2[5]);
                        logger_->debug("     [6] Active Mean:           {:.2f}", features_l2[6]);
                        logger_->debug("     [7] Idle Mean:             {:.2f}", features_l2[7]);

						// ═══════════════════════════════════════════════════════════
                        // ANÁLISIS DE FEATURES PARA TOMA DE DECISIONES
                        // ═══════════════════════════════════════════════════════════
                        int zero_features = 0;
                        int non_zero_features = 0;
                        for (size_t i = 0; i < features_l2.size(); ++i) {
                            if (features_l2[i] == 0.0f) {
                                zero_features++;
                            } else {
                                non_zero_features++;
                            }
                        }
                        logger_->debug("   📊 Feature Analysis: {} non-zero, {} zero ({}% coverage)",
                                      non_zero_features, zero_features,
                                      (non_zero_features * 100.0f / features_l2.size()));

                        if (features_l2[6] == 0.0f && features_l2[7] == 0.0f) {
                            logger_->debug("   ⚠️  Active Mean & Idle Mean = 0 (esperado, sniffer no las captura aún)");
                        }
                        // ═══════════════════════════════════════════════════════════


                    } catch (const std::exception& e) {
                        logger_->error("❌ Level 2 DDoS feature extraction failed: {}", e.what());
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        stats_.feature_extraction_errors++;
                        return;
                    }

                    // SAFETY CHECK 4: Predicción con manejo de errores
                    int64_t label_l2 = -1;
                    float confidence_l2 = 0.0f;

                    try {
                        auto [pred_label, pred_confidence] = level2_ddos_model_->predict(features_l2);
                        label_l2 = pred_label;
                        confidence_l2 = pred_confidence;

                        logger_->debug("🤖 Level 2 DDoS prediction: label={} ({}), confidence={:.4f}",
                                      label_l2, (label_l2 == 0 ? "NOT-DDOS" : "DDOS"), confidence_l2);

                    } catch (const std::exception& e) {
                        logger_->error("❌ Level 2 DDoS inference failed: {}", e.what());
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        stats_.inference_errors++;
                        return;
                    }

                    // SAFETY CHECK 5: Validar predicción
                    if (label_l2 < 0 || label_l2 > 1) {
                        logger_->error("❌ Invalid Level 2 DDoS prediction label: {}", label_l2);
                        return;
                    }

                    if (confidence_l2 < 0.0f || confidence_l2 > 1.0f) {
                        logger_->warn("⚠️  Invalid Level 2 DDoS confidence: {:.4f}, clamping", confidence_l2);
                        confidence_l2 = std::max(0.0f, std::min(1.0f, confidence_l2));
                    }

                    // ========== ENRIQUECER EVENTO CON PREDICCIÓN LEVEL 2 ==========
                    auto* level2_ddos_pred = ml_analysis->add_level2_specialized_predictions();
                    level2_ddos_pred->set_model_name("level2_ddos_binary_detector");
                    level2_ddos_pred->set_model_version("1.0.0");
                    level2_ddos_pred->set_model_type(protobuf::ModelPrediction::RANDOM_FOREST_DDOS);
                    level2_ddos_pred->set_prediction_class(label_l2 == 0 ? "NOT-DDOS" : "DDOS");
                    level2_ddos_pred->set_confidence_score(confidence_l2);

                    // Actualizar threat category si es DDoS
                    if (label_l2 == 1 && confidence_l2 >= config_.ml.thresholds.level2_ddos) {
                        event.set_threat_category("DDOS");
                        ml_analysis->set_final_threat_classification("DDOS");

                        logger_->warn("🔴 DDoS ATTACK CONFIRMED: event={}, L1_conf={:.2f}%, L2_conf={:.2f}%",
                                     event.event_id(), confidence * 100, confidence_l2 * 100);
                    } else if (label_l2 == 1) {
                        // DDoS detectado pero bajo confianza
                        logger_->info("🟡 Possible DDoS (low confidence): event={}, confidence={:.2f}% (threshold={:.2f}%)",
                                     event.event_id(), confidence_l2 * 100, config_.ml.thresholds.level2_ddos * 100);
                    } else {
                        // NOT-DDOS
                        logger_->info("🟢 Attack detected but NOT DDoS: event={}, confidence={:.2f}%",
                                     event.event_id(), confidence_l2 * 100);
                    }

                    logger_->debug("✅ Level 2 DDoS classification completed successfully");

                } catch (const std::exception& e) {
                    // Catch-all para cualquier error no manejado
                    logger_->error("❌ Unexpected error in Level 2 DDoS processing: {}", e.what());
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.inference_errors++;
                    // Continuar sin crashear
                }
            } else {
                // Modelo Level 2 no cargado
                logger_->debug("⚠️  Level 2 DDoS model not loaded, skipping specialized detection");
            }
            // ========== FIN LEVEL 2 ==========
            // Después del bloque de Level 2 DDoS, añadir:

        // ========== LEVEL 2: Ransomware Detection ==========
if (ransomware_detector_ && config_.ml.level2.ransomware.enabled) {
    try {
        logger_->debug("🔍 Running Level 2 Ransomware classification...");

        // SAFETY CHECK 1: Verificar que network_features existe
        if (!event.has_network_features()) {
            logger_->error("❌ Event {} missing network_features, skipping Level 2 Ransomware",
                          event.event_id());
            return;
        }

        const auto& nf = event.network_features();
        logger_->debug("   Network features available for Level 2 Ransomware extraction");

        // SAFETY CHECK 2: Extraer features con manejo de errores
        std::vector<float> ransomware_features_vec;
        try {
            ransomware_features_vec = extractor_->extract_level2_ransomware_features(nf);

            // SAFETY CHECK 3: Validar tamaño de features
            if (ransomware_features_vec.size() != 10) {
                logger_->error("❌ Level 2 Ransomware feature extraction returned {} features, expected 10",
                              ransomware_features_vec.size());
                return;
            }

            // Log features para debug
            logger_->debug("   Level 2 Ransomware Features (10):");
            logger_->debug("     [0] IO Intensity:           {:.3f}", ransomware_features_vec[0]);
            logger_->debug("     [1] Entropy (36% import):   {:.3f}", ransomware_features_vec[1]);
            logger_->debug("     [2] Resource Usage (25%):   {:.3f}", ransomware_features_vec[2]);
            logger_->debug("     [3] Network Activity:       {:.3f}", ransomware_features_vec[3]);
            logger_->debug("     [4] File Operations:        {:.3f}", ransomware_features_vec[4]);
            logger_->debug("     [5] Process Anomaly:        {:.3f}", ransomware_features_vec[5]);
            logger_->debug("     [6] Temporal Pattern:       {:.3f}", ransomware_features_vec[6]);
            logger_->debug("     [7] Access Frequency:       {:.3f}", ransomware_features_vec[7]);
            logger_->debug("     [8] Data Volume:            {:.3f}", ransomware_features_vec[8]);
            logger_->debug("     [9] Behavior Consistency:   {:.3f}", ransomware_features_vec[9]);

        } catch (const std::exception& e) {
            logger_->error("❌ Level 2 Ransomware feature extraction failed: {}", e.what());
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.feature_extraction_errors++;
            return;
        }

        // Convertir a estructura del detector
        ml_defender::RansomwareDetector::Features rf_features{
            .io_intensity = ransomware_features_vec[0],
            .entropy = ransomware_features_vec[1],
            .resource_usage = ransomware_features_vec[2],
            .network_activity = ransomware_features_vec[3],
            .file_operations = ransomware_features_vec[4],
            .process_anomaly = ransomware_features_vec[5],
            .temporal_pattern = ransomware_features_vec[6],
            .access_frequency = ransomware_features_vec[7],
            .data_volume = ransomware_features_vec[8],
            .behavior_consistency = ransomware_features_vec[9]
        };

        // SAFETY CHECK 4: Predicción <100μs
        ml_defender::RansomwareDetector::Prediction result;
        try {
            result = ransomware_detector_->predict(rf_features);

            logger_->debug("🤖 Level 2 Ransomware prediction: class={} ({}), confidence={:.4f}",
                          result.class_id,
                          (result.class_id == 0 ? "BENIGN" : "RANSOMWARE"),
                          result.ransomware_prob);

        } catch (const std::exception& e) {
            logger_->error("❌ Level 2 Ransomware inference failed: {}", e.what());
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.inference_errors++;
            return;
        }

        // SAFETY CHECK 5: Validar predicción
        if (result.class_id < 0 || result.class_id > 1) {
            logger_->error("❌ Invalid Level 2 Ransomware prediction class: {}", result.class_id);
            return;
        }

        if (result.ransomware_prob < 0.0f || result.ransomware_prob > 1.0f) {
            logger_->warn("⚠️  Invalid Level 2 Ransomware confidence: {:.4f}, clamping",
                         result.ransomware_prob);
            result.ransomware_prob = std::max(0.0f, std::min(1.0f, result.ransomware_prob));
        }

        // ========== ENRIQUECER EVENTO CON PREDICCIÓN RANSOMWARE ==========
        auto* level2_ransomware_pred = ml_analysis->add_level2_specialized_predictions();
        level2_ransomware_pred->set_model_name("ransomware_detector_embedded_cpp20");
        level2_ransomware_pred->set_model_version("1.0.0");
        level2_ransomware_pred->set_model_type(protobuf::ModelPrediction::RANDOM_FOREST_RANSOMWARE);
        level2_ransomware_pred->set_prediction_class(
            result.class_id == 0 ? "BENIGN" : "RANSOMWARE"
        );
        level2_ransomware_pred->set_confidence_score(result.ransomware_prob);

        // Actualizar threat category si es Ransomware
        if (result.class_id == 1 &&
            result.ransomware_prob >= config_.ml.thresholds.level2_ransomware) {

            event.set_threat_category("RANSOMWARE");
            ml_analysis->set_final_threat_classification("RANSOMWARE");

            logger_->warn("🔴 RANSOMWARE ATTACK CONFIRMED: event={}, "
                         "L1_conf={:.2f}%, L2_conf={:.2f}%, entropy={:.3f}",
                         event.event_id(),
                         confidence * 100,
                         result.ransomware_prob * 100,
                         ransomware_features_vec[1]);  // entropy (most important)

            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.attacks_detected++;
            }

        } else if (result.class_id == 1) {
            // Ransomware detectado pero bajo confianza
            logger_->info("🟡 Possible Ransomware (low confidence): event={}, "
                         "confidence={:.2f}% (threshold={:.2f}%)",
                         event.event_id(),
                         result.ransomware_prob * 100,
                         config_.ml.thresholds.level2_ransomware * 100);
        } else {
            // BENIGN
            logger_->info("🟢 Attack detected but NOT Ransomware: event={}, "
                         "confidence={:.2f}%",
                         event.event_id(),
                         result.benign_prob * 100);
        }

        logger_->debug("✅ Level 2 Ransomware classification completed successfully");

    } catch (const std::exception& e) {
        // Catch-all para cualquier error no manejado
        logger_->error("❌ Unexpected error in Level 2 Ransomware processing: {}", e.what());
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.inference_errors++;
        // Continuar sin crashear
    }
}
// ========== FIN LEVEL 2 RANSOMWARE ==========
        } else {
            // Label = 0 (BENIGN)
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
