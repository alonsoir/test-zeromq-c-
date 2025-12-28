#include "zmq_handler.hpp"
#include "rag_logger.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <chrono>
#include <sstream>
#include <unistd.h>  // Para sysconf()
#include <cstdio>    // Para FILE, fopen, etc.

/*
 * todo esta clase necesita refactorizacion porque hay muchas números mágicos.
 */
namespace ml_detector {

ZMQHandler::ZMQHandler(
    const DetectorConfig& config,
    std::shared_ptr<ONNXModel> level1_model,
    std::shared_ptr<FeatureExtractor> extractor,
    std::shared_ptr<ml_defender::DDoSDetector> ddos_detector,
    std::shared_ptr<ml_defender::RansomwareDetector> ransomware_detector,
    std::shared_ptr<ml_defender::TrafficDetector> traffic_detector,
    std::shared_ptr<ml_defender::InternalDetector> internal_detector,
    std::shared_ptr<crypto::CryptoManager> crypto_manager  // 🎯 DAY 27: NEW
)
    : config_(config)
    , context_(1)  // 1 IO thread
    , level1_model_(level1_model)
    , ddos_detector_(ddos_detector)
    , ransomware_detector_(ransomware_detector)
    , traffic_detector_(traffic_detector)
    , internal_detector_(internal_detector)
    , extractor_(extractor)
    , crypto_manager_(crypto_manager)  // 🎯 DAY 27: NEW
    , running_(false)
    , logger_(spdlog::get("ml-detector"))
    , last_stats_report_(std::chrono::steady_clock::now())
    , start_time_(std::chrono::system_clock::now())
{
    if (!logger_) {
        logger_ = spdlog::stdout_color_mt("zmq-handler");
    }

    logger_->info("🔌 Initializing ZMQ Handler");

    // 🎯 DAY 27: Log crypto status
    if (crypto_manager_) {
        logger_->info("🔐 Crypto-Transport enabled (ChaCha20-Poly1305 + LZ4)");
    } else {
        logger_->error("❌ Crypto-Transport NOT initialized");
        throw std::runtime_error("CryptoManager required for ZMQHandler");
    }

    // Log detector status
    logger_->info("📊 ML Detectors loaded:");
    logger_->info("   Level 1: General Attack (ONNX)");
    if (ddos_detector_) {
        logger_->info("   Level 2: DDoS ({} trees, {} features)",
                     ddos_detector_->num_trees(), ddos_detector_->num_features());
    }
    if (ransomware_detector_) {
        logger_->info("   Level 2: Ransomware ({} trees, {} features)",
                     ransomware_detector_->num_trees(), ransomware_detector_->num_features());
    }
    if (traffic_detector_) {
        logger_->info("   Level 3: Traffic ({} trees, {} features)",
                     traffic_detector_->num_trees(), traffic_detector_->num_features());
    }
    if (internal_detector_) {
        logger_->info("   Level 3: Internal ({} trees, {} features)",
                     internal_detector_->num_trees(), internal_detector_->num_features());
    }

    // Crear sockets (sin cambios)
    try {
        // Input socket (PULL)
        input_socket_ = std::make_unique<zmq::socket_t>(context_, zmq::socket_type::pull);

        int hwm = config_.network.input_socket.high_water_mark;
        input_socket_->set(zmq::sockopt::rcvhwm, hwm);
        input_socket_->set(zmq::sockopt::linger, config_.zmq.connection_settings.linger_ms);

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

    // 🎯 DAY 14: Initialize RAG Logger
    try {
        rag_logger_ = ml_defender::create_rag_logger_from_config(
            "../config/rag_logger_config.json",
            logger_
        );
        logger_->info("✅ RAG Logger initialized successfully");
    } catch (const std::exception& e) {
        logger_->error("❌ Failed to initialize RAG Logger: {}", e.what());
        logger_->warn("⚠️  Continuing without RAG logging");
        rag_logger_ = nullptr;
    }
}

ZMQHandler::~ZMQHandler() {
    stop();
    if (rag_logger_) {
        logger_->info("🔄 Flushing RAG logger...");
        rag_logger_->flush();

        // Print statistics
        auto stats = rag_logger_->get_statistics();
        logger_->info("📊 RAG Statistics: {}", stats.dump(2));
    }
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
            // ================================================================
            // 🎯 DAY 27: RECEIVE ENCRYPTED MESSAGE
            // ================================================================
            zmq::message_t encrypted_message;
            auto result = input_socket_->recv(encrypted_message, zmq::recv_flags::dontwait);

            if (result) {
                // Extract encrypted data
                std::string encrypted_data(
                    static_cast<char*>(encrypted_message.data()),
                    encrypted_message.size()
                );

                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.events_received++;
                }

                // Decrypt → Decompress (pattern from firewall-acl-agent)
                try {
                    auto decrypted = crypto_manager_->decrypt(encrypted_data);
                    auto decompressed = crypto_manager_->decompress(decrypted);

                    logger_->trace("🔓 Decrypted: {} bytes → {} bytes (decompressed)",
                                  encrypted_data.size(), decompressed.size());

                    // Process the decrypted/decompressed message
                    process_event(decompressed);

                } catch (const std::exception& e) {
                    logger_->error("❌ Decryption/decompression failed: {}", e.what());
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.deserialization_errors++;
                }

            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            // Reportar stats
            auto now = std::chrono::steady_clock::now();
            if (now - last_stats_report_ >= stats_interval) {
                auto stats = get_stats();
                logger_->info("📊 Stats: received={}, processed={}, sent={}, attacks={}, "
                             "errors=(deser:{}, feat:{}, inf:{})",
                             stats.events_received, stats.events_processed, stats.events_sent,
                             stats.attacks_detected, stats.deserialization_errors,
                             stats.feature_extraction_errors, stats.inference_errors);
                last_stats_report_ = now;
            }

        } catch (const zmq::error_t& e) {
            if (e.num() == ETERM) break;
            logger_->error("ZMQ error: {}", e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } catch (const std::exception& e) {
            logger_->error("Unexpected error: {}", e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    logger_->info("📥 ZMQ Handler loop stopped");
}

void ZMQHandler::process_event(const std::string& message) {
    auto start_time = std::chrono::steady_clock::now();

    try {
        // ====================================================================
        // DESERIALIZATION (message already decrypted/decompressed)
        // ====================================================================
        protobuf::NetworkSecurityEvent event;
        if (!event.ParseFromString(message)) {
            logger_->error("Failed to deserialize protobuf message");
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.deserialization_errors++;
            return;
        }

        logger_->debug("📦 Event received: id={}", event.event_id());

        // ====================================================================
        // DAY 13: READ FAST DETECTOR SCORE
        // ====================================================================
        double fast_score = event.fast_detector_score();
        bool fast_triggered = event.fast_detector_triggered();
        std::string fast_reason = event.fast_detector_reason();

        logger_->debug("🎯 Fast Detector: score={:.4f}, triggered={}, reason={}",
                       fast_score, fast_triggered, fast_reason);

        // ====================================================================
        // LEVEL 1: GENERAL ATTACK DETECTION (ONNX)
        // ====================================================================
        std::vector<float> features_l1;
        try {
            features_l1 = extractor_->extract_level1_features(event);

            if (!extractor_->validate_features(features_l1)) {
                logger_->error("Feature validation failed for event {}", event.event_id());
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.feature_extraction_errors++;
                return;
            }

        } catch (const std::exception& e) {
            logger_->error("Level 1 feature extraction failed: {}", e.what());
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.feature_extraction_errors++;
            return;
        }

        // Level 1 Inference
        int64_t label_l1 = -1;
        float confidence_l1 = 0.0f;

        try {
            auto [pred_label, pred_confidence] = level1_model_->predict(features_l1);
            label_l1 = pred_label;
            confidence_l1 = pred_confidence;

            logger_->debug("🤖 Level 1: label={} ({}), confidence={:.4f}",
                          label_l1, (label_l1 == 0 ? "BENIGN" : "ATTACK"), confidence_l1);

        } catch (const std::exception& e) {
            logger_->error("Level 1 inference failed: {}", e.what());
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.inference_errors++;
            return;
        }

        // Enriquecer con Level 1
        auto* ml_analysis = event.mutable_ml_analysis();
        auto* level1_pred = ml_analysis->mutable_level1_general_detection();
        level1_pred->set_model_name("level1_attack_detector");
        level1_pred->set_model_version("1.0.0");
        level1_pred->set_model_type(protobuf::ModelPrediction::RANDOM_FOREST_GENERAL);
        level1_pred->set_prediction_class(label_l1 == 0 ? "BENIGN" : "ATTACK");
        level1_pred->set_confidence_score(confidence_l1);

        ml_analysis->set_attack_detected_level1(label_l1 == 1);
        ml_analysis->set_level1_confidence(confidence_l1);

        // ====================================================================
        // DAY 13: DUAL-SCORE ARCHITECTURE - Maximum Threat Wins
        // ====================================================================
        // Calculate ML score (from Level 1)
        double ml_score = label_l1 == 1 ? confidence_l1 : (1.0 - confidence_l1);
        event.set_ml_detector_score(ml_score);

        // Maximum Threat Wins
        double final_score = std::max(fast_score, ml_score);
        event.set_overall_threat_score(final_score);

        // Determine authoritative source
        double score_divergence = std::abs(fast_score - ml_score);

        if (score_divergence > 0.30) {
            // Divergencia significativa
            event.set_authoritative_source(protobuf::DETECTOR_SOURCE_DIVERGENCE);
            logger_->warn("⚠️  Score divergence: fast={:.4f}, ml={:.4f}, diff={:.4f}",
                          fast_score, ml_score, score_divergence);
        } else if (fast_triggered && ml_score > 0.5) {
            // Ambos detectores activos y concordantes
            event.set_authoritative_source(protobuf::DETECTOR_SOURCE_CONSENSUS);
        } else if (fast_score > ml_score) {
            event.set_authoritative_source(protobuf::DETECTOR_SOURCE_FAST_PRIORITY);
        } else {
            event.set_authoritative_source(protobuf::DETECTOR_SOURCE_ML_PRIORITY);
        }

        // Decision metadata
        // todo esos números mágicos dan miedo, además, esta función es ENORME!!
        auto* metadata = event.mutable_decision_metadata();
        metadata->set_score_divergence(score_divergence);
        metadata->set_requires_rag_analysis(score_divergence > 0.30 || final_score >= 0.85);
        metadata->set_confidence_level(std::min(fast_score, ml_score)); // Conservative

        // F1-Score Logging para validation
        logger_->info("[DUAL-SCORE] event={}, fast={:.4f}, ml={:.4f}, final={:.4f}, source={}, div={:.4f}",
                      event.event_id(), fast_score, ml_score, final_score,
                      protobuf::DetectorSource_Name(event.authoritative_source()),
                      score_divergence);

        // Set classification based on final score
        event.set_final_classification(final_score >= 0.70 ? "MALICIOUS" : "BENIGN");

        // ========================================================================
        // 🎯 DAY 14: RAG LOGGING
        // ========================================================================

        // Build ML context for RAG
        ml_defender::MLContext ml_context;

        // System state
        ml_context.events_processed_total = ++events_processed_total_;
        ml_context.events_in_last_minute = calculate_events_per_minute();
        ml_context.memory_usage_mb = get_memory_usage_mb();
        ml_context.cpu_usage_percent = 0.0;  // TODO: Implement if needed
        ml_context.uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now() - start_time_
        ).count();

        // Detection context (from Level 1 detector)
        ml_context.attack_family = "RANSOMWARE";  // TODO: Get from detector
        ml_context.level_1_label = label_l1 == 1 ? "MALICIOUS" : "BENIGN";
        ml_context.level_2_category = "UNKNOWN";   // Level 2 not always active
        ml_context.level_3_subcategory = "UNKNOWN"; // Level 3 not always active
        ml_context.level_1_confidence = confidence_l1;

        // Temporal window
        ml_context.window_start = std::chrono::system_clock::now() - std::chrono::seconds(30);
        ml_context.window_end = std::chrono::system_clock::now();
        ml_context.events_in_window = 1;  // TODO: Track window events

        // Investigation priority
        if (score_divergence > 0.40) {
            ml_context.investigation_priority = "HIGH";
        } else if (score_divergence > 0.30) {
            ml_context.investigation_priority = "MEDIUM";
        } else {
            ml_context.investigation_priority = "LOW";
        }

        // Log to RAG if enabled
        if (rag_logger_) {
            bool logged = rag_logger_->log_event(event, ml_context);
            if (logged) {
                logger_->debug("📝 Event logged to RAG: {}", event.event_id());
            }
        }

        // ====================================================================
        // LEVEL 2 & 3: SPECIALIZED DETECTORS (si Level 1 detectó ATTACK)
        // ====================================================================
        if (label_l1 == 1 && confidence_l1 >= config_.ml.thresholds.level1_attack) {
            event.set_threat_category("ATTACK");

            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.attacks_detected++;
            }

            // ================================================================
            // LEVEL 2: DDoS DETECTION (C++20 Embedded)
            // ================================================================
            if (ddos_detector_ && config_.ml.level2.ddos.enabled) {
                try {
                    logger_->debug("🔍 Running Level 2 DDoS classification...");

                    if (!event.has_network_features()) {
                        logger_->error("❌ Event {} missing network_features, skipping Level 2 DDoS",
                                      event.event_id());
                    } else {
                        const auto& nf = event.network_features();

                        // Extract DDoS features
                        std::vector<float> ddos_features_vec;
                        try {
                            ddos_features_vec = extractor_->extract_level2_ddos_features(nf);

                            if (ddos_features_vec.size() != 10) {
                                logger_->error("❌ DDoS feature extraction returned {} features, expected 10",
                                              ddos_features_vec.size());
                                throw std::runtime_error("Invalid DDoS feature count");
                            }

                            logger_->debug("   DDoS Features: syn_ack={:.3f}, entropy={:.3f}, amp={:.3f}",
                                          ddos_features_vec[0], ddos_features_vec[4], ddos_features_vec[5]);

                        } catch (const std::exception& e) {
                            logger_->error("❌ DDoS feature extraction failed: {}", e.what());
                            std::lock_guard<std::mutex> lock(stats_mutex_);
                            stats_.feature_extraction_errors++;
                            throw;
                        }

                        // Convert to detector structure
                        ml_defender::DDoSDetector::Features ddos_features{
                            .syn_ack_ratio = ddos_features_vec[0],
                            .packet_symmetry = ddos_features_vec[1],
                            .source_ip_dispersion = ddos_features_vec[2],
                            .protocol_anomaly_score = ddos_features_vec[3],
                            .packet_size_entropy = ddos_features_vec[4],
                            .traffic_amplification_factor = ddos_features_vec[5],
                            .flow_completion_rate = ddos_features_vec[6],
                            .geographical_concentration = ddos_features_vec[7],
                            .traffic_escalation_rate = ddos_features_vec[8],
                            .resource_saturation_score = ddos_features_vec[9]
                        };

                        // Predict (target: <100μs)
                        auto ddos_result = ddos_detector_->predict(ddos_features);

                        logger_->debug("🤖 DDoS: class={} ({}), conf={:.4f}",
                                      ddos_result.class_id,
                                      (ddos_result.class_id == 0 ? "NORMAL" : "DDOS"),
                                      ddos_result.probability);

                        // Enriquecer evento
                        auto* level2_ddos_pred = ml_analysis->add_level2_specialized_predictions();
                        level2_ddos_pred->set_model_name("ddos_detector_embedded_cpp20");
                        level2_ddos_pred->set_model_version("1.0.0");
                        level2_ddos_pred->set_model_type(protobuf::ModelPrediction::RANDOM_FOREST_DDOS);
                        level2_ddos_pred->set_prediction_class(
                            ddos_result.class_id == 0 ? "NORMAL" : "DDOS"
                        );
                        level2_ddos_pred->set_confidence_score(ddos_result.ddos_prob);

                        if (ddos_result.is_ddos(config_.ml.thresholds.level2_ddos)) {
                            event.set_threat_category("DDOS");
                            ml_analysis->set_final_threat_classification("DDOS");

                            logger_->warn("🔴 DDoS ATTACK: event={}, L1={:.2f}%, L2={:.2f}%",
                                         event.event_id(), confidence_l1 * 100, ddos_result.ddos_prob * 100);
                        }
                    }

                } catch (const std::exception& e) {
                    logger_->error("❌ Level 2 DDoS processing failed: {}", e.what());
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.inference_errors++;
                }
            }

            // ================================================================
            // LEVEL 2: RANSOMWARE DETECTION (C++20 Embedded)
            // ================================================================
            if (ransomware_detector_ && config_.ml.level2.ransomware.enabled) {
                try {
                    logger_->debug("🔍 Running Level 2 Ransomware classification...");

                    if (!event.has_network_features()) {
                        logger_->error("❌ Event {} missing network_features, skipping Ransomware",
                                      event.event_id());
                    } else {
                        const auto& nf = event.network_features();

                        // Extract Ransomware features
                        std::vector<float> ransomware_features_vec;
                        try {
                            ransomware_features_vec = extractor_->extract_level2_ransomware_features(nf);

                            if (ransomware_features_vec.size() != 10) {
                                logger_->error("❌ Ransomware feature extraction returned {} features, expected 10",
                                              ransomware_features_vec.size());
                                throw std::runtime_error("Invalid Ransomware feature count");
                            }

                            logger_->debug("   Ransomware Features: entropy={:.3f}, io={:.3f}, resource={:.3f}",
                                          ransomware_features_vec[1], ransomware_features_vec[0],
                                          ransomware_features_vec[2]);

                        } catch (const std::exception& e) {
                            logger_->error("❌ Ransomware feature extraction failed: {}", e.what());
                            std::lock_guard<std::mutex> lock(stats_mutex_);
                            stats_.feature_extraction_errors++;
                            throw;
                        }

                        // Convert to detector structure
                        ml_defender::RansomwareDetector::Features ransomware_features{
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

                        // Predict (target: <100μs)
                        auto ransomware_result = ransomware_detector_->predict(ransomware_features);

                        logger_->debug("🤖 Ransomware: class={} ({}), conf={:.4f}",
                                      ransomware_result.class_id,
                                      (ransomware_result.class_id == 0 ? "BENIGN" : "RANSOMWARE"),
                                      ransomware_result.probability);

                        // Enriquecer evento
                        auto* level2_ransomware_pred = ml_analysis->add_level2_specialized_predictions();
                        level2_ransomware_pred->set_model_name("ransomware_detector_embedded_cpp20");
                        level2_ransomware_pred->set_model_version("1.0.0");
                        level2_ransomware_pred->set_model_type(protobuf::ModelPrediction::RANDOM_FOREST_RANSOMWARE);
                        level2_ransomware_pred->set_prediction_class(
                            ransomware_result.class_id == 0 ? "BENIGN" : "RANSOMWARE"
                        );
                        level2_ransomware_pred->set_confidence_score(ransomware_result.ransomware_prob);

                        if (ransomware_result.is_ransomware(config_.ml.thresholds.level2_ransomware)) {
                            event.set_threat_category("RANSOMWARE");
                            ml_analysis->set_final_threat_classification("RANSOMWARE");

                            logger_->warn("🔴 RANSOMWARE ATTACK: event={}, L1={:.2f}%, L2={:.2f}%, entropy={:.3f}",
                                         event.event_id(), confidence_l1 * 100,
                                         ransomware_result.ransomware_prob * 100,
                                         ransomware_features_vec[1]);
                        }
                    }

                } catch (const std::exception& e) {
                    logger_->error("❌ Level 2 Ransomware processing failed: {}", e.what());
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.inference_errors++;
                }
            }

            // ================================================================
            // LEVEL 3: TRAFFIC CLASSIFICATION (Internet vs Internal)
            // ================================================================
            if (traffic_detector_ && config_.ml.level3.web.enabled) {
                try {
                    logger_->debug("🔍 Running Level 3 Traffic classification...");

                    if (!event.has_network_features()) {
                        logger_->error("❌ Event {} missing network_features, skipping Traffic",
                                      event.event_id());
                    } else {
                        const auto& nf = event.network_features();

                        // Extract Traffic features
                        std::vector<float> traffic_features_vec;
                        try {
                            traffic_features_vec = extractor_->extract_level3_traffic_features(nf);

                            if (traffic_features_vec.size() != 10) {
                                logger_->error("❌ Traffic feature extraction returned {} features, expected 10",
                                              traffic_features_vec.size());
                                throw std::runtime_error("Invalid Traffic feature count");
                            }

                        } catch (const std::exception& e) {
                            logger_->error("❌ Traffic feature extraction failed: {}", e.what());
                            std::lock_guard<std::mutex> lock(stats_mutex_);
                            stats_.feature_extraction_errors++;
                            throw;
                        }

                        // Convert to detector structure
                        ml_defender::TrafficDetector::Features traffic_features{
                            .packet_rate = traffic_features_vec[0],
                            .connection_rate = traffic_features_vec[1],
                            .tcp_udp_ratio = traffic_features_vec[2],
                            .avg_packet_size = traffic_features_vec[3],
                            .port_entropy = traffic_features_vec[4],
                            .flow_duration_std = traffic_features_vec[5],
                            .src_ip_entropy = traffic_features_vec[6],
                            .dst_ip_concentration = traffic_features_vec[7],
                            .protocol_variety = traffic_features_vec[8],
                            .temporal_consistency = traffic_features_vec[9]
                        };

                        // Predict (target: <100μs)
                        auto traffic_result = traffic_detector_->predict(traffic_features);

                        logger_->debug("🤖 Traffic: class={} ({}), conf={:.4f}",
                                      traffic_result.class_id,
                                      (traffic_result.class_id == 0 ? "INTERNET" : "INTERNAL"),
                                      traffic_result.probability);

                        // Enriquecer evento
                        auto* level3_traffic_pred = ml_analysis->add_level3_specialized_predictions();
                        level3_traffic_pred->set_model_name("traffic_detector_embedded_cpp20");
                        level3_traffic_pred->set_model_version("1.0.0");
                        level3_traffic_pred->set_prediction_class(
                            traffic_result.class_id == 0 ? "INTERNET" : "INTERNAL"
                        );
                        level3_traffic_pred->set_confidence_score(traffic_result.probability);

                        // ========================================================
                        // LEVEL 3: INTERNAL TRAFFIC ANALYSIS (si es Internal)
                        // ========================================================
                        if (traffic_result.is_internal(config_.ml.thresholds.level3_web) &&
                            internal_detector_ && config_.ml.level3.internal.enabled) {

                            try {
                                logger_->debug("🔍 Running Level 3 Internal analysis...");

                                // Extract Internal features
                                std::vector<float> internal_features_vec;
                                try {
                                    internal_features_vec = extractor_->extract_level3_internal_features(nf);

                                    if (internal_features_vec.size() != 10) {
                                        logger_->error("❌ Internal feature extraction returned {} features, expected 10",
                                                      internal_features_vec.size());
                                        throw std::runtime_error("Invalid Internal feature count");
                                    }

                                } catch (const std::exception& e) {
                                    logger_->error("❌ Internal feature extraction failed: {}", e.what());
                                    std::lock_guard<std::mutex> lock(stats_mutex_);
                                    stats_.feature_extraction_errors++;
                                    throw;
                                }

                                // Convert to detector structure
                                ml_defender::InternalDetector::Features internal_features{
                                    .internal_connection_rate = internal_features_vec[0],
                                    .service_port_consistency = internal_features_vec[1],
                                    .protocol_regularity = internal_features_vec[2],
                                    .packet_size_consistency = internal_features_vec[3],
                                    .connection_duration_std = internal_features_vec[4],
                                    .lateral_movement_score = internal_features_vec[5],
                                    .service_discovery_patterns = internal_features_vec[6],
                                    .data_exfiltration_indicators = internal_features_vec[7],
                                    .temporal_anomaly_score = internal_features_vec[8],
                                    .access_pattern_entropy = internal_features_vec[9]
                                };

                                // Predict (target: <100μs)
                                auto internal_result = internal_detector_->predict(internal_features);

                                logger_->debug("🤖 Internal: class={} ({}), conf={:.4f}",
                                              internal_result.class_id,
                                              (internal_result.class_id == 0 ? "BENIGN" : "SUSPICIOUS"),
                                              internal_result.probability);

                                // Enriquecer evento
                                auto* level3_internal_pred = ml_analysis->add_level3_specialized_predictions();
                                level3_internal_pred->set_model_name("internal_detector_embedded_cpp20");
                                level3_internal_pred->set_model_version("1.0.0");
                                level3_internal_pred->set_prediction_class(
                                    internal_result.class_id == 0 ? "BENIGN" : "SUSPICIOUS"
                                );
                                level3_internal_pred->set_confidence_score(internal_result.suspicious_prob);

                                if (internal_result.is_suspicious(config_.ml.thresholds.level3_internal)) {
                                    event.set_threat_category("SUSPICIOUS_INTERNAL");
                                    ml_analysis->set_final_threat_classification("SUSPICIOUS_INTERNAL");

                                    logger_->warn("🔴 SUSPICIOUS INTERNAL ACTIVITY: event={}, "
                                                 "lateral_movement={:.3f}, exfiltration={:.3f}, conf={:.2f}%",
                                                 event.event_id(),
                                                 internal_features_vec[5],  // lateral_movement_score
                                                 internal_features_vec[7],  // data_exfiltration_indicators
                                                 internal_result.suspicious_prob * 100);
                                }

                            } catch (const std::exception& e) {
                                logger_->error("❌ Level 3 Internal processing failed: {}", e.what());
                                std::lock_guard<std::mutex> lock(stats_mutex_);
                                stats_.inference_errors++;
                            }
                        }

                    }

                } catch (const std::exception& e) {
                    logger_->error("❌ Level 3 Traffic processing failed: {}", e.what());
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.inference_errors++;
                }
            }

        } else {
            // BENIGN traffic
            event.set_threat_category("NORMAL");
        }

        // ====================================================================
        // SEND ENRICHED EVENT
        // ====================================================================
        send_enriched_event(event);

        // ====================================================================
        // STATS
        // ====================================================================
        auto end_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.events_processed++;
            stats_.avg_processing_time_ms =
                (stats_.avg_processing_time_ms * (stats_.events_processed - 1) + duration_ms) /
                static_cast<double>(stats_.events_processed);
        }

        if (label_l1 == 1) {
            logger_->info("🚨 ATTACK: event={}, L1_conf={:.2f}%, processing={:.2f}ms",
                         event.event_id(), confidence_l1 * 100, duration_ms);
        } else {
            logger_->debug("✅ BENIGN: event={}, confidence={:.2f}%, processing={:.2f}ms",
                          event.event_id(), confidence_l1 * 100, duration_ms);
        }

    } catch (const std::exception& e) {
        logger_->error("Failed to process event: {}", e.what());
    }
}

void ZMQHandler::send_enriched_event(const protobuf::NetworkSecurityEvent& event) {
    try {
        // ================================================================
        // 🎯 DAY 27: ENCRYPT & SEND
        // ================================================================

        // Serialize protobuf
        std::string serialized;
        if (!event.SerializeToString(&serialized)) {
            logger_->error("Failed to serialize enriched event {}", event.event_id());
            return;
        }

        // Compress → Encrypt (pattern from firewall-acl-agent)
        auto compressed = crypto_manager_->compress(serialized);
        auto encrypted = crypto_manager_->encrypt(compressed);

        logger_->trace("🔒 Encrypted: {} bytes → {} bytes (compressed+encrypted)",
                      serialized.size(), encrypted.size());

        // Send encrypted message
        zmq::message_t message(encrypted.size());
        memcpy(message.data(), encrypted.data(), encrypted.size());

        auto result = output_socket_->send(message, zmq::send_flags::dontwait);

        if (result) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.events_sent++;

            logger_->debug("📤 Event sent: id={}, encrypted_size={} bytes",
                          event.event_id(), encrypted.size());
        } else {
            logger_->warn("Failed to send event {} (queue full?)", event.event_id());
        }

    } catch (const zmq::error_t& e) {
        logger_->error("ZMQ send error: {}", e.what());
    } catch (const std::exception& e) {
        logger_->error("Encryption/send error: {}", e.what());
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

// ============================================================================
// 🎯 DAY 14: RAG Logger Helper Methods
// ============================================================================

void ZMQHandler::log_rag_statistics() {
    if (rag_logger_) {
        auto stats = rag_logger_->get_statistics();
        logger_->info("📊 RAG Stats: {} events logged, {} divergent, {} consensus",
                     stats["events_logged"],
                     stats["divergent_events"],
                     stats["consensus_events"]);
    }
}

uint64_t ZMQHandler::calculate_events_per_minute() {
    // Calculate uptime in seconds
    auto uptime_s = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now() - start_time_
    ).count();

    // Avoid division by zero
    if (uptime_s < 60) {
        return events_processed_total_;
    }

    // Return events per minute (average)
    return events_processed_total_ / (uptime_s / 60);
}

void ZMQHandler::log_periodic_stats() {
    double memory_mb = get_memory_usage_mb();
    auto stats = get_stats();

    logger_->info("📈 Periodic Stats - Memory: {:.1f} MB, Events: {} received, {} processed",
                  memory_mb, stats.events_received, stats.events_processed);
}

void ZMQHandler::periodic_health_check() {
    double memory_mb = get_memory_usage_mb();

    auto uptime_s = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now() - start_time_
    ).count();

    logger_->info("🧠 ML Detector Health - Memory: {:.1f} MB, Uptime: {}s",
                  memory_mb, uptime_s);

    // Alert if memory is too high
    if (memory_mb > 500.0) {  // 500 MB threshold
        logger_->warn("⚠️  High memory usage: {:.1f} MB", memory_mb);
    }
}

// ============================================================================
// Memory Monitoring
// ============================================================================

void ZMQHandler::start_memory_monitoring() {
    memory_monitor_running_ = true;
    memory_monitor_thread_ = std::thread(&ZMQHandler::memory_monitor_loop, this);
    logger_->info("📊 Memory monitoring started");
}

void ZMQHandler::stop_memory_monitoring() {
    memory_monitor_running_ = false;
    if (memory_monitor_thread_.joinable()) {
        memory_monitor_thread_.join();
    }
    logger_->info("📊 Memory monitoring stopped");
}

void ZMQHandler::memory_monitor_loop() {
    while (memory_monitor_running_) {
        // Read from /proc/self/statm (lightweight)
        FILE* file = fopen("/proc/self/statm", "r");
        if (file) {
            long pages = 0;
            if (fscanf(file, "%*d %ld", &pages) == 1) {
                long page_size = sysconf(_SC_PAGESIZE);
                current_memory_mb_.store((pages * page_size) / (1024.0 * 1024.0));
            }
            fclose(file);
        }

        // Sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

double ZMQHandler::get_memory_usage_mb() {
    return current_memory_mb_.load();
}

} // namespace ml_detector