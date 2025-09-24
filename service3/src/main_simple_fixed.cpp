#include <zmq.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <signal.h>
#include <fstream>
#include "config_manager.h"
#include "protobuf/network_security.pb.h"

class SnifferEventConsumer {
private:
    zmq::context_t context_;
    zmq::socket_t socket_;
    std::atomic<bool> running_;
    uint64_t events_processed_;
    uint64_t events_failed_;
    std::chrono::steady_clock::time_point start_time_;
    ConfigManager config_;

public:
    SnifferEventConsumer(const std::string& config_file)
        : context_(1)
        , socket_(context_, ZMQ_PULL)
        , running_(true)
        , events_processed_(0)
        , events_failed_(0)
        , start_time_(std::chrono::steady_clock::now())
        , config_(config_file)
    {
        if (!config_.loadConfig()) {
            throw std::runtime_error("Failed to load configuration");
        }

        socket_.close();
        int socket_type = getSocketTypeFromString(config_.getSocketType());
        socket_ = zmq::socket_t(context_, socket_type);

        socket_.set(zmq::sockopt::rcvhwm, config_.getReceiveHighWaterMark());
        socket_.set(zmq::sockopt::rcvtimeo, config_.getReceiveTimeout());

        std::string endpoint = config_.getSnifferEndpoint();
        std::string conn_type = config_.getConnectionType();

        if (conn_type == "connect") {
            socket_.connect(endpoint);
            std::cout << "[Service3] Conectado a " << endpoint;
        } else if (conn_type == "bind") {
            socket_.bind(endpoint);
            std::cout << "[Service3] Escuchando en " << endpoint;
        } else {
            throw std::runtime_error("Invalid connection_type: " + conn_type);
        }

        std::cout << " (socket: " << config_.getSocketType() << ")" << std::endl;
        std::cout << "[Service3] Node ID: " << config_.getNodeId() << std::endl;
        std::cout << "[Service3] Cluster: " << config_.getClusterName() << std::endl;
        std::cout << "[Service3] PROTOCOLO: SOLO PROTOBUF - Sin fallbacks JSON" << std::endl;
        std::cout << "[Service3] Iniciando consumo de eventos..." << std::endl;
    }

    ~SnifferEventConsumer() {
        socket_.close();
        context_.close();
    }

    void stop() {
        running_ = false;
    }

    void run() {
        zmq::message_t message;
        auto last_stats = std::chrono::steady_clock::now();
        uint64_t last_count = 0;

        while (running_) {
            try {
                zmq::recv_result_t result = socket_.recv(message, zmq::recv_flags::dontwait);

                if (result) {
                    if (processEvent(message)) {
                        events_processed_++;
                    } else {
                        events_failed_++;
                    }
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }

                auto now = std::chrono::steady_clock::now();
                int stats_interval = config_.getStatsIntervalSeconds();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats).count() >= stats_interval) {
                    uint64_t current_count = events_processed_;
                    double rate = static_cast<double>(current_count - last_count) / stats_interval;

                    std::cout << "[Service3] Eventos procesados: " << current_count
                              << " | Rate: " << rate << " events/sec";

                    if (events_failed_ > 0) {
                        std::cout << " | Fallos: " << events_failed_;
                    }

                    std::cout << std::endl;

                    last_stats = now;
                    last_count = current_count;
                }

            } catch (const zmq::error_t& e) {
                if (e.num() != EAGAIN) {
                    std::cerr << "[Service3] Error ZMQ: " << e.what() << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            } catch (const std::exception& e) {
                std::cerr << "[Service3] Error: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time_);
        double avg_rate = static_cast<double>(events_processed_) / duration.count();

        std::cout << "[Service3] Finalizando..." << std::endl;
        std::cout << "[Service3] Total procesados: " << events_processed_ << std::endl;
        std::cout << "[Service3] Total fallidos: " << events_failed_ << std::endl;
        std::cout << "[Service3] Tiempo total: " << duration.count() << " segundos" << std::endl;
        std::cout << "[Service3] Rate promedio: " << avg_rate << " events/sec" << std::endl;

        if (events_processed_ > 0) {
            double success_rate = (double)events_processed_ / (events_processed_ + events_failed_) * 100.0;
            std::cout << "[Service3] Tasa éxito protobuf: " << success_rate << "%" << std::endl;
        }
    }

private:
    bool processEvent(const zmq::message_t& message) {
        try {
            // Deserializar protobuf binario ÚNICAMENTE
            protobuf::NetworkSecurityEvent event;
            std::string binary_data(static_cast<const char*>(message.data()), message.size());

            if (!event.ParseFromString(binary_data)) {
                std::cerr << "[Service3] FALLO: Datos no son protobuf válido ("
                          << message.size() << " bytes recibidos)" << std::endl;
                return false;
            }

            // Procesar evento protobuf exitosamente deserializado
            processProtobufEvent(event);
            return true;

        } catch (const std::exception& e) {
            std::cerr << "[Service3] Error procesando protobuf: " << e.what() << std::endl;
            return false;
        }
    }

    void processProtobufEvent(const protobuf::NetworkSecurityEvent& event) {
        bool verbose = config_.isVerboseMode();

        if (verbose && events_processed_ % 100 == 0) {
            std::cout << "[Service3] Protobuf Event: ID=" << event.event_id();

            if (!event.originating_node_id().empty()) {
                std::cout << ", Node=" << event.originating_node_id();
            }

            if (!event.final_classification().empty()) {
                std::cout << ", Classification=" << event.final_classification();
            }

            std::cout << std::endl;

            // Mostrar network features si están disponibles
            if (event.has_network_features()) {
                const auto& features = event.network_features();
                std::cout << "[Service3] Network: "
                          << features.source_ip() << ":" << features.source_port()
                          << " -> " << features.destination_ip() << ":" << features.destination_port()
                          << " (" << features.protocol_name() << ")";

                if (features.total_forward_bytes() > 0) {
                    std::cout << ", " << features.total_forward_bytes() << " bytes";
                }

                std::cout << std::endl;
            }
        }
    }

    static int getSocketTypeFromString(const std::string& type) {
        if (type == "PULL") return ZMQ_PULL;
        if (type == "SUB") return ZMQ_SUB;
        if (type == "REP") return ZMQ_REP;
        if (type == "DEALER") return ZMQ_DEALER;
        if (type == "ROUTER") return ZMQ_ROUTER;

        std::cerr << "[Service3] Socket type desconocido: " << type << ", usando PULL" << std::endl;
        return ZMQ_PULL;
    }
};

std::unique_ptr<SnifferEventConsumer> consumer;

void signalHandler(int signal) {
    std::cout << "\n[Service3] Señal recibida (" << signal << "), cerrando..." << std::endl;
    if (consumer) {
        consumer->stop();
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== Service3: Sniffer Event Consumer (PROTOBUF ONLY) ===" << std::endl;
    std::cout << "[Service3] Iniciando consumidor de eventos del sniffer..." << std::endl;
    std::cout << "[Service3] ARQUITECTURA: ZeroMQ + Protobuf binario - Alto rendimiento" << std::endl;

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    std::string config_file = "config/service3.json";
    if (argc > 1) {
        config_file = argv[1];
    }

    try {
        consumer = std::make_unique<SnifferEventConsumer>(config_file);
        consumer->run();

    } catch (const std::exception& e) {
        std::cerr << "[Service3] Error fatal: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "[Service3] Terminado correctamente" << std::endl;
    return 0;
}