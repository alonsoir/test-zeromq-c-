#include <zmq.hpp>
#include <jsoncpp/json/json.h>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <signal.h>
#include <fstream>
#include "config_manager.h"

class SnifferEventConsumer {
private:
    zmq::context_t context_;
    zmq::socket_t socket_;
    std::atomic<bool> running_;
    uint64_t events_processed_;
    std::chrono::steady_clock::time_point start_time_;
    ConfigManager config_;

public:
    SnifferEventConsumer(const std::string& config_file)
        : context_(1)
        , socket_(context_, ZMQ_PULL)
        , running_(true)
        , events_processed_(0)
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
                    processEvent(message);
                    events_processed_++;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }

                auto now = std::chrono::steady_clock::now();
                int stats_interval = config_.getStatsIntervalSeconds();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats).count() >= stats_interval) {
                    uint64_t current_count = events_processed_;
                    double rate = static_cast<double>(current_count - last_count) / stats_interval;

                    std::cout << "[Service3] Eventos procesados: " << current_count
                              << " | Rate: " << rate << " events/sec" << std::endl;

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
        std::cout << "[Service3] Tiempo total: " << duration.count() << " segundos" << std::endl;
        std::cout << "[Service3] Rate promedio: " << avg_rate << " events/sec" << std::endl;
    }

private:
    void processEvent(const zmq::message_t& message) {
        try {
            std::string data(static_cast<const char*>(message.data()), message.size());

            // Solo procesar JSON por ahora - versión simple para el consumer de apoyo
            Json::Value root;
            Json::CharReaderBuilder builder;
            std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
            std::string errors;

            if (reader->parse(data.c_str(), data.c_str() + data.length(), &root, &errors)) {
                processJsonEvent(root);
            } else {
                // Si no es JSON, asumir que es texto plano del sniffer
                if (config_.isVerboseMode() && events_processed_ % 100 == 0) {
                    std::cout << "[Service3] Evento de texto: " << data.substr(0, 100) << "..." << std::endl;
                }
            }

        } catch (const std::exception& e) {
            std::cerr << "[Service3] Error procesando evento: " << e.what() << std::endl;
        }
    }

    void processJsonEvent(const Json::Value& event) {
        std::string src_ip = event.get("src_ip", "unknown").asString();
        std::string dst_ip = event.get("dst_ip", "unknown").asString();
        int src_port = event.get("src_port", 0).asInt();
        int dst_port = event.get("dst_port", 0).asInt();
        std::string protocol = event.get("protocol", "unknown").asString();
        int packet_size = event.get("packet_size", 0).asInt();

        bool verbose = config_.isVerboseMode();
        if (verbose && events_processed_ % 100 == 0) {
            std::cout << "[Service3] JSON Event: "
                      << src_ip << ":" << src_port << " -> "
                      << dst_ip << ":" << dst_port
                      << " (" << protocol << ", " << packet_size << " bytes)" << std::endl;
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
    std::cout << "=== Service3: Sniffer Event Consumer (Simple Version) ===" << std::endl;
    std::cout << "[Service3] Iniciando consumidor de eventos del sniffer..." << std::endl;

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