#pragma once

#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <unordered_map>

// Forward declaration for SecretsManager
namespace etcd_server {
    class SecretsManager;
}

//etcd-server/include/etcd_server/etcd_server.hpp
class ComponentRegistry;

class EtcdServer {
private:
    std::unique_ptr<ComponentRegistry> component_registry_;
    etcd_server::SecretsManager* secrets_manager_ = nullptr;  // Non-owning pointer
    std::atomic<bool> running_{false};
    int port_;
    std::thread server_thread_;

public:
    EtcdServer(int port = 2379);
    ~EtcdServer();

    void set_secrets_manager(etcd_server::SecretsManager* manager) {
        secrets_manager_ = manager;
    }
    bool initialize();
    void start();
    void stop();
    bool is_running() const { return running_; }

    // Gestión de componentes
    bool register_component(const std::string& component_name, const std::string& config_json);
    std::string get_component_config(const std::string& component_name);
    bool update_component_config(const std::string& component_name, const std::string& config_path, const std::string& value);

    // Validación
    std::string validate_configuration();

private:
    void run_server();
    std::string handle_request(const std::string& method, const std::string& path, const std::string& body);
};