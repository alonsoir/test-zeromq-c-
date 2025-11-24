#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>  // Necesario para std::function

namespace Rag {

    class EtcdClient {
    public:
        EtcdClient(const std::string& endpoint = "http://localhost:2379",
                   const std::string& component_name = "rag");
        ~EtcdClient();

        // Inicialización y conexión
        bool initialize();
        bool is_connected() const;

        // Gestión de componentes
        bool get_component_config(const std::string& component_name);
        bool validate_configuration();
        bool update_component_config(const std::string& component_name,
                                    const std::string& path,
                                    const std::string& value);

        // Cifrado
        std::string get_encryption_seed();
        bool test_encryption(const std::string& test_data = "test_message");

        // Pipeline
        bool get_pipeline_status();
        bool start_component(const std::string& component_name);
        bool stop_component(const std::string& component_name);

        // Comandos específicos del RAG
        bool show_rag_config();
        bool set_rag_setting(const std::string& setting, const std::string& value);
        bool get_rag_capabilities();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
        bool connected_ = false;
    };

} // namespace Rag