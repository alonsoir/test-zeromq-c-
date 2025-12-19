#pragma once
#include <memory>
#include <string>
//rag/include/rag/etcd_client.hpp
namespace Rag {

class EtcdClient {
public:
    EtcdClient(const std::string& endpoint, const std::string& component_name);
    ~EtcdClient();

    bool initialize();
    bool is_connected() const;
    bool registerService();
	bool unregisterService();
    bool get_component_config(const std::string& component_name);
    bool validate_configuration();
    bool update_component_config(const std::string& component_name, const std::string& config);
    std::string get_encryption_seed();
    bool test_encryption(const std::string& test_data);
    bool get_pipeline_status();
    bool start_component(const std::string& component_name);
    bool stop_component(const std::string& component_name);
    bool show_rag_config();
    bool set_rag_setting(const std::string& setting, const std::string& value);
    bool get_rag_capabilities();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace Rag