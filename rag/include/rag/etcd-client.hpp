#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace rag {

    class EtcdClient {
    public:
        EtcdClient(const std::string& endpoints, int timeout_ms = 5000);
        ~EtcdClient();

        bool connect();
        void disconnect();

        // Operaciones básicas de etcd
        bool put(const std::string& key, const std::string& value);
        std::string get(const std::string& key);
        bool deleteKey(const std::string& key);

        // Watcher para cambios
        void watch(const std::string& key,
                   std::function<void(const std::string&, const std::string&)> callback);

        // Listado de keys
        std::vector<std::string> listKeys(const std::string& prefix);

    private:
        class Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace rag