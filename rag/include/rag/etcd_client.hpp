// include/rag/etcd_client.hpp
#pragma once
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace rag {

    class EtcdClient {
    public:
        EtcdClient(const std::string& endpoints, int timeout_ms = 5000);
        ~EtcdClient();

        bool connect();
        void disconnect();

        bool put(const std::string& key, const std::string& value);
        std::string get(const std::string& key);
        bool deleteKey(const std::string& key);

        void watch(const std::string& key,
                   std::function<void(const std::string&, const std::string&)> callback);

        std::vector<std::string> listKeys(const std::string& prefix);

    private:
        class Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace rag