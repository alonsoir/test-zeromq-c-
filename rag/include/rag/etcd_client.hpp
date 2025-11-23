#pragma once
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace Rag {

    class EtcdClient {
    public:
        explicit EtcdClient(const std::string& endpoint);
        ~EtcdClient();

        bool put(const std::string& key, const std::string& value);
        std::pair<bool, std::string> get(const std::string& key);
        bool watch(const std::string& key, std::function<void(const std::string&)> callback);
        std::vector<std::string> listKeys(const std::string& prefix);

    private:
        class Impl;
        std::unique_ptr<Impl> pImpl;
    };

} // namespace Rag