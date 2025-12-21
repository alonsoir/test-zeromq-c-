#pragma once

#include <memory>
#include <string>

namespace ml_detector {

    /**
     * @brief PIMPL Adapter for etcd-client library
     *
     * Wraps etcd_client::EtcdClient to avoid exposing implementation details
     * and maintain zero breaking changes to main.cpp
     */
    class EtcdClient {
    public:
        /**
         * @brief Construct etcd client adapter
         * @param endpoint etcd server endpoint (e.g., "localhost:2379")
         * @param component_name Component identifier (e.g., "ml-detector")
         */
        EtcdClient(const std::string& endpoint, const std::string& component_name);

        ~EtcdClient();

        // Disable copy/move to simplify PIMPL
        EtcdClient(const EtcdClient&) = delete;
        EtcdClient& operator=(const EtcdClient&) = delete;

        /**
         * @brief Initialize connection and verify server availability
         * @return true if initialization successful
         */
        bool initialize();

        /**
         * @brief Register service with etcd-server and upload config
         * @return true if registration successful
         */
        bool registerService();

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

} // namespace ml_detector