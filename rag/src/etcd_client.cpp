// /vagrant/rag/src/etcd_client.cpp
#include "rag/etcd_client.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>

using json = nlohmann::json;

namespace Rag {

// Callback para escribir respuesta de curl
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
    size_t totalSize = size * nmemb;
    response->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

class EtcdClient::Impl {
private:
    std::string etcdEndpoint;
    CURL* curl;

public:
    Impl(const std::string& endpoint) : etcdEndpoint(endpoint) {
        curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to initialize CURL");
        }
    }

    ~Impl() {
        if (curl) {
            curl_easy_cleanup(curl);
        }
    }

    // QUITAR 'override' - estas no son funciones virtuales
    bool put(const std::string& key, const std::string& value) {
        std::string url = etcdEndpoint + "/v3/kv/put";

        json request = {
            {"key", key},
            {"value", value}
        };

        std::string response;
        return sendPostRequest(url, request.dump(), response);
    }

    // QUITAR 'override'
    std::pair<bool, std::string> get(const std::string& key) {
        std::string url = etcdEndpoint + "/v3/kv/range";

        json request = {
            {"key", key}
        };

        std::string response;
        if (!sendPostRequest(url, request.dump(), response)) {
            return {false, ""};
        }

        try {
            json jsonResponse = json::parse(response);
            if (jsonResponse.contains("kvs") && !jsonResponse["kvs"].empty()) {
                std::string value = jsonResponse["kvs"][0]["value"];
                return {true, value};
            }
        } catch (const json::exception& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
        }

        return {false, ""};
    }

    // QUITAR 'override'
    bool watch(const std::string& key, std::function<void(const std::string&)> callback) {
        // Implementación simple de watch usando polling
        std::thread([this, key, callback]() {
            std::string lastValue;
            while (true) {
                auto [success, value] = this->get(key);
                if (success && value != lastValue) {
                    callback(value);
                    lastValue = value;
                }
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }).detach();

        return true;
    }

    // QUITAR 'override'
    std::vector<std::string> listKeys(const std::string& prefix) {
        std::string url = etcdEndpoint + "/v3/kv/range";

        json request = {
            {"key", prefix},
            {"range_end", prefix + "z"}  // Simple range calculation
        };

        std::string response;
        std::vector<std::string> keys;

        if (!sendPostRequest(url, request.dump(), response)) {
            return keys;
        }

        try {
            json jsonResponse = json::parse(response);
            if (jsonResponse.contains("kvs")) {
                for (const auto& kv : jsonResponse["kvs"]) {
                    keys.push_back(kv["key"]);
                }
            }
        } catch (const json::exception& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
        }

        return keys;
    }

private:
    bool sendPostRequest(const std::string& url, const std::string& data, std::string& response) {
        if (!curl) return false;

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER,
            curl_slist_append(nullptr, "Content-Type: application/json"));

        CURLcode res = curl_easy_perform(curl);
        return (res == CURLE_OK);
    }
};

// Implementación de la clase pública
EtcdClient::EtcdClient(const std::string& endpoint)
    : pImpl(std::make_unique<Impl>(endpoint)) {}

EtcdClient::~EtcdClient() = default;

bool EtcdClient::put(const std::string& key, const std::string& value) {
    return pImpl->put(key, value);
}

std::pair<bool, std::string> EtcdClient::get(const std::string& key) {
    return pImpl->get(key);
}

bool EtcdClient::watch(const std::string& key, std::function<void(const std::string&)> callback) {
    return pImpl->watch(key, callback);
}

std::vector<std::string> EtcdClient::listKeys(const std::string& prefix) {
    return pImpl->listKeys(prefix);
}

} // namespace Rag