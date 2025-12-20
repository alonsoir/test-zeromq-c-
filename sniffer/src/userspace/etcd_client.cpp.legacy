// etcd Client Implementation for eBPF Sniffer
// Handles configuration, tokens, and service discovery

#include "etcd_client.hpp"
#include <curl/curl.h>
#include <json/json.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>

namespace sniffer {

// Callback for libcurl to write response data
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    size_t totalSize = size * nmemb;
    userp->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

EtcdClient::EtcdClient(const std::string& endpoint, int timeout_ms)
    : endpoint_(endpoint), timeout_ms_(timeout_ms), curl_handle_(nullptr) {

    // Initialize libcurl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_handle_ = curl_easy_init();

    if (!curl_handle_) {
        throw std::runtime_error("Failed to initialize libcurl");
    }

    // Set common curl options
    curl_easy_setopt(curl_handle_, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl_handle_, CURLOPT_TIMEOUT_MS, timeout_ms_);
    curl_easy_setopt(curl_handle_, CURLOPT_CONNECTTIMEOUT_MS, timeout_ms_ / 2);
    curl_easy_setopt(curl_handle_, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl_handle_, CURLOPT_FAILONERROR, 0L);  // Don't fail on HTTP error codes
}

EtcdClient::~EtcdClient() {
    // Stop watch thread
    watch_running_ = false;
    if (watch_thread_.joinable()) {
        watch_thread_.join();
    }

    // Cleanup curl
    if (curl_handle_) {
        curl_easy_cleanup(curl_handle_);
    }
    curl_global_cleanup();
}

bool EtcdClient::test_connection() {
    auto response = http_get("/version");
    bool success = (response.status_code == 200);
    connected_ = success;
    return success;
}

EtcdClient::EtcdResponse EtcdClient::get(const std::string& key) {
    std::string json_data = create_get_json(key);
    auto response = http_post("/v3/kv/range", json_data);

    requests_sent_++;
    if (response.status_code != 200) {
        requests_failed_++;
        EtcdResponse result;
        result.success = false;
        result.error = "HTTP " + std::to_string(response.status_code) + ": " + response.error;
        return result;
    }

    return parse_get_response(response.body);
}

EtcdClient::EtcdResponse EtcdClient::put(const std::string& key, const std::string& value) {
    std::string json_data = create_put_json(key, value);
    auto response = http_post("/v3/kv/put", json_data);

    requests_sent_++;
    if (response.status_code != 200) {
        requests_failed_++;
        EtcdResponse result;
        result.success = false;
        result.error = "HTTP " + std::to_string(response.status_code) + ": " + response.error;
        return result;
    }

    return parse_put_response(response.body);
}

EtcdClient::EtcdResponse EtcdClient::del(const std::string& key) {
    std::string json_data = create_delete_json(key);
    auto response = http_post("/v3/kv/deleterange", json_data);

    requests_sent_++;
    if (response.status_code != 200) {
        requests_failed_++;
        EtcdResponse result;
        result.success = false;
        result.error = "HTTP " + std::to_string(response.status_code) + ": " + response.error;
        return result;
    }

    EtcdResponse result;
    result.success = true;
    return result;
}

std::map<std::string, std::string> EtcdClient::get_prefix(const std::string& prefix) {
    std::string json_data = create_range_json(prefix);
    auto response = http_post("/v3/kv/range", json_data);

    requests_sent_++;
    if (response.status_code != 200) {
        requests_failed_++;
        return {};
    }

    return parse_range_response(response.body);
}

std::string EtcdClient::get_encryption_token(const std::string& service_name) {
    std::string token_key = "/security/tokens/" + service_name;
    auto response = get(token_key);

    if (response.success) {
        return response.value;
    }

    // Fallback: try global token
    auto global_response = get("/security/tokens/global");
    return global_response.success ? global_response.value : "";
}

bool EtcdClient::register_service(const std::string& service_name,
                                 const std::string& endpoint,
                                 int ttl_seconds) {
    // Create lease for service registration
    int64_t lease_id = create_lease(ttl_seconds);
    if (lease_id == 0) {
        return false;
    }

    // Store lease for renewal
    {
        std::lock_guard<std::mutex> lock(leases_mutex_);
        service_leases_[service_name] = lease_id;
    }

    // Register service with lease
    std::string service_key = "/services/heartbeat/" + service_name;
    return put_with_lease(service_key, endpoint, lease_id);
}

bool EtcdClient::renew_service(const std::string& service_name) {
    int64_t lease_id = 0;
    {
        std::lock_guard<std::mutex> lock(leases_mutex_);
        auto it = service_leases_.find(service_name);
        if (it == service_leases_.end()) {
            return false;
        }
        lease_id = it->second;
    }

    return renew_lease(lease_id);
}

std::string EtcdClient::get_sniffer_config(const std::string& config_key) {
    auto response = get(config_key);
    return response.success ? response.value : "";
}

bool EtcdClient::update_sniffer_config(const std::string& config_json,
                                      const std::string& config_key) {
    auto response = put(config_key, config_json);
    return response.success;
}

std::map<std::string, std::string> EtcdClient::get_compression_config() {
    return get_prefix("/sniffer/compression/");
}

bool EtcdClient::is_connected() const {
    return connected_.load();
}

std::map<std::string, int64_t> EtcdClient::get_stats() const {
    std::map<std::string, int64_t> stats;
    stats["requests_sent"] = requests_sent_.load();
    stats["requests_failed"] = requests_failed_.load();
    stats["watches_active"] = watches_active_.load();
    stats["connected"] = connected_.load() ? 1 : 0;
    return stats;
}

// HTTP helpers implementation
EtcdClient::HttpResponse EtcdClient::http_get(const std::string& path) {
    HttpResponse response;
    std::string url = endpoint_ + path;
    std::string response_body;

    curl_easy_setopt(curl_handle_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_handle_, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl_handle_, CURLOPT_HTTPGET, 1L);

    CURLcode res = curl_easy_perform(curl_handle_);

    if (res != CURLE_OK) {
        response.error = curl_easy_strerror(res);
        return response;
    }

    curl_easy_getinfo(curl_handle_, CURLINFO_RESPONSE_CODE, &response.status_code);
    response.body = response_body;

    return response;
}

EtcdClient::HttpResponse EtcdClient::http_post(const std::string& path, const std::string& data) {
    HttpResponse response;
    std::string url = endpoint_ + path;
    std::string response_body;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl_handle_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_handle_, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl_handle_, CURLOPT_POSTFIELDS, data.c_str());
    curl_easy_setopt(curl_handle_, CURLOPT_POSTFIELDSIZE, data.length());
    curl_easy_setopt(curl_handle_, CURLOPT_HTTPHEADER, headers);

    CURLcode res = curl_easy_perform(curl_handle_);

    curl_slist_free_all(headers);

    if (res != CURLE_OK) {
        response.error = curl_easy_strerror(res);
        return response;
    }

    curl_easy_getinfo(curl_handle_, CURLINFO_RESPONSE_CODE, &response.status_code);
    response.body = response_body;

    return response;
}

// JSON helpers implementation
std::string EtcdClient::create_get_json(const std::string& key) {
    Json::Value root;
    root["key"] = base64_encode(key);

    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, root);
}

std::string EtcdClient::create_put_json(const std::string& key, const std::string& value) {
    Json::Value root;
    root["key"] = base64_encode(key);
    root["value"] = base64_encode(value);

    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, root);
}

std::string EtcdClient::create_range_json(const std::string& prefix) {
    Json::Value root;
    root["key"] = base64_encode(prefix);
    root["range_end"] = base64_encode(prefix + "\xFF");  // Get all keys with prefix

    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, root);
}

EtcdClient::EtcdResponse EtcdClient::parse_get_response(const std::string& json) {
    EtcdResponse response;

    try {
        Json::Value root;
        Json::CharReaderBuilder builder;
        std::string errors;
        std::istringstream stream(json);

        if (!Json::parseFromStream(builder, stream, &root, &errors)) {
            response.error = "JSON parse error: " + errors;
            return response;
        }

        if (root.isMember("kvs") && root["kvs"].isArray() && !root["kvs"].empty()) {
            const Json::Value& kv = root["kvs"][0];
            if (kv.isMember("value")) {
                response.value = base64_decode(kv["value"].asString());
                response.success = true;

                if (kv.isMember("create_revision")) {
                    response.create_revision = kv["create_revision"].asInt64();
                }
                if (kv.isMember("mod_revision")) {
                    response.mod_revision = kv["mod_revision"].asInt64();
                }
                if (kv.isMember("version")) {
                    response.version = kv["version"].asInt64();
                }
            }
        } else {
            response.error = "Key not found";
        }
    } catch (const std::exception& e) {
        response.error = std::string("Parse error: ") + e.what();
    }

    return response;
}

EtcdClient::EtcdResponse EtcdClient::parse_put_response(const std::string& json) {
    EtcdResponse response;
    response.success = true;  // PUT typically succeeds if we get a 200 response

    try {
        Json::Value root;
        Json::CharReaderBuilder builder;
        std::string errors;
        std::istringstream stream(json);

        if (Json::parseFromStream(builder, stream, &root, &errors)) {
            if (root.isMember("header") && root["header"].isMember("revision")) {
                response.revision = root["header"]["revision"].asInt64();
            }
        }
    } catch (const std::exception& e) {
        // Log error but still consider PUT successful if we got here
        std::cerr << "PUT response parse warning: " << e.what() << std::endl;
    }

    return response;
}

std::map<std::string, std::string> EtcdClient::parse_range_response(const std::string& json) {
    std::map<std::string, std::string> result;

    try {
        Json::Value root;
        Json::CharReaderBuilder builder;
        std::string errors;
        std::istringstream stream(json);

        if (!Json::parseFromStream(builder, stream, &root, &errors)) {
            return result;
        }

        if (root.isMember("kvs") && root["kvs"].isArray()) {
            for (const auto& kv : root["kvs"]) {
                if (kv.isMember("key") && kv.isMember("value")) {
                    std::string key = base64_decode(kv["key"].asString());
                    std::string value = base64_decode(kv["value"].asString());
                    result[key] = value;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Range response parse error: " << e.what() << std::endl;
    }

    return result;
}

// Base64 helpers (simplified implementation)
std::string EtcdClient::base64_encode(const std::string& data) {
    static const char encoding_table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    size_t input_length = data.length();
    size_t output_length = 4 * ((input_length + 2) / 3);
    std::string encoded;
    encoded.reserve(output_length);

    for (size_t i = 0; i < input_length; i += 3) {
        uint32_t octet_a = i < input_length ? static_cast<unsigned char>(data[i]) : 0;
        uint32_t octet_b = i + 1 < input_length ? static_cast<unsigned char>(data[i + 1]) : 0;
        uint32_t octet_c = i + 2 < input_length ? static_cast<unsigned char>(data[i + 2]) : 0;

        uint32_t triple = (octet_a << 0x10) + (octet_b << 0x08) + octet_c;

        encoded.push_back(encoding_table[(triple >> 3 * 6) & 0x3F]);
        encoded.push_back(encoding_table[(triple >> 2 * 6) & 0x3F]);
        encoded.push_back(encoding_table[(triple >> 1 * 6) & 0x3F]);
        encoded.push_back(encoding_table[(triple >> 0 * 6) & 0x3F]);
    }

    int padding = input_length % 3;
    if (padding) {
        for (int i = 0; i < 3 - padding; i++) {
            encoded[encoded.length() - 1 - i] = '=';
        }
    }

    return encoded;
}

std::string EtcdClient::base64_decode(const std::string& data) {
    static const int decoding_table[128] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1
    };

    size_t input_length = data.length();
    if (input_length % 4 != 0) return "";

    size_t output_length = input_length / 4 * 3;
    if (data[input_length - 1] == '=') output_length--;
    if (data[input_length - 2] == '=') output_length--;

    std::string decoded;
    decoded.reserve(output_length);

    for (size_t i = 0; i < input_length; i += 4) {
        int a = decoding_table[static_cast<int>(data[i])];
        int b = decoding_table[static_cast<int>(data[i + 1])];
        int c = decoding_table[static_cast<int>(data[i + 2])];
        int d = decoding_table[static_cast<int>(data[i + 3])];

        if (a == -1 || b == -1 || c == -1 || d == -1) return "";

        uint32_t triple = (a << 3 * 6) + (b << 2 * 6) + (c << 1 * 6) + (d << 0 * 6);

        if (decoded.size() < output_length) decoded.push_back((triple >> 2 * 8) & 0xFF);
        if (decoded.size() < output_length) decoded.push_back((triple >> 1 * 8) & 0xFF);
        if (decoded.size() < output_length) decoded.push_back((triple >> 0 * 8) & 0xFF);
    }

    return decoded;
}

// Lease management helpers
int64_t EtcdClient::create_lease(int ttl_seconds) {
    Json::Value root;
    root["TTL"] = ttl_seconds;

    Json::StreamWriterBuilder builder;
    std::string json_data = Json::writeString(builder, root);

    auto response = http_post("/v3/lease/grant", json_data);

    if (response.status_code != 200) {
        return 0;
    }

    try {
        Json::Value result;
        Json::CharReaderBuilder reader_builder;
        std::string errors;
        std::istringstream stream(response.body);

        if (Json::parseFromStream(reader_builder, stream, &result, &errors)) {
            if (result.isMember("ID")) {
                return result["ID"].asInt64();
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Lease creation parse error: " << e.what() << std::endl;
    }

    return 0;
}

bool EtcdClient::put_with_lease(const std::string& key, const std::string& value, int64_t lease_id) {
    Json::Value root;
    root["key"] = base64_encode(key);
    root["value"] = base64_encode(value);
    root["lease"] = lease_id;

    Json::StreamWriterBuilder builder;
    std::string json_data = Json::writeString(builder, root);

    auto response = http_post("/v3/kv/put", json_data);
    return response.status_code == 200;
}

bool EtcdClient::renew_lease(int64_t lease_id) {
    Json::Value root;
    root["ID"] = lease_id;

    Json::StreamWriterBuilder builder;
    std::string json_data = Json::writeString(builder, root);

    auto response = http_post("/v3/lease/keepalive", json_data);
    return response.status_code == 200;
}

} // namespace sniffer