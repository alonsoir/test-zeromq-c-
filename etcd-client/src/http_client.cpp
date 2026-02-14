// etcd-client/src/http_client.cpp
#include "etcd_client/etcd_client.hpp"
#include <httplib.h>
#include <string>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <chrono>

namespace etcd_client {
namespace http {

// ============================================================================
// HTTP Response struct is declared in etcd_client.hpp
// No need to redefine here
// ============================================================================

// Perform HTTP POST with retry logic
Response post(const std::string& host,
              int port,
              const std::string& path,
              const std::string& body,
              int timeout_seconds,
              int max_retries,
              int backoff_seconds) {

    Response response;

    for (int attempt = 0; attempt < max_retries; ++attempt) {
        try {
            // Create HTTP client
            httplib::Client cli(host, port);
            cli.set_connection_timeout(timeout_seconds);
            cli.set_read_timeout(timeout_seconds);
            cli.set_write_timeout(timeout_seconds);

            // Perform POST
            auto res = cli.Post(path, body, "application/json");

            if (res) {
                response.status_code = res->status;
                response.body = res->body;
                response.success = (res->status >= 200 && res->status < 300);

                if (response.success) {
                    return response;
                }

                // Server error, log and retry
                std::cerr << "⚠️  HTTP POST failed: " << res->status
                          << " - " << res->body << std::endl;
            } else {
                // Connection error
                std::cerr << "⚠️  HTTP POST connection failed to "
                          << host << ":" << port << path << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "⚠️  HTTP POST exception: " << e.what() << std::endl;
        }

        // Retry with backoff (except on last attempt)
        if (attempt < max_retries - 1) {
            std::cerr << "🔄 Retrying in " << backoff_seconds
                      << "s (attempt " << (attempt + 2) << "/" << max_retries << ")"
                      << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(backoff_seconds));
        }
    }

    // All retries failed
    response.success = false;
    return response;
}

// Perform HTTP GET with retry logic
Response get(const std::string& host,
             int port,
             const std::string& path,
             int timeout_seconds,
             int max_retries,
             int backoff_seconds) {

    Response response;

    for (int attempt = 0; attempt < max_retries; ++attempt) {
        try {
            httplib::Client cli(host, port);
            cli.set_connection_timeout(timeout_seconds);
            cli.set_read_timeout(timeout_seconds);

            auto res = cli.Get(path);

            if (res) {
                response.status_code = res->status;
                response.body = res->body;
                response.success = (res->status >= 200 && res->status < 300);

                if (response.success) {
                    return response;
                }

                std::cerr << "⚠️  HTTP GET failed: " << res->status
                          << " - " << res->body << std::endl;
            } else {
                std::cerr << "⚠️  HTTP GET connection failed to "
                          << host << ":" << port << path << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "⚠️  HTTP GET exception: " << e.what() << std::endl;
        }

        if (attempt < max_retries - 1) {
            std::cerr << "🔄 Retrying in " << backoff_seconds
                      << "s (attempt " << (attempt + 2) << "/" << max_retries << ")"
                      << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(backoff_seconds));
        }
    }

    response.success = false;
    return response;
}

// Perform HTTP PUT with retry logic
// Note: Default value for original_size is specified in the header
Response put(const std::string& host,
             int port,
             const std::string& path,
             const std::string& body,
             const std::string& content_type,
             int timeout_seconds,
             int max_retries,
             int backoff_seconds,
             size_t original_size) {

    Response response;

    for (int attempt = 0; attempt < max_retries; ++attempt) {
        try {
            httplib::Client cli(host, port);
            cli.set_connection_timeout(timeout_seconds);
            cli.set_read_timeout(timeout_seconds);
            cli.set_write_timeout(timeout_seconds);

            httplib::Headers headers = {
                {"Content-Type", content_type}
            };

            if (original_size > 0) {
                headers.insert({"X-Original-Size", std::to_string(original_size)});
            }

            auto res = cli.Put(path.c_str(), headers, body, content_type.c_str());

            if (res) {
                response.status_code = res->status;
                response.body = res->body;
                response.success = (res->status >= 200 && res->status < 300);

                if (response.success) {
                    return response;
                }

                std::cerr << "⚠️  HTTP PUT failed: " << res->status
                          << " - " << res->body << std::endl;
            } else {
                std::cerr << "⚠️  HTTP PUT connection failed to "
                          << host << ":" << port << path << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "⚠️  HTTP PUT exception: " << e.what() << std::endl;
        }

        if (attempt < max_retries - 1) {
            std::cerr << "🔄 Retrying in " << backoff_seconds
                      << "s (attempt " << (attempt + 2) << "/" << max_retries << ")"
                      << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(backoff_seconds));
        }
    }

    response.success = false;
    return response;
}

// Perform HTTP DELETE with retry logic
Response del(const std::string& host,
             int port,
             const std::string& path,
             int timeout_seconds,
             int max_retries,
             int backoff_seconds) {

    Response response;

    for (int attempt = 0; attempt < max_retries; ++attempt) {
        try {
            httplib::Client cli(host, port);
            cli.set_connection_timeout(timeout_seconds);
            cli.set_read_timeout(timeout_seconds);

            auto res = cli.Delete(path);

            if (res) {
                response.status_code = res->status;
                response.body = res->body;
                response.success = (res->status >= 200 && res->status < 300);

                if (response.success) {
                    return response;
                }

                std::cerr << "⚠️  HTTP DELETE failed: " << res->status
                          << " - " << res->body << std::endl;
            } else {
                std::cerr << "⚠️  HTTP DELETE connection failed to "
                          << host << ":" << port << path << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "⚠️  HTTP DELETE exception: " << e.what() << std::endl;
        }

        if (attempt < max_retries - 1) {
            std::cerr << "🔄 Retrying in " << backoff_seconds
                      << "s (attempt " << (attempt + 2) << "/" << max_retries << ")"
                      << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(backoff_seconds));
        }
    }

    response.success = false;
    return response;
}

} // namespace http
} // namespace etcd_client