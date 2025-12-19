#include "etcd_server/etcd_server.hpp"
#include "etcd_server/component_registry.hpp"
#include "etcd_server/compression_lz4.hpp"
#include "httplib.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>
// etcd-server/src/etcd_server.cpp
using json = nlohmann::json;

EtcdServer::EtcdServer(int port) : port_(port) {
    component_registry_ = std::make_unique<ComponentRegistry>();
}

EtcdServer::~EtcdServer() {
    stop();
}

bool EtcdServer::initialize() {
    std::cout << "[ETCD-SERVER] 🔧 Inicializando servidor en puerto " << port_ << std::endl;
    return true;
}

void EtcdServer::start() {
    if (running_) {
        std::cout << "[ETCD-SERVER] ⚠️  Servidor ya está ejecutándose" << std::endl;
        return;
    }

    running_ = true;
    server_thread_ = std::thread(&EtcdServer::run_server, this);
    std::cout << "[ETCD-SERVER] 🚀 Servidor iniciado" << std::endl;
}

void EtcdServer::stop() {
    if (!running_) return;

    running_ = false;
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    std::cout << "[ETCD-SERVER] 🛑 Servidor detenido" << std::endl;
}

bool EtcdServer::register_component(const std::string& component_name, const std::string& config_json) {
    return component_registry_->register_component(component_name, config_json);
}

std::string EtcdServer::get_component_config(const std::string& component_name) {
    return component_registry_->get_component_config(component_name);
}

bool EtcdServer::update_component_config(const std::string& component_name, const std::string& config_path, const std::string& value) {
    return component_registry_->update_component_config(component_name, config_path, value);
}

std::string EtcdServer::validate_configuration() {
    return component_registry_->validate_configuration();
}

void EtcdServer::run_server() {
    httplib::Server server;

    // Endpoint de registro de componentes
    server.Post("/register", [this](const httplib::Request& req, httplib::Response& res) {
        std::cout << "[ETCD-SERVER] 📝 POST /register recibido" << std::endl;

        try {
            auto json_body = json::parse(req.body);

            if (!json_body.contains("component") || !json_body["component"].is_string()) {
                res.status = 400;
                res.set_content(R"({"status": "error", "message": "Campo 'component' requerido"})", "application/json");
                return;
            }

            std::string component_name = json_body["component"];

            if (component_registry_->register_component(component_name, req.body)) {
                json response = {
                    {"status", "success"},
                    {"message", "Componente registrado correctamente"},
                    {"component", component_name},
                    {"encryption_key", component_registry_->get_encryption_key()}
                };
                res.set_content(response.dump(), "application/json");
            } else {
                res.status = 400;
                res.set_content(R"({"status": "error", "message": "Error en el registro"})", "application/json");
            }
        } catch (const std::exception& e) {
            res.status = 400;
            json error = {
                {"status", "error"},
                {"message", "JSON inválido"},
                {"details", e.what()}
            };
            res.set_content(error.dump(), "application/json");
        }
    });

    // Endpoint para desregistrar componentes - IMPLEMENTACIÓN REAL
    server.Post("/unregister", [this](const httplib::Request& req, httplib::Response& res) {
        std::cout << "[ETCD-SERVER] 📝 POST /unregister recibido" << std::endl;

        try {
            auto json_body = json::parse(req.body);

            if (!json_body.contains("component") || !json_body["component"].is_string()) {
                res.status = 400;
                res.set_content(R"({"status": "error", "message": "Campo 'component' requerido"})", "application/json");
                return;
            }

            std::string component_name = json_body["component"];

            // Usar el método real de desregistro del ComponentRegistry
            if (component_registry_->unregister_component(component_name)) {
                json response = {
                    {"status", "success"},
                    {"message", "Componente desregistrado correctamente"},
                    {"component", component_name},
                    {"remaining_components", component_registry_->get_component_count()}
                };
                res.set_content(response.dump(), "application/json");
                std::cout << "[ETCD-SERVER] ✅ Desregistro completado para: " << component_name << std::endl;
            } else {
                res.status = 404;
                json error = {
                    {"status", "error"},
                    {"message", "Componente no encontrado: " + component_name}
                };
                res.set_content(error.dump(), "application/json");
                std::cout << "[ETCD-SERVER] ❌ Componente no encontrado: " << component_name << std::endl;
            }

        } catch (const std::exception& e) {
            res.status = 400;
            json error = {
                {"status", "error"},
                {"message", "JSON inválido"},
                {"details", e.what()}
            };
            res.set_content(error.dump(), "application/json");
            std::cerr << "[ETCD-SERVER] ❌ Error en /unregister: " << e.what() << std::endl;
        }
    });

    // Endpoint para listar componentes registrados
    server.Get("/components", [this](const httplib::Request& /*req*/, httplib::Response& res) {
        std::cout << "[ETCD-SERVER] 📋 GET /components solicitado" << std::endl;

        auto components = component_registry_->get_registered_components();
        json response = {
            {"status", "success"},
            {"component_count", components.size()},
            {"components", components}
        };
        res.set_content(response.dump(), "application/json");
        std::cout << "[ETCD-SERVER] 📊 Listando " << components.size() << " componentes" << std::endl;
    });

    // Endpoint para obtener seed de cifrado
    server.Get("/seed", [this](const httplib::Request& /*req*/, httplib::Response& res) {
        std::cout << "[ETCD-SERVER] 🔑 GET /seed solicitado" << std::endl;

        std::string seed = component_registry_->get_encryption_seed();
        json response = {
            {"status", "success"},
            {"seed", seed}
        };
        res.set_content(response.dump(), "application/json");
    });

    // Endpoint de validación de configuración
    server.Get("/validate", [this](const httplib::Request& /*req*/, httplib::Response& res) {
        std::cout << "[ETCD-SERVER] 🔍 GET /validate solicitado" << std::endl;

        std::string validation = component_registry_->validate_configuration();
        res.set_content(validation, "application/json");
    });

    // Endpoint para obtener configuración de componente
    server.Get("/config/(.*)", [this](const httplib::Request& req, httplib::Response& res) {
        std::string component = req.matches[1];
        std::cout << "[ETCD-SERVER] 📋 GET /config/" << component << " solicitado" << std::endl;

        std::string config = component_registry_->get_component_config(component);
        if (config == "{}") {
            res.status = 404;
            json error = {
                {"status", "error"},
                {"message", "Componente no encontrado: " + component}
            };
            res.set_content(error.dump(), "application/json");
        } else {
            res.set_content(config, "application/json");
        }
    });

    // Endpoint para actualizar configuración
    server.Put("/config/(.*)", [this](const httplib::Request& req, httplib::Response& res) {
        std::string component = req.matches[1];
        std::cout << "[ETCD-SERVER] ✏️  PUT /config/" << component << " solicitado" << std::endl;

        try {
            auto json_body = json::parse(req.body);

            if (!json_body.contains("path") || !json_body.contains("value")) {
                res.status = 400;
                res.set_content(R"({"status": "error", "message": "Campos 'path' y 'value' requeridos"})", "application/json");
                return;
            }

            std::string path = json_body["path"];
            std::string value = json_body["value"];

            if (component_registry_->update_component_config(component, path, value)) {
                json response = {
                    {"status", "success"},
                    {"message", "Configuración actualizada"},
                    {"component", component},
                    {"path", path},
                    {"value", value}
                };
                res.set_content(response.dump(), "application/json");
            } else {
                res.status = 400;
                res.set_content(R"({"status": "error", "message": "Error actualizando configuración"})", "application/json");
            }
        } catch (const std::exception& e) {
            res.status = 400;
            json error = {
                {"status", "error"},
                {"message", "JSON inválido"},
                {"details", e.what()}
            };
            res.set_content(error.dump(), "application/json");
        }
    });
    
    // Endpoint v1 para subir configuración completa (cifrada/comprimida)
server.Put("/v1/config/(.*)", [this](const httplib::Request& req, httplib::Response& res) {
    std::string component = req.matches[1];
    std::cout << "[ETCD-SERVER] 📤 PUT /v1/config/" << component
              << " (" << req.body.size() << " bytes)" << std::endl;

    try {
        std::string processed_data = req.body;
        std::string content_type = req.get_header_value("Content-Type");

        // Step 1: Decrypt if encrypted (octet-stream indicates encryption)
        if (content_type == "application/octet-stream") {
            std::cout << "[ETCD-SERVER] 🔓 Descifrando datos..." << std::endl;
            processed_data = component_registry_->decrypt_data(processed_data);
            std::cout << "[ETCD-SERVER] ✅ Descifrado: " << processed_data.size() << " bytes" << std::endl;
        }

        // Step 2: Decompress if compressed (check size difference)
            std::string original_size_header = req.get_header_value("X-Original-Size");
            if (!original_size_header.empty()) {
                size_t original_size = std::stoull(original_size_header);

                // Only decompress if data was actually compressed (size < original)
                if (processed_data.size() < original_size) {
                    std::cout << "[ETCD-SERVER] 📦 Descomprimiendo datos (tamaño original: "
                              << original_size << " bytes)..." << std::endl;

                    processed_data = compression::decompress_lz4(processed_data, original_size);
                    std::cout << "[ETCD-SERVER] ✅ Descomprimido: " << processed_data.size() << " bytes" << std::endl;
                } else {
                    std::cout << "[ETCD-SERVER] ℹ️  Datos no comprimidos (tamaño: "
                              << processed_data.size() << " bytes)" << std::endl;
                }
            }

        // Step 3: Validate JSON
        auto parsed = json::parse(processed_data);

        // Step 4: Register component
        if (component_registry_->register_component(component, processed_data)) {
            json response = {
                {"status", "success"},
                {"component_id", component},
                {"size_bytes", processed_data.size()},
                {"timestamp", std::time(nullptr)}
            };
            res.set_content(response.dump(2), "application/json");
            std::cout << "[ETCD-SERVER] ✅ Config guardada para " << component << std::endl;
        } else {
            res.status = 500;
            json error = {
                {"status", "error"},
                {"message", "Error guardando configuración"}
            };
            res.set_content(error.dump(), "application/json");
        }

    } catch (const json::parse_error& e) {
        res.status = 400;
        json error = {
            {"status", "error"},
            {"error", "Invalid JSON"},
            {"details", e.what()}
        };
        res.set_content(error.dump(), "application/json");
    } catch (const std::exception& e) {
        res.status = 500;
        json error = {
            {"status", "error"},
            {"error", "Internal error"},
            {"details", e.what()}
        };
        res.set_content(error.dump(), "application/json");
    }
});

    // Endpoint de salud
    server.Get("/health", [](const httplib::Request& /*req*/, httplib::Response& res) {
        json response = {
            {"status", "healthy"},
            {"service", "etcd-server"},
            {"timestamp", time(nullptr)}
        };
        res.set_content(response.dump(), "application/json");
    });

    // Endpoint de información del sistema
    server.Get("/info", [this](const httplib::Request& /*req*/, httplib::Response& res) {
        json response = {
            {"status", "success"},
            {"service", "etcd-server"},
            {"version", "1.0.0"},
            {"components_registered", component_registry_->get_component_count()},
            {"port", port_}
        };
        res.set_content(response.dump(), "application/json");
    });

    std::cout << "[ETCD-SERVER] 🌐 Iniciando servidor HTTP en 0.0.0.0:" << port_ << std::endl;

    try {
        if (!server.listen("0.0.0.0", port_)) {
            std::cerr << "[ETCD-SERVER] ❌ Error iniciando servidor en puerto " << port_ << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ETCD-SERVER] 💥 Excepción en servidor: " << e.what() << std::endl;
    }

    running_ = false;
    std::cout << "[ETCD-SERVER] 📡 Servidor HTTP terminado" << std::endl;
}