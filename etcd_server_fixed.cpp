// etcd-server/src/etcd_server.cpp
#include "etcd_server/etcd_server.hpp"
#include <iostream>
#include <thread>
#include <chrono>

using json = nlohmann::json;

EtcdServer::EtcdServer(const std::string& host, int port)
    : host_(host)
    , port_(port)
    , running_(false)
    , component_registry_(std::make_unique<ComponentRegistry>()) {
    std::cout << "🚀 EtcdServer inicializado en " << host_ << ":" << port_ << std::endl;
}

EtcdServer::~EtcdServer() {
    stop();
}

void EtcdServer::start() {
    if (running_) {
        std::cout << "⚠️  Servidor ya está corriendo" << std::endl;
        return;
    }
    
    running_ = true;
    std::cout << "🔌 Iniciando servidor HTTP..." << std::endl;
    
    // Inicia el servidor en un hilo separado
    server_thread_ = std::thread(&EtcdServer::run_server, this);
}

void EtcdServer::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    
    std::cout << "🛑 Servidor detenido" << std::endl;
}

void EtcdServer::wait() {
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

void EtcdServer::run_server() {
    httplib::Server server;
    
    std::cout << "🌐 Configurando endpoints..." << std::endl;
    
    // Endpoint de heartbeat
    server.Post("/heartbeat", [this](const httplib::Request& req, httplib::Response& res) {
        std::cout << "[ETCD-SERVER] 💓 Heartbeat recibido" << std::endl;
        res.set_content(R"({"status": "ok"})", "application/json");
    });
    
    // Endpoint de registro
    server.Post("/register", [this](const httplib::Request& req, httplib::Response& res) {
        std::cout << "[ETCD-SERVER] 📝 Solicitud de registro recibida" << std::endl;
        
        try {
            auto json_body = json::parse(req.body);
            
            if (!json_body.contains("component_name")) {
                res.status = 400;
                res.set_content(R"({"status": "error", "message": "Falta 'component_name'"})", "application/json");
                return;
            }
            
            std::string component_name = json_body["component_name"];
            
            // Registrar componente
            if (component_registry_->register_component(component_name, req.body)) {
                json response = {
                    {"status", "success"},
                    {"message", "Componente registrado"},
                    {"component", component_name},
                    {"encryption_seed", component_registry_->get_encryption_seed()}
                };
                res.set_content(response.dump(), "application/json");
                std::cout << "✅ Componente registrado: " << component_name << std::endl;
            } else {
                res.status = 400;
                res.set_content(R"({"status": "error", "message": "Error al registrar componente"})", "application/json");
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
    
    // Endpoint de desregistro
    server.Post("/unregister", [this](const httplib::Request& req, httplib::Response& res) {
        std::cout << "[ETCD-SERVER] 📤 Solicitud de desregistro recibida" << std::endl;
        
        try {
            auto json_body = json::parse(req.body);
            
            if (!json_body.contains("component_name")) {
                res.status = 400;
                res.set_content(R"({"status": "error", "message": "Falta 'component_name'"})", "application/json");
                return;
            }
            
            std::string component_name = json_body["component_name"];
            
            if (component_registry_->unregister_component(component_name)) {
                json response = {
                    {"status", "success"},
                    {"message", "Componente desregistrado"},
                    {"component", component_name}
                };
                res.set_content(response.dump(), "application/json");
                std::cout << "✅ Componente desregistrado: " << component_name << std::endl;
            } else {
                res.status = 404;
                json error = {
                    {"status", "error"},
                    {"message", "Componente no encontrado"}
                };
                res.set_content(error.dump(), "application/json");
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
    
    // Endpoint para listar componentes
    server.Get("/components", [this](const httplib::Request& /*req*/, httplib::Response& res) {
        std::cout << "[ETCD-SERVER] 📋 Listado de componentes solicitado" << std::endl;
        
        json response = {
            {"components", component_registry_->get_registered_components()},
            {"count", component_registry_->get_component_count()}
        };
        
        res.set_content(response.dump(), "application/json");
    });
    
    // Endpoint para obtener seed de cifrado
    server.Get("/seed", [this](const httplib::Request& /*req*/, httplib::Response& res) {
        json response = {
            {"seed", component_registry_->get_encryption_seed()}
        };
        res.set_content(response.dump(), "application/json");
    });
    
    // Endpoint para validar configuración
    server.Get("/validate", [this](const httplib::Request& /*req*/, httplib::Response& res) {
        std::string validation = component_registry_->validate_configuration();
        res.set_content(validation, "application/json");
    });
    
    // Endpoint para obtener configuración
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
    
    // Endpoint para actualizar configuración (path específico)
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
            std::string config_json;
            std::string content_type = req.get_header_value("Content-Type");
            
            if (content_type == "application/octet-stream") {
                // Datos cifrados - descifrar primero
                std::cout << "[ETCD-SERVER] 🔓 Descifrando datos..." << std::endl;
                config_json = component_registry_->decrypt_data(req.body);
                std::cout << "[ETCD-SERVER] ✅ Descifrado: " << config_json.size() << " bytes" << std::endl;
            } else {
                // JSON plano (Phase 1 MVP)
                config_json = req.body;
            }
            
            // Validar JSON
            auto parsed = json::parse(config_json);
            
            // Guardar usando register_component (actualiza si ya existe)
            if (component_registry_->register_component(component, config_json)) {
                json response = {
                    {"status", "success"},
                    {"component_id", component},
                    {"size_bytes", config_json.size()},
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
            {"timestamp", std::time(nullptr)}
        };
        res.set_content(response.dump(), "application/json");
    });
    
    // Endpoint de información
    server.Get("/info", [this](const httplib::Request& /*req*/, httplib::Response& res) {
        json response = {
            {"server", "etcd-server"},
            {"version", "1.0.0"},
            {"components_registered", component_registry_->get_component_count()}
        };
        res.set_content(response.dump(), "application/json");
    });
    
    std::cout << "✅ Endpoints configurados" << std::endl;
    std::cout << "🚀 Servidor escuchando en " << host_ << ":" << port_ << std::endl;
    
    // Iniciar servidor
    server.listen(host_.c_str(), port_);
}
