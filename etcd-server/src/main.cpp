// etcd-server/src/main.cpp
#include "etcd_server/etcd_server.hpp"
#include "etcd_server/secrets_manager.hpp"
#include <iostream>
#include <csignal>

std::unique_ptr<EtcdServer> g_server;
std::unique_ptr<etcd::SecretsManager> g_secrets_manager;

void signal_handler(int signal) {
    std::cout << std::endl << "📡 Recibida señal " << signal << ", cerrando etcd-server..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

int main() {
    std::cout << "🚀 Iniciando etcd-server v0.2 con cpp-httplib + SecretsManager..." << std::endl;

    // Registrar manejadores de señales
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        // ═══════════════════════════════════════════════════════════
        // STEP 1: Initialize SecretsManager (Day 53)
        // ═══════════════════════════════════════════════════════════
        std::cout << std::endl;
        std::cout << "═══════════════════════════════════════════════════════" << std::endl;
        std::cout << "  Initializing SecretsManager (Day 53 - HMAC Support)" << std::endl;
        std::cout << "═══════════════════════════════════════════════════════" << std::endl;

        etcd::SecretsManager::Config secrets_config;
        secrets_config.enabled = true;
        secrets_config.default_key_length = 32;              // 256-bit keys
        secrets_config.rotation_interval_hours = 168;        // Weekly rotation
        secrets_config.auto_generate_on_startup = true;      // Auto-generate default keys

        g_secrets_manager = std::make_unique<etcd::SecretsManager>(secrets_config);

        if (!g_secrets_manager->initialize()) {
            std::cerr << "❌ Error inicializando SecretsManager" << std::endl;
            return 1;
        }

        std::cout << "✅ SecretsManager inicializado correctamente" << std::endl;

        // Display generated keys
        auto keys = g_secrets_manager->list_keys();
        std::cout << "📋 Generated " << keys.size() << " default key(s):" << std::endl;
        for (const auto& key_path : keys) {
            auto metadata = g_secrets_manager->get_key_metadata(key_path);
            if (metadata.has_value()) {
                std::cout << "   - " << key_path
                          << " (" << metadata->key_length << " bytes, "
                          << metadata->algorithm << ")" << std::endl;
            }
        }

        // ═══════════════════════════════════════════════════════════
        // STEP 2: Initialize EtcdServer
        // ═══════════════════════════════════════════════════════════
        std::cout << std::endl;
        std::cout << "═══════════════════════════════════════════════════════" << std::endl;
        std::cout << "  Initializing EtcdServer" << std::endl;
        std::cout << "═══════════════════════════════════════════════════════" << std::endl;

        g_server = std::make_unique<EtcdServer>(2379);

        // Connect SecretsManager to EtcdServer (Day 53)
        g_server->set_secrets_manager(g_secrets_manager.get());

        if (!g_server->initialize()) {
            std::cerr << "❌ Error inicializando etcd-server" << std::endl;
            return 1;
        }

        std::cout << "✅ etcd-server inicializado correctamente" << std::endl;
        std::cout << std::endl;

        // ═══════════════════════════════════════════════════════════
        // STEP 3: Display Available Endpoints
        // ═══════════════════════════════════════════════════════════
        std::cout << "🌐 Servidor HTTP escuchando en: http://0.0.0.0:2379" << std::endl;
        std::cout << "📚 Endpoints disponibles:" << std::endl;
        std::cout << "   POST /register      - Registrar componente" << std::endl;
        std::cout << "   POST /unregister    - Desregistrar componente" << std::endl;
        std::cout << "   GET  /components    - Listar componentes" << std::endl;
        std::cout << "   GET  /config/*      - Obtener configuración" << std::endl;
        std::cout << "   PUT  /config/*      - Actualizar configuración" << std::endl;
        std::cout << "   GET  /seed          - Obtener seed de cifrado" << std::endl;
        std::cout << "   GET  /validate      - Validar configuración global" << std::endl;
        std::cout << "   GET  /health        - Estado del servidor" << std::endl;
        std::cout << "   GET  /info          - Información del sistema" << std::endl;
        std::cout << std::endl;
        std::cout << "🔐 Secrets Endpoints (Day 53 - NEW):" << std::endl;
        std::cout << "   GET  /secrets/keys  - List all secret keys" << std::endl;
        std::cout << "   GET  /secrets/*     - Get specific key (hex-encoded)" << std::endl;
        std::cout << "   POST /secrets/rotate/* - Rotate specific key" << std::endl;
        std::cout << std::endl;

        // ═══════════════════════════════════════════════════════════
        // STEP 4: Start Server
        // ═══════════════════════════════════════════════════════════
        std::cout << "🚀 Starting HTTP server..." << std::endl;
        g_server->start();

        // Esperar a que termine
        while (g_server->is_running()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ Excepción fatal: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "👋 etcd-server terminado" << std::endl;
    return 0;
}