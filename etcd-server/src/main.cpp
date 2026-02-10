// etcd-server/src/main.cpp
// Day 54: Compatible con namespace etcd_server::
//
// Co-authored-by: Claude (Anthropic)
// Co-authored-by: Alonso

#include "etcd_server/etcd_server.hpp"
#include "etcd_server/secrets_manager.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <csignal>

std::unique_ptr<EtcdServer> g_server;
std::shared_ptr<etcd_server::SecretsManager> g_secrets_manager;

void signal_handler(int signal) {
    std::cout << std::endl << "📡 Recibida señal " << signal << ", cerrando etcd-server..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

int main() {
    std::cout << "🚀 Iniciando etcd-server v0.3 - Day 54 Grace Period..." << std::endl;

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        // ═══════════════════════════════════════════════════════════
        // STEP 1: Initialize SecretsManager (Day 54)
        // ═══════════════════════════════════════════════════════════
        std::cout << std::endl;
        std::cout << "═══════════════════════════════════════════════════════" << std::endl;
        std::cout << "  Initializing SecretsManager (Day 54)" << std::endl;
        std::cout << "═══════════════════════════════════════════════════════" << std::endl;

        // Crear config JSON (hardcoded por ahora, Day 55: leer de archivo)
        nlohmann::json config = {
            {"secrets", {
                {"grace_period_seconds", 300},
                {"rotation_interval_hours", 168},
                {"default_key_length_bytes", 32}
            }}
        };

        g_secrets_manager = std::make_shared<etcd_server::SecretsManager>(config);

        std::cout << "✅ SecretsManager inicializado correctamente" << std::endl;
        std::cout << "   - Grace period: " << g_secrets_manager->get_grace_period_seconds() << "s" << std::endl;
        std::cout << "   - Namespace: etcd_server::" << std::endl;

        // ═══════════════════════════════════════════════════════════
        // STEP 2: Initialize EtcdServer
        // ═══════════════════════════════════════════════════════════
        std::cout << std::endl;
        std::cout << "═══════════════════════════════════════════════════════" << std::endl;
        std::cout << "  Initializing EtcdServer" << std::endl;
        std::cout << "═══════════════════════════════════════════════════════" << std::endl;

        g_server = std::make_unique<EtcdServer>(2379);

        // NOTA: set_secrets_manager espera etcd::SecretsManager*
        // Esto puede causar incompatibilidad de tipos
        // TODO Day 55: Actualizar EtcdServer para aceptar etcd_server::SecretsManager
        // Por ahora, comentamos esta línea para que compile
        // g_server->set_secrets_manager(g_secrets_manager.get());

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
        std::cout << "   GET  /seed          - Obtener seed de cifrado ChaCha20" << std::endl;
        std::cout << "   GET  /validate      - Validar configuración global" << std::endl;
        std::cout << "   GET  /health        - Estado del servidor" << std::endl;
        std::cout << "   GET  /info          - Información del sistema" << std::endl;
        std::cout << std::endl;
        std::cout << "🔐 Secrets Endpoints (Day 54 - NEW):" << std::endl;
        std::cout << "   (SecretsManager activo pero NO integrado aún con EtcdServer)" << std::endl;
        std::cout << "   TODO Day 55: Integrar etcd_server::SecretsManager con EtcdServer" << std::endl;
        std::cout << std::endl;

        // ═══════════════════════════════════════════════════════════
        // STEP 4: Start Server
        // ═══════════════════════════════════════════════════════════
        std::cout << "🚀 Starting HTTP server..." << std::endl;
        std::cout << "💡 ChaCha20 seed encryption: ACTIVE (EtcdServer)" << std::endl;
        std::cout << "💡 HMAC Grace Period: READY (etcd_server::SecretsManager)" << std::endl;
        std::cout << "⚠️  SecretsManager NO integrado aún (namespace mismatch)" << std::endl;
        std::cout << std::endl;

        g_server->start();

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