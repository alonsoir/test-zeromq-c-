#include "etcd_server/etcd_server.hpp"
#include <iostream>
#include <csignal>

std::unique_ptr<EtcdServer> g_server;

void signal_handler(int signal) {
    std::cout << std::endl << "📡 Recibida señal " << signal << ", cerrando etcd-server..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

int main() {
    std::cout << "🚀 Iniciando etcd-server v0.1 con cpp-httplib..." << std::endl;

    // Registrar manejadores de señales
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        g_server = std::make_unique<EtcdServer>(2379);

        if (!g_server->initialize()) {
            std::cerr << "❌ Error inicializando etcd-server" << std::endl;
            return 1;
        }

        std::cout << "✅ etcd-server inicializado correctamente" << std::endl;
        std::cout << "🌐 Servidor HTTP escuchando en: http://0.0.0.0:2379" << std::endl;
        std::cout << "📚 Endpoints disponibles:" << std::endl;
        std::cout << "   POST /register  - Registrar componente" << std::endl;
        std::cout << "   GET  /config    - Obtener configuración" << std::endl;
        std::cout << "   PUT  /config    - Actualizar configuración" << std::endl;
        std::cout << "   GET  /seed      - Obtener seed de cifrado" << std::endl;
        std::cout << "   GET  /validate  - Validar configuración global" << std::endl;

        // Iniciar servidor
        g_server->start();

        // Esperar a que termine
        while (g_server->is_running()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

    } catch (const std::exception& e) {
        std::cerr << "💥 Excepción en etcd-server: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "👋 etcd-server terminado correctamente" << std::endl;
    return 0;
}