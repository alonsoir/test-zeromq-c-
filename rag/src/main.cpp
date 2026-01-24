#include "rag/whitelist_manager.hpp"
#include "rag/rag_command_manager.hpp"
#include "rag/config_manager.hpp"
#include "rag/llama_integration.hpp"
#include "embedders/embedder_factory.hpp"
#include "embedders/cached_embedder.hpp"  // ← AÑADIDO: Para dynamic_cast
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>  // read_index, write_index
#include <iostream>
#include <csignal>
#include <memory>

// ============================================================================
// GLOBALS
// ============================================================================
std::unique_ptr<LlamaIntegration> llama_integration;
std::unique_ptr<Rag::WhiteListManager> whitelist_manager;

// NUEVOS: Embedder + FAISS
std::unique_ptr<rag::IEmbedder> embedder;
std::unique_ptr<faiss::IndexFlatL2> chronos_index;
std::unique_ptr<faiss::IndexFlatL2> sbert_index;
std::unique_ptr<faiss::IndexFlatL2> attack_index;

// ============================================================================
// SIGNAL HANDLER
// ============================================================================
void signalHandler(int signal) {
    std::cout << "\n🛑 Señal " << signal << " recibida. Cerrando..." << std::endl;

    if (whitelist_manager) {
        whitelist_manager.reset();
    }
    if (llama_integration) {
        llama_integration.reset();
    }
    if (embedder) {
        embedder.reset();
    }

    // FAISS indices se destruyen automáticamente

    exit(0);
}

// ============================================================================
// HELPER: Test Embedder Command
// ============================================================================
void testEmbedder() {
    std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Testing Embedder System                                  ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

    if (!embedder) {
        std::cout << "❌ Embedder no inicializado" << std::endl;
        return;
    }

    // Generar features de prueba (105 dimensiones)
    std::vector<float> test_features(105);
    for (size_t i = 0; i < 105; ++i) {
        test_features[i] = static_cast<float>(i) / 105.0f;
    }

    std::cout << "\n📊 Generando embeddings para vector de prueba (105-d)..." << std::endl;

    // Generar embeddings
    auto chronos_emb = embedder->embed_chronos(test_features);
    auto sbert_emb = embedder->embed_sbert(test_features);
    auto attack_emb = embedder->embed_attack(test_features);

    std::cout << "   ✅ Chronos: " << chronos_emb.size() << " dims" << std::endl;
    std::cout << "   ✅ SBERT:   " << sbert_emb.size() << " dims" << std::endl;
    std::cout << "   ✅ Attack:  " << attack_emb.size() << " dims" << std::endl;

    // Añadir a índices FAISS
    std::cout << "\n💾 Añadiendo a índices FAISS..." << std::endl;

    chronos_index->add(1, chronos_emb.data());
    sbert_index->add(1, sbert_emb.data());
    attack_index->add(1, attack_emb.data());

    std::cout << "   ✅ Chronos index: " << chronos_index->ntotal << " vectors" << std::endl;
    std::cout << "   ✅ SBERT index:   " << sbert_index->ntotal << " vectors" << std::endl;
    std::cout << "   ✅ Attack index:  " << attack_index->ntotal << " vectors" << std::endl;

    // Test búsqueda
    std::cout << "\n🔍 Buscando vector similar (k=1)..." << std::endl;

    std::vector<float> distances(1);
    std::vector<faiss::idx_t> labels(1);

    chronos_index->search(1, chronos_emb.data(), 1, distances.data(), labels.data());

    std::cout << "   ✅ Nearest neighbor: index=" << labels[0]
              << ", distance=" << distances[0] << std::endl;

    // Cache stats (si es CachedEmbedder)
    auto* cached = dynamic_cast<rag::CachedEmbedder*>(embedder.get());
    if (cached) {
        auto stats = cached->cache_stats();
        std::cout << "\n📈 Cache Stats:" << std::endl;
        std::cout << "   Hits:     " << stats.hits << std::endl;
        std::cout << "   Misses:   " << stats.misses << std::endl;
        std::cout << "   Hit rate: " << stats.hit_rate << "%" << std::endl;
    }

    std::cout << "\n✅ Test completado exitosamente!" << std::endl;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "🚀 Iniciando RAG Security System - Arquitectura Centralizada" << std::endl;
    std::cout << "============================================================" << std::endl;

    // Configurar manejador de señales
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    try {
        // ====================================================================
        // 1. CARGAR CONFIGURACIÓN
        // ====================================================================
        auto& config_manager = Rag::ConfigManager::getInstance();
        if (!config_manager.loadFromFile("../config/rag-config.json")) {
            std::cerr << "❌ Error crítico: No se pudo cargar la configuración" << std::endl;
            return 1;
        }

        // ====================================================================
        // 2. INICIALIZAR LLAMA INTEGRATION
        // ====================================================================
        llama_integration = std::make_unique<LlamaIntegration>();
        auto rag_config = config_manager.getRagConfig();

        std::cout << "🤖 Intentando cargar modelo LLM: " << rag_config.model_name << std::endl;
        if (llama_integration->loadModel("../models/" + rag_config.model_name)) {
            std::cout << "✅ Modelo LLM cargado exitosamente" << std::endl;
        } else {
            std::cout << "❌ No se pudo cargar el modelo LLM" << std::endl;
        }

        // ====================================================================
        // 3. INICIALIZAR EMBEDDER SYSTEM (NUEVO)
        // ====================================================================
        std::cout << "\n🧮 Inicializando Embedder System..." << std::endl;

        // Crear config map desde JSON (simple por ahora)
        std::map<std::string, std::string> embedder_config = {
            {"cache_enabled", "true"},
            {"cache_ttl_seconds", "300"},
            {"cache_max_entries", "1000"}
        };

        embedder = rag::EmbedderFactory::create(
            rag::EmbedderFactory::Type::SIMPLE,
            embedder_config
        );

        std::cout << "✅ Embedder inicializado: " << embedder->name() << std::endl;
        auto [c, s, a] = embedder->dimensions();
        std::cout << "   Dimensiones: " << c << "/" << s << "/" << a << std::endl;
        std::cout << "   Efectividad: " << embedder->effectiveness_percent() << "%" << std::endl;

        // ====================================================================
// 4. LOAD FAISS INDICES FROM DISK (Producer creates them)
// ====================================================================
std::cout << "\n💾 Loading FAISS indices from disk..." << std::endl;

std::string indices_path = "/vagrant/shared/indices/";

std::unique_ptr<faiss::IndexFlatL2> chronos_index;
std::unique_ptr<faiss::IndexFlatL2> sbert_index;
std::unique_ptr<faiss::IndexFlatL2> attack_index;

try {
    chronos_index.reset(dynamic_cast<faiss::IndexFlatL2*>(
        faiss::read_index((indices_path + "chronos.faiss").c_str())
    ));

    sbert_index.reset(dynamic_cast<faiss::IndexFlatL2*>(
        faiss::read_index((indices_path + "sbert.faiss").c_str())
    ));

    attack_index.reset(dynamic_cast<faiss::IndexFlatL2*>(
        faiss::read_index((indices_path + "attack.faiss").c_str())
    ));

    std::cout << "✅ FAISS indices loaded:" << std::endl;
    std::cout << "   Chronos: " << chronos_index->ntotal << " vectors" << std::endl;
    std::cout << "   SBERT:   " << sbert_index->ntotal << " vectors" << std::endl;
    std::cout << "   Attack:  " << attack_index->ntotal << " vectors" << std::endl;

} catch (const std::exception& e) {
    std::cerr << "⚠️  Cannot load indices: " << e.what() << std::endl;
    std::cerr << "⚠️  Creating empty indices (wait for rag-ingester)" << std::endl;

    chronos_index = std::make_unique<faiss::IndexFlatL2>(128);
    sbert_index = std::make_unique<faiss::IndexFlatL2>(96);
    attack_index = std::make_unique<faiss::IndexFlatL2>(64);
}

// ====================================================================
// 5. LOAD METADATA DATABASE
// ====================================================================
std::cout << "\n📊 Loading metadata database..." << std::endl;

std::unique_ptr<ml_defender::MetadataReader> metadata;

try {
    metadata = std::make_unique<ml_defender::MetadataReader>(
        indices_path + "metadata.db"
    );

    size_t event_count = metadata->count();
    std::cout << "✅ Metadata loaded: " << event_count << " events" << std::endl;

} catch (const std::exception& e) {
    std::cerr << "⚠️  Cannot load metadata: " << e.what() << std::endl;
    std::cerr << "⚠️  query_similar will not be available" << std::endl;
}

        // ====================================================================
        // 6. INICIALIZAR WHITELIST MANAGER
        // ====================================================================
        whitelist_manager = std::make_unique<Rag::WhiteListManager>();

        // Registrar RagCommandManager
        auto rag_manager = std::make_shared<Rag::RagCommandManager>();
        // ========== Day 41: Register FAISS indices ==========
        if (chronos_index && sbert_index && attack_index) {
            rag_manager->setFAISSIndices(
                chronos_index.get(),
                sbert_index.get(),
                attack_index.get()
            );
        }

        if (metadata) {
            rag_manager->setMetadataReader(metadata.get());
        }
        // ===================================================
        whitelist_manager->registerCommandManager("rag", rag_manager);

        // Inicializar sistema
        if (!whitelist_manager->initialize()) {
            std::cerr << "❌ Error inicializando WhiteListManager" << std::endl;
            return 1;
        }

        // ====================================================================
        // 7. SISTEMA LISTO
        // ====================================================================
        std::cout << "\n✅ Sistema listo. Escribe 'help' para ver comandos disponibles." << std::endl;
        std::cout << "   Comandos especiales:" << std::endl;
        //std::cout << "   - test_embedder : Probar sistema de embeddings" << std::endl;
        std::cout << "   - exit/quit     : Salir del sistema" << std::endl;

        // ====================================================================
        // 8. BUCLE PRINCIPAL DE COMANDOS
        // ====================================================================
        std::string input;
        while (true) {
            std::cout << "\nSECURITY_SYSTEM> ";
            std::getline(std::cin, input);

            if (input.empty()) continue;

            // Comandos especiales
            if (input == "exit" || input == "quit") {
                break;
            }

            //if (input == "test_embedder") {
            //    testEmbedder();
            //    continue;
            //}

            // Comandos regulares (WhiteListManager)
            whitelist_manager->processCommand(input);
        }

        std::cout << "\n👋 Cerrando sistema..." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error crítico: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}