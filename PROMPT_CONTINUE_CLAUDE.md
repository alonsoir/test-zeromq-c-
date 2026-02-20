## Resumen Day 63 — Estado al cierre
Alonso: He aplicado lo mejor que he podido todos los cambios descritos en Ml detector patches day63
Habría que revisarlos. Necesitamos tests para ello. No se ha modificado nada tests/CMakelists.txt

Ml detector patches day63.md

// ================================================================
// PATCH 1: include/etcd_client.hpp
// Añadir get_hmac_key() a la interfaz PIMPL
// ================================================================
// En la clase EtcdClient, después de get_encryption_seed(), añadir:

        /**
         * @brief Get HMAC key for CSV integrity from etcd-server
         *
         * Calls GET /secrets/ml-detector — returns 64-char hex key (32 bytes).
         * Called once at startup, stored in CsvEventWriter.
         *
         * @return 64-char hex string, empty string on failure
         */
        std::string get_hmac_key() const;


// ================================================================
// PATCH 2: src/etcd_client.cpp
// Implementar get_hmac_key() en el Impl
// ================================================================
// El patrón es idéntico a get_encryption_seed() pero llama a /secrets/{component}.
// Añadir al final de etcd_client.cpp, fuera del namespace de Impl:

std::string EtcdClient::get_hmac_key() const {
return pImpl->get_hmac_key();
}

// Y dentro de struct Impl, añadir el método:
// (buscar donde está get_encryption_seed() en Impl y añadir después)

std::string get_hmac_key() {
// Path: /secrets/{short_component_name}
// Equivale a /secrets/ml-detector para component_name = "ml-detector"
std::string path = "/secrets/" + short_name_;  // mismo short_name_ que usa /register

    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(5);

    auto res = cli.Get(path.c_str());

    if (!res || res->status != 200) {
        std::cerr << "[etcd] Failed to get HMAC key from " << path
                  << " status=" << (res ? res->status : -1) << std::endl;
        return "";
    }

    try {
        auto j = nlohmann::json::parse(res->body);
        // etcd-server returns: {"key_hex": "...", "component": "...", ...}
        if (j.contains("key_hex")) {
            std::string key_hex = j["key_hex"].get<std::string>();
            std::cout << "[etcd] HMAC key received for " << path
                      << " (" << key_hex.size() << " chars)" << std::endl;
            return key_hex;
        }
        // Fallback: some versions return "key" directly
        if (j.contains("key")) {
            return j["key"].get<std::string>();
        }
        std::cerr << "[etcd] Unexpected HMAC key response format" << std::endl;
        return "";
    } catch (const std::exception& e) {
        std::cerr << "[etcd] Failed to parse HMAC key response: " << e.what() << std::endl;
        return "";
    }
}


// ================================================================
// PATCH 3: include/zmq_handler.hpp
// Añadir hmac_key_hex al constructor
// ================================================================
// En la declaración del constructor ZMQHandler, añadir último parámetro:

    ZMQHandler(
        const DetectorConfig& config,
        std::shared_ptr<ONNXModel> level1_model,
        std::shared_ptr<FeatureExtractor> extractor,
        std::shared_ptr<ml_defender::DDoSDetector> ddos_detector,
        std::shared_ptr<ml_defender::RansomwareDetector> ransomware_detector,
        std::shared_ptr<ml_defender::TrafficDetector> traffic_detector,
        std::shared_ptr<ml_defender::InternalDetector> internal_detector,
        std::shared_ptr<crypto::CryptoManager> crypto_manager,
        std::string hmac_key_hex = ""   // Day 63: CSV integrity key from etcd
    );


// ================================================================
// PATCH 4: src/zmq_handler.cpp
// ================================================================

// ── 4a: Añadir include al principio ─────────────────────────────
#include "csv_event_writer.hpp"
#include <filesystem>

// ── 4b: Añadir parámetro al constructor ─────────────────────────
// En la firma del constructor (línea ~17), añadir:
std::string hmac_key_hex  // Day 63
// En la lista de inicialización, añadir:
, hmac_key_hex_(std::move(hmac_key_hex))

// ── 4c: En el cuerpo del constructor, después de inicializar
//        rag_logger_ (línea ~115), añadir:

    // Day 63: Initialize CsvEventWriter if HMAC key available
    if (rag_logger_ && !hmac_key_hex_.empty()) {
        try {
            std::string csv_dir = "/vagrant/logs/ml-detector/events";
            std::filesystem::create_directories(csv_dir);

            ml_defender::CsvEventWriterConfig csv_cfg;
            csv_cfg.base_dir            = csv_dir;
            csv_cfg.hmac_key_hex        = hmac_key_hex_;
            csv_cfg.max_events_per_file = 10000;
            csv_cfg.min_score_threshold = 0.5f;

            auto csv_writer = std::make_unique<ml_defender::CsvEventWriter>(
                csv_cfg, logger_);

            rag_logger_->set_csv_writer(std::move(csv_writer));

            logger_->info("✅ CsvEventWriter initialized");
            logger_->info("   Output: {}/YYYY-MM-DD.csv", csv_dir);
            logger_->info("   Columns: {} (14 meta + 105 features + 1 hmac)",
                         ml_defender::CSV_TOTAL_COLS);

        } catch (const std::exception& e) {
            logger_->error("❌ Failed to initialize CsvEventWriter: {}", e.what());
            logger_->warn("⚠️  Continuing without CSV output");
        }
    } else if (hmac_key_hex_.empty()) {
        logger_->warn("⚠️  CsvEventWriter disabled — no HMAC key available");
    }

// ── 4d: Añadir hmac_key_hex_ como miembro privado en zmq_handler.hpp ──
std::string hmac_key_hex_;  // Day 63: stored for CsvEventWriter init


// ================================================================
// PATCH 5: src/main.cpp
// Obtener HMAC key y pasarla al ZMQHandler
// ================================================================
// Después del bloque donde se crea crypto_manager (~línea 150),
// justo antes de construir ZMQHandler, añadir:

        // Day 63: Get HMAC key for CSV integrity
        std::string hmac_key_hex;
        {
            std::string key = etcd_client->get_hmac_key();
            if (key.size() == 64) {
                hmac_key_hex = key;
                log->info("✅ [csv] HMAC key retrieved ({} chars)", key.size());
            } else {
                log->warn("⚠️  [csv] HMAC key not available — CSV output disabled");
                log->warn("   etcd-server SecretsManager may not have key for ml-detector");
            }
        }

// Y en la construcción de ZMQHandler, añadir el parámetro final:
ZMQHandler zmq_handler(
config,
model,
feature_extractor,
ddos_detector,
ransomware_detector,
traffic_detector,
internal_detector,
crypto_manager,
hmac_key_hex       // Day 63: CSV HMAC key
);

**Completado:**
- Diagnóstico completo del pipeline rag-ingester — `IngesterService` era stub, el pipeline real está en `main.cpp`, FAISS vacío por clave rotada
- Decisión arquitectural: dos CSVs independientes, dos embedders, dos índices FAISS
- `csv_event_writer.hpp/cpp` — writer completo, 120 columnas, HMAC-SHA256, rotación diaria
- `rag_logger.hpp` actualizado con `set_csv_writer()`
- 5 patches documentados para ml-detector (etcd_client, zmq_handler, main.cpp)
- Corrección identificada: `short_name_` → `component_name_` en Patch 2

**Pendiente Day 64:**
1. Aplicar los 5 patches en ml-detector
2. Compilar y verificar que arranca limpio
3. Tests: `test_csv_event_writer`, `test_etcd_client_hmac`, `test_csv_feature_extraction`
4. Verificar end-to-end: ml-detector produce CSV → `/vagrant/logs/ml-detector/events/YYYY-MM-DD.csv`
5. Diseñar `CsvEventLoader` en rag-ingester para consumir ese CSV

Piano piano 🏛️