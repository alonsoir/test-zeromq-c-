cd /vagrant && git add \
tools/synthetic_sniffer_injector.cpp \
ml-detector/include/config_loader.hpp \
ml-detector/include/zmq_handler.hpp \
ml-detector/src/zmq_handler.cpp \
ml-detector/src/rag_logger.cpp \
ml-detector/include/rag_logger.hpp \
rag-ingester/include/csv_file_watcher.hpp \
rag-ingester/src/csv_file_watcher.cpp \
rag-ingester/include/csv_event_loader.hpp \
rag-ingester/src/csv_event_loader.cpp \
rag-ingester/CMakeLists.txt \
rag-ingester/tests/CMakeLists.txt \
rag-ingester/tests/test_csv_file_watcher.cpp \
rag-ingester/tests/test_csv_event_loader.cpp \
rag-ingester/tests/test_file_watcher.cpp

git commit -m "feat(day67): ransomware injector, CSV writer fix, CsvFileWatcher+CsvEventLoader

Day 67 — Via Appia Quality

## Fixes
- zmq_handler: csv_writer_ unique_ptr→shared_ptr + set_csv_writer() call missing
  Root cause: rag_logger_ non-null blocked standalone CSV path, set_csv_writer()
  never called → CSV empty. Fix: shared ownership, explicit attach after RAG init.
- rag_logger: set_csv_writer() signature unique_ptr→shared_ptr
- test_file_watcher: setup/teardown per-test (leftover .pb files broke pattern test)

## Features
- tools: synthetic_sniffer_injector --ransomware mode
  Signature: entropy 7-8, io_intensity 0.85-1.0, file_operations 0.90-1.0
  ransomware20: rdp_failed_auth 10-50, smb_connection_diversity 20-50
  syn_without_ack_ratio ~0 (TCP completo, contrario al DDoS SYN flood)
  Destination ports: SMB(445)/RDP(3389) vs HTTPS(443) del DDoS
- rag-ingester: CsvFileWatcher — inotify IN_MODIFY + byte offset (tail semantics)
  Handles daily rotation via IN_CREATE on parent dir
  Independent inotify_fd from FileWatcher — no shared state
- rag-ingester: CsvEventLoader — 127-column CSV parser + HMAC-SHA256 verification
  Reuses Event+EngineVerdict structs from event_loader.hpp
  Sections: S1 metadata, S2 62-feature vector, S4 ML decisions + verdicts
  Stats: parsed_ok, hmac_failures, parse_errors, column_errors

## Verified
- CSV: 50 lines, 127 cols, HMAC intact (ransomware mode)
- JSONL: 100 lines valid JSON (RAG Logger)
- ctest 4/4: config_parser, file_watcher, csv_file_watcher, csv_event_loader

## Investigation
- etcd compression: channels already separated (ZMQ uses crypto_manager_ directly,
  etcd upload uses etcd_client Config.compression_enabled). No action required.
  Day 65/66 regression was isolated, Day 66 revert was correct.

## Backlog
- firewall-acl-agent: hardcoded values audit (same pattern as zmq_handler Day 66)
- rag-ingester: wire CsvFileWatcher+CsvEventLoader into IngesterService
- rag-ingester: SQLite WAL mode for streaming inserts
- synthetic_sniffer_injector: trace_id when available

AUTHORS: Alonso Isidoro Roman + Claude (Anthropic)"
```

---

Y el prompt de continuidad:

---
```
# Day 68 — Prompt de Continuidad
## ML Defender (aegisIDS) — Via Appia Quality

### Estado al cierre de Day 67

**Completado Day 67:**

Punto 1 — --ransomware en synthetic_sniffer_injector: ✅
- Firma: entropy 7-8, io_intensity 0.85-1.0, file_operations 0.90-1.0
- ransomware20: rdp_failed_auth 10-50, smb_connection_diversity 20-50
- syn_without_ack_ratio ~0 (TCP completo, opuesto al DDoS SYN flood)
- fast_detector_score=0.9f forzado, reason="synthetic_ransomware_forced"

Punto 2 — RAG Logger diagnóstico: ✅
- Problema resuelto como efecto del fix del shared_ptr (punto 3)
- 100 líneas JSONL válido verificadas

Punto 3 — CSV writer bug (shared_ptr + attach faltante): ✅
- csv_writer_ unique_ptr→shared_ptr en zmq_handler + rag_logger
- set_csv_writer() nunca se llamaba → rag_logger_ no-null bloqueaba path standalone
- 50 líneas, 127 cols, HMAC íntegro verificados con --ransomware

Punto 4 — etcd compresión: ✅ investigado, sin acción requerida
- ZMQ usa crypto_manager_ directamente (independiente de etcd Config)
- Errores LZ4 eran de diciembre 30, no actuales

Punto 5 — CsvFileWatcher + CsvEventLoader (4 ficheros nuevos): ✅
- CsvFileWatcher: inotify IN_MODIFY + byte offset (tail semantics)
  Rotación diaria via IN_CREATE en directorio padre
  inotify_fd independiente del FileWatcher existente
- CsvEventLoader: parser 127 columnas + HMAC-SHA256
  Reutiliza Event+EngineVerdict de event_loader.hpp
  S2 verificada: 57/62 cols no-cero en datos reales
- ctest 4/4: config_parser, file_watcher, csv_file_watcher, csv_event_loader

---

### Pendiente Day 68 — en orden

**1. Wire CsvFileWatcher+CsvEventLoader en IngesterService (objetivo principal)**

CsvFileWatcher y CsvEventLoader están implementados y testeados pero NO
conectados al IngesterService. El IngesterService.run() actual no los usa.

Diseño acordado:
- Modo streaming: CsvFileWatcher tail → CsvEventLoader.parse() → FAISS + SQLite
- SQLite WAL mode obligatorio: PRAGMA journal_mode=WAL al inicializar metadata_db
- Callback secuencial: un hilo, no concurrencia (CSV es append-only, orden garantizado)

Preguntas de diseño a resolver antes de implementar:
- ¿Config tiene ya un campo csv_source_path? Ver config_parser.hpp/cpp
- ¿IngesterService usa IngesterService::config_.* o lee directo del JSON?
- ¿MetadataDB tiene WAL mode activo? Ver metadata_db.cpp

Verificar antes de diseñar:
```bash
grep -n "csv\|wal\|WAL" /vagrant/rag-ingester/src/metadata_db.cpp | head -10
grep -n "csv\|source\|watch" /vagrant/rag-ingester/include/common/config_parser.hpp
grep -n "run\|watcher\|loader" /vagrant/rag-ingester/src/ingester_service.cpp
```

**2. firewall-acl-agent — hardcoded values (backlog)**

Mismo patrón que zmq_handler Day 66. Aplazar hasta que
CsvFileWatcher+IngesterService estén validados.

**3. synthetic_sniffer_injector — trace_id (backlog)**

Cuando trace_id esté disponible en el protobuf, añadirlo al inyector.
Beneficio: correlación sniffer→ml-detector→firewall en FAISS.

---

### Ficheros tocados Day 67
```
tools/synthetic_sniffer_injector.cpp
  - modo --ransomware (firma 3 capas: embedded, ransomware20, internal_anomaly)
  - inject_events() y main() actualizados con bool is_ransomware

ml-detector/include/zmq_handler.hpp
  - csv_writer_: unique_ptr → shared_ptr

ml-detector/src/zmq_handler.cpp
  - csv_writer_: make_unique → make_shared
  - set_csv_writer(csv_writer_) llamado explícitamente tras init RAG Logger

ml-detector/include/rag_logger.hpp
  - set_csv_writer(): unique_ptr → shared_ptr
  - csv_writer_: unique_ptr → shared_ptr

ml-detector/src/rag_logger.cpp
  - set_csv_writer(): unique_ptr → shared_ptr, std::move eliminado

rag-ingester/include/csv_file_watcher.hpp  [NUEVO]
rag-ingester/src/csv_file_watcher.cpp      [NUEVO]
rag-ingester/include/csv_event_loader.hpp  [NUEVO]
rag-ingester/src/csv_event_loader.cpp      [NUEVO]

rag-ingester/CMakeLists.txt
  - csv_file_watcher + csv_event_loader targets
  - OpenSSL::Crypto dependency
  - enable_testing() + add_subdirectory(tests) (tests movidos a tests/CMakeLists.txt)

rag-ingester/tests/CMakeLists.txt
  - test_csv_file_watcher + test_csv_event_loader targets
  - test_file_watcher movido aquí (eliminado del root)

rag-ingester/tests/test_csv_file_watcher.cpp  [NUEVO]
rag-ingester/tests/test_csv_event_loader.cpp  [NUEVO]
rag-ingester/tests/test_file_watcher.cpp
  - main() refactorizado: setup/teardown por test (fix assertion failure)
```