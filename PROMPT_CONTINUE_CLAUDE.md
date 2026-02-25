# Day 69 — Prompt de Continuidad
## ML Defender (aegisIDS) — Via Appia Quality

### Estado al cierre de Day 68

**Completado Day 68:**

Punto 1 — Wire CsvFileWatcher+CsvEventLoader en main.cpp: ✅
- config_parser.hpp/.cpp: csv_source_path + csv_hmac_key_hex opcionales (default "")
- main.cpp: CSV streaming path paralelo al path .pb existente
  CsvFileWatcher → CsvEventLoader → SimpleEmbedder(INPUT_DIM=105) → FAISS + MetadataDB
  Zero-pad S2 features 62→105: correcto matemáticamente para proyección aleatoria
  csv_loader declarado fuera del if para acceso en stats loop
  Stats en loop: lines_detected, parsed_ok, hmac_failures, parse_errors
  Shutdown limpio: csv_watcher->stop() junto a watcher.stop()
- rag-ingester.json: csv_source_path + csv_hmac_key_hex añadidos al bloque input
- ctest 4/4: config_parser, file_watcher, csv_file_watcher, csv_event_loader

Arquitectura Day 68 (ADR implícito):
- Path CSV completamente independiente del path .pb
- Comparten FAISS (entity_malicious) y MetadataDB — sin sincronización extra
  porque CsvFileWatcher es single-thread y FileWatcher también (append-only)
- IngesterService sigue siendo stub — refactor aplazado hasta integración
  firewall-acl-agent (3 fuentes = momento natural para abstraer)

---

### Pendiente Day 69 — en orden

**1. Smoke test end-to-end (objetivo principal)**

Verificar que el pipeline completo funciona:
synthetic_sniffer_injector --ransomware → ml-detector → CSV → rag-ingester → FAISS + MetadataDB

Secuencia:
```bash
# Terminal 1: lanzar rag-ingester
cd /vagrant/rag-ingester/build && ./rag-ingester ../config/rag-ingester.json

# Terminal 2: inyectar eventos ransomware
cd /vagrant/build && ./synthetic_sniffer_injector --ransomware --count 20

# Verificar MetadataDB
sqlite3 /vagrant/shared/indices/metadata.db \
  "SELECT event_id, classification, discrepancy_score FROM events ORDER BY rowid DESC LIMIT 10;"

# Verificar FAISS vectors
# (ntotal debe incrementar)
```

Criterios de éxito:
- rag-ingester log muestra "CSV event indexed" para eventos ransomware
- MetadataDB contiene filas con classification="ransomware"
- csv_watcher->lines_detected() > 0 en el stats loop
- 0 hmac_failures (si csv_hmac_key_hex está configurado)
  ó verify_hmac=false si la key está vacía

Posibles problemas a investigar:
- ¿csv_source_path apunta al mismo fichero que genera ml-detector?
  Verificar: grep -r "csv\|events.csv" /vagrant/ml-detector/src/zmq_handler.cpp
- ¿El fichero CSV existe antes de que arranque CsvFileWatcher?
  Si no existe, CsvFileWatcher lanza runtime_error — puede necesitar
  creación previa del fichero vacío o manejo de ENOENT en start()

**2. firewall-acl-agent — hardcoded values audit (backlog)**

Mismo patrón que zmq_handler Day 66.
Aplazar hasta que smoke test esté validado.

Verificar antes:
```bash
grep -rn "hardcod\|TODO\|FIXME\|magic\|\"127\.\|\"localhost\|port.*=.*[0-9]" \
  /vagrant/firewall-acl-agent/src/ | grep -v "\.pb\." | head -20
```

**3. synthetic_sniffer_injector — trace_id (backlog)**

Cuando trace_id esté disponible en el protobuf, añadirlo al inyector.
Beneficio: correlación sniffer→ml-detector→firewall en FAISS.

---

### Ficheros tocados Day 68
```
rag-ingester/include/common/config_parser.hpp
  - InputConfig: +csv_source_path, +csv_hmac_key_hex

rag-ingester/src/common/config_parser.cpp
  - parse csv_source_path + csv_hmac_key_hex con .value() (optional, default "")

rag-ingester/config/rag-ingester.json
  - "csv_source_path": "/vagrant/logs/ml-detector/events.csv"
  - "csv_hmac_key_hex": ""

rag-ingester/main.cpp
  - includes: csv_file_watcher.hpp + csv_event_loader.hpp
  - bloque Day 68: csv_loader + csv_watcher declarados, init condicional
  - callback lambda: parse → zero-pad → embed_attack → FAISS + MetadataDB
  - stats loop: lines_detected + parsed_ok/hmac_failures/parse_errors
  - shutdown: csv_watcher->stop()
```

### Contexto arquitectural acumulado
- SimpleEmbedder::INPUT_DIM = 105 (101 core + 4 embedded)
- MetadataDB::insert_event(faiss_idx, event_id, classification, discrepancy_score)
- MultiIndexManager: add_entity_malicious() para eventos de ataque
- WAL mode activo en MetadataDB desde Day 40
- CsvFileWatcher: inotify IN_MODIFY + byte offset, single background thread
- CsvEventLoader: parser 127 cols + HMAC-SHA256, stateless parse()
- IngesterService: stub — no se usa desde main.cpp todavía