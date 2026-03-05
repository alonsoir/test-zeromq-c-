# ML Defender (aRGUsEDR) — BACKLOG
## Via Appia Quality 🏛️

---

## ✅ COMPLETADO

### Day 76 (5 Mar 2026) — Proto3 Sentinel Fix + Pipeline Estable
- **SIGSEGV ByteSizeLong eliminado definitivamente**
  - Root cause: Proto3 C++ 3.21 no serializa submensajes donde todos los floats == 0.0f
  - Receptor recibe puntero null → SIGSEGV en ByteSizeLong()
  - 3 rutas afectadas en ring_consumer.cpp: populate_protobuf_event, send_fast_alert, send_ransomware_features
  - Fix: `init_embedded_sentinels()` helper — 40 campos, 4 submensajes, 0.5f sentinel
- **snappy::Uncompress() firma incorrecta** — corregido (2 args → 3: `.data(), .size()`)
- **libsnappy.pc symlink** — cmake pkg-config discovery reparado
- **Pipeline 6/6 estable**: ml-detector VIVO tras 60s+ de operación continua
- Regression tests: todos green ✅

### Day 75 (4 Mar 2026) — Root Cause SIGSEGV identificado
- GDB confirma: `RansomwareEmbeddedFeatures::_internal_io_intensity (this=0x0)`
- Fix parcial en send_ransomware_features (solo 3 de 10 campos por submensaje)
- Makefile: *-start desacoplado de compilación
- Tests regresión Proto3 sniffer: 3/3 ✅
- Tests regresión RAG Logger ml-detector: 3/3 ✅

### Day 72 (Feb 2026) — Deterministic trace_id correlation
- SHA256 hashing de identificadores de red + temporal buckets
- Sistema procesa eventos con zero coordinación entre componentes
- 36K+ eventos procesados, 0 errores crypto

### Day 64 (21 Feb 2026) — CSV Pipeline + Test Suite
- `tests/CMakeLists.txt` recreado — cubre unit/ e integration/
- CSV schema 127 columnas definido y documentado (FEATURE_SCHEMA.md)
- `test_csv_event_writer`: 127 cols, HMAC, rotación, zero-fill, concurrencia ✅
- `test_csv_feature_extraction`: contrato proto↔CSV, reproducibilidad ✅
- `ml-detector/CMakeLists.txt`: `add_subdirectory(tests)` — estructura limpia

### Day 53 (9 Feb 2026) — HMAC Infrastructure
- etcd-server: SecretsManager + HTTP endpoints (/secrets/keys, /secrets/*, /secrets/rotate/*)
- etcd-client: get_hmac_key(), compute_hmac_sha256(), validate_hmac_sha256(), bytes_to_hex()/hex_to_bytes()
- Tests: 24 unit + 8 integration = 32/32 ✅
- Seguridad: 32-byte keys, constant-time validation, libsodium, key rotation con audit trail

### Day 52 (8 Feb 2026) — Stress Testing + Config-Driven
- firewall-acl-agent: 364 events/sec, 54% CPU, 127MB RAM, 0 crypto errors @ 36K events
- Config-driven architecture completa (JSON is law)
- IPSet/IPTables kernel-level blocking validado

---

## 🔄 EN CURSO / INMEDIATO

### Day 77 — feature/ring-consumer-real-features

**Objetivo**: Reemplazar sentinels 0.5f con valores reales para validar F1-score

**Problema actual en populate_protobuf_event():**
```
1. ml_extractor_.populate_ml_defender_features(flow_stats, proto_event)  ← valores reales
2. run_ml_detection(proto_event)                                           ← infiere pero no escribe
3. init_embedded_sentinels(proto_event.mutable_network_features())        ← SOBRESCRIBE con 0.5f ❌
```

**Fix requerido:**
```cpp
// Llamar sentinels SOLO si el flow no existe (fallback)
if (!flow_stats_opt.has_value()) {
    init_embedded_sentinels(proto_event.mutable_network_features());
}
// Si flow existe, populate_ml_defender_features() ya llenó los campos reales
```

**Primer comando DAY 77:**
```bash
vagrant ssh -c "grep -n 'populate_ml_defender_features\|mutable_ddos\|mutable_ransomware_embedded\|mutable_traffic\|mutable_internal' /vagrant/sniffer/src/userspace/ml_defender_features.cpp | head -30"
```

**Trabajo completo DAY 77:**
- [ ] Verificar qué extrae realmente `populate_ml_defender_features()` del FlowStatistics
- [ ] Corregir orden de llamadas en `populate_protobuf_event()` — sentinels solo como fallback
- [ ] Completar `run_ml_detection()` — escribir resultados de inferencia al proto_event
- [ ] Verificar E2E: confirmar que los 40 campos llegan poblados con datos reales al ml-detector
- [ ] `make test-replay-neris` — validar F1-score contra CTU-13 (492K eventos)

**Ficheros clave:**
```
/vagrant/sniffer/src/userspace/ring_consumer.cpp
/vagrant/sniffer/src/userspace/ml_defender_features.cpp
/vagrant/sniffer/include/ml_defender_features.hpp
```

---

## 📋 BACKLOG — COMMUNITY

### FASE 3 — rag-ingester HMAC validation
**Prioridad:** ALTA — completa la infraestructura del Day 53
**Estimación:** 1-2 días

- [ ] EventLoader valida HMAC antes de descifrar (reject early, save CPU)
- [ ] Formato JSON forward-compatible con `hmac.version` (preparado para FASE 4)
- [ ] Métricas: hmac_validation_success/failed, tampering_attempts (atomic counter)
- [ ] Tests: 10+ escenarios (valid/invalid/tampered/expired)
- Ficheros: rag-ingester/src/event_loader.cpp, tests/test_hmac_validation.cpp

### CsvEventLoader — rag-ingester
**Prioridad:** ALTA — prerequisito para desactivar JSONL
**Estimación:** 2-3 días
**Prerequisito:** verificación E2E Day 77 (confirmar que S2 llega poblado con valores reales)

- [ ] Parsear 127 cols, verificar HMAC por fila, reconstruir vector 102 features (S2+S3)
- [ ] Descompresión gzip/lz4 antes de parsear
- [ ] Batch embedding hacia FAISS/SQLite
- [ ] Watcher de directorio: detecta nuevos CSV diarios automáticamente

### simple-embedder — adaptación CSV
**Prioridad:** ALTA
**Prerequisito:** CsvEventLoader funcionando

- [ ] Consumir CSV en lugar de JSONL como input
- [ ] Validar que rag-local puede consultar con datos CSV como origen
- [ ] Una vez validado: desactivar generación JSONL en ml-detector
  (elimina dependencia de librería json que causaba fugas de memoria)

### CsvRetentionManager
**Prioridad:** MEDIA
**Estimación:** 1 día

```cpp
struct CsvRetentionConfig {
    uint32_t compress_after_hours       = 1;    // gzip al rotar
    uint32_t move_to_archive_after_days = 7;    // tras indexar en FAISS
    std::string archive_path = "/data/ml-defender/archive/";
    bool delete_after_archive = false;          // NUNCA en producción
};
```
- archive_path configurable desde etcd
- Proceso de rotación: ACTIVO → L1-CONSUMIDO → L2-ARCHIVO

### FASE 4 — Grace Period + Key Versioning
**Prioridad:** MEDIA — producción hardening
**Estimación:** 2-3 días
**Prerequisito:** FASE 3 completa

- [ ] KeyVersion struct (version, key, timestamp, is_current)
- [ ] SecretsManager: deque<KeyVersion> por key path + pruning automático
- [ ] Endpoint: GET /secrets/*/versions
- [ ] Validador: prueba current → previous (dentro de grace period)
- [ ] Config: grace_period_seconds=86400, max_previous_keys=5

### firewall-acl-agent — CSV pipeline
**Prioridad:** MEDIA
**Prerequisito:** Ruta A (rag-ingester CSV) validada con ml-detector

- [ ] Replicar CsvEventWriter de ml-detector para firewall-acl-agent
- [ ] Schema CSV propio (diferente al de ml-detector)
- [ ] HMAC + retención + compresión — misma arquitectura

### rag-local — comandos adicionales
**Prioridad:** MEDIA
**Estimación:** 2-3 días

- [ ] Generar informes PDF desde consultas RAG
- [ ] Geolocalización GeoIP de eventos concretos
- [ ] Exportar resultados de consultas a CSV/JSON
- [ ] Historial de consultas con timestamps

### FASE 5 — Auto-Rotation de claves HMAC
**Prioridad:** BAJA
**Prerequisito:** FASE 4 completa

- [ ] Rotación automática programada
- [ ] Pre-rotation alerts
- [ ] Audit log (quién rotó, cuándo, por qué)
- [ ] Rollback capability

---

## 🏢 BACKLOG — ENTERPRISE

### ENT-1 — Entrenamiento Caché L2 (Fine-tuning LLM)
**Prioridad:** ALTA enterprise

Pipeline de datos confirmado:
```
[ACUMULACIÓN]     →    [ANONIMIZACIÓN]    →    [ENTRENAMIENTO]
CSV comprimidos        proceso offline         dataset limpio
en archive/            - elimina IPs, MACs     para fine-tuning
sin transformar        - puertos sensibles      del LLM
                       - normaliza timestamps
                       (GDPR / ENS España)
```

- [ ] Acumulador: mover CSV a cold storage tras consumo L1
- [ ] Anonimizador offline: pipeline separado, configurable por normativa
- [ ] Dataset builder: transforma CSV anonimizados en formato fine-tuning
- [ ] Pipeline de entrenamiento: compatible con LLM local (llama.cpp / vLLM)

### ENT-2 — Watcher de Configuración en Runtime
**Prioridad:** ALTA enterprise

- [ ] Watcher sobre etcd-server que monitoriza cambios en los JSON de configuración
- [ ] Hot-reload de parámetros: umbrales ML, retención CSV, rutas de archivo
- [ ] Notificación a componentes activos via ZeroMQ o señal interna
- [ ] Validación del nuevo JSON antes de aplicar (rollback si inválido)
- [ ] Audit log de cambios de configuración

### ENT-3 — SecureBusNode (Cifrado sin etcd)
**Prioridad:** ALTA enterprise

- [ ] Implementar SecureBusNode según especificación del documento
- [ ] Soporte para USB encrypted storage como origen de clave raíz
- [ ] Soporte para Hardware Security Modules (HSM)
- [ ] Detección y recuperación ante compromiso de clave raíz

### ENT-4 — rag-world (Telemetría Global Federada)
**Prioridad:** MEDIA enterprise

```
[Instalación A]    [Instalación B]    [Instalación C]
  rag-local  ──┐     rag-local  ──┤     rag-local  ──┐
               ↓                  ↓                  ↓
          ┌─────────────────────────────────────────┐
          │              rag-world                   │
          │   telemetría global + caché L2 global    │
          └─────────────────────────────────────────┘
```

- [ ] rag-world: agregador que habla con múltiples rag-local
- [ ] Privacy-preserving: ninguna instalación expone datos sin anonimizar
- [ ] Federación opt-in: cada instalación decide si participar

### ENT-5 — Integración Threat Intelligence (MISP)
**Prioridad:** ALTA enterprise

- [ ] Integración con MISP (open source) via API REST
- [ ] Consulta automática por src_ip/dst_ip en cada evento MALICIOUS
- [ ] Cache local de IOCs (TTL configurable)
- [ ] Feeds prioritarios: CERTs europeos, listas negras de ransomware conocido
- [ ] Compatible con OpenCTI como alternativa (misma API STIX/TAXII)
- [ ] Community: GeoLite2 (MaxMind gratuito) para geolocalización básica
- [ ] Enterprise: GeoIP2 de pago — precisión + ASN + tipo de conexión

### ENT-6 — Observabilidad OpenTelemetry + Grafana
**Prioridad:** MEDIA enterprise

- [ ] Exportar métricas del pipeline en formato OTEL
- [ ] Dashboards Grafana predefinidos (latencia, throughput, eventos/seg, HMAC failures)
- [ ] Alertas: anomalías en volumen de eventos, fallos de HMAC, rotación de claves

---

## 📊 Estado global del proyecto

```
Foundation + Thread-Safety:       ████████████████████ 100% ✅
Contract Validation:              ████████████████████ 100% ✅
Build System:                     ████████████████████ 100% ✅
HMAC Infrastructure (F1+F2):      ████████████████████ 100% ✅
Proto3 Pipeline Stability:        ████████████████████ 100% ✅  ← DAY 76
CSV Pipeline (ml-detector):       ████████████████░░░░  80% 🟡
Test Suite:                       ████████████████░░░░  80% 🟡
Ring Consumer Real Features:      ████░░░░░░░░░░░░░░░░  20% 🔴  ← DAY 77 target
F1-Score Validation (CTU-13):     ░░░░░░░░░░░░░░░░░░░░   0% ⏳  ← DAY 78 target
FASE 3 rag-ingester HMAC:         ░░░░░░░░░░░░░░░░░░░░   0% ⏳
CsvEventLoader rag-ingester:      ░░░░░░░░░░░░░░░░░░░░   0% ⏳
simple-embedder CSV:              ░░░░░░░░░░░░░░░░░░░░   0% ⏳
firewall-acl-agent CSV:           ░░░░░░░░░░░░░░░░░░░░   0% ⏳
rag-local (community):            ████░░░░░░░░░░░░░░░░  20% 🟡
rag-world (enterprise):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳

Pipeline Security:
├─ Crypto-Transport:   ✅ ChaCha20-Poly1305 + LZ4
├─ HMAC (F1+F2):       ✅ SHA256 key management
├─ CSV Integrity:      ✅ HMAC por fila en producción
├─ Proto3 Stability:   ✅ sentinel init — DAY 76
├─ FASE 3 HMAC:        ⏳ rag-ingester validation
└─ SecureBusNode:      ⏳ enterprise only
```

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución |
|----------|------------|
| CSV cifrado | ❌ No — sin cifrado, con HMAC por fila |
| CSV compresión | ✅ gzip (archivo) / lz4 (streaming caliente) |
| CSV retención | Configurable desde etcd, nunca borrar en producción |
| Raw vs anonimizado | Acumular raw, anonimizar offline antes de L2 |
| JSONL deprecación | Tras validar CSV E2E — desactivar para eliminar fuga de memoria |
| S2 NetworkFeatures | ⚠️ Verificar en DAY 77 con real features (sentinels 0.5f hasta entonces) |
| Proto3 submensajes | `init_embedded_sentinels()` como fallback, no como override — DAY 77 |
| FASE 4 Grace Period | Preparar hmac.version en FASE 3 para facilitar migración |
| SecureBusNode | Enterprise only — ver doc specs |
| run_ml_detection() | Incompleto — infiere pero no escribe al proto — completar DAY 77 |

---
*Última actualización: Day 76 — 5 Mar 2026*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic) GROK4, Gemini, Qwen, DeepSeek, ChatGPT*