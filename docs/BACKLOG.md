# ML Defender (aegisIDS) — BACKLOG
## Via Appia Quality 🏛️

---

## ✅ COMPLETADO

### Day 53 (9 Feb 2026) — HMAC Infrastructure
- etcd-server: SecretsManager + HTTP endpoints (/secrets/keys, /secrets/*, /secrets/rotate/*)
- etcd-client: get_hmac_key(), compute_hmac_sha256(), validate_hmac_sha256(), bytes_to_hex()/hex_to_bytes()
- Tests: 24 unit + 8 integration = 32/32 ✅
- Seguridad: 32-byte keys, constant-time validation, libsodium, key rotation con audit trail

### Day 64 (21 Feb 2026) — CSV Pipeline + Test Suite
- `tests/CMakeLists.txt` recreado — cubre unit/ e integration/
- `etcd_client.cpp` parcheado: `get_hmac_key()` dentro de `struct Impl`, usa API etcd-client
- CSV schema 127 columnas definido y documentado (FEATURE_SCHEMA.md)
- `test_csv_event_writer`: 127 cols, HMAC, rotación, zero-fill, concurrencia ✅
- `test_csv_feature_extraction`: contrato proto↔CSV, reproducibilidad ✅
- `test_etcd_client_hmac`: mock httplib, happy path + errores (pending include path)
- `ml-detector/CMakeLists.txt`: `add_subdirectory(tests)` — estructura limpia
- Decisiones de diseño: CSV sin cifrado, compresión gzip/lz4, retención configurable,
  pipeline L2 con anonimización offline previa al fine-tuning

---

## 🔄 EN CURSO / INMEDIATO

### Day 65 — Cierre CSV pipeline + verificación E2E

**1. Cerrar test_etcd_client_hmac** (30 min)
```cmake
target_include_directories(test_etcd_client_hmac PRIVATE /usr/local/include)
```

**2. Correr todos los tests CSV**
```bash
ctest -R "csv|hmac" -V
```

**3. Verificación E2E con inyector sintético**
- Confirmar que ml-detector produce CSV 127 cols
- ⚠️ CRÍTICO: verificar si S2 (NetworkFeatures) llega poblada o a cero
  → Si S2 = cero, el vector útil para FAISS es solo 40 features (S3)
  → Esto condiciona el diseño de CsvEventLoader

**4. Sistema completo sniffer→ml-detector con tráfico real**
- Comparar CSV vs JSONL del mismo período (mismos event_ids, campos S1, S2)

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
**Prerequisito:** verificación E2E Day 65 (confirmar que S2 llega poblado)

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
- [ ] Métricas: hmac_key_current_version, hmac_validation_success_previous,
  hmac_validation_failed_expired

### firewall-acl-agent — CSV pipeline
**Prioridad:** MEDIA
**Prerequisito:** Ruta A (rag-ingester CSV) validada con ml-detector

- [ ] Replicar CsvEventWriter de ml-detector para firewall-acl-agent
- [ ] Schema CSV propio (diferente al de ml-detector)
- [ ] HMAC + retención + compresión — misma arquitectura
- [ ] Tests equivalentes a los de ml-detector

### rag-local — comandos adicionales
**Prioridad:** MEDIA
**Estimación:** 2-3 días

- [ ] Generar informes PDF desde consultas RAG
- [ ] Geolocalización GeoIP de eventos concretos (integración MaxMind o similar)
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
**Componentes:** ml-detector + firewall-acl-agent

Pipeline de datos confirmado:
```
[ACUMULACIÓN]     →    [ANONIMIZACIÓN]    →    [ENTRENAMIENTO]
CSV comprimidos        proceso offline         dataset limpio
en archive/            - elimina IPs, MACs     para fine-tuning
sin transformar        - puertos sensibles      del LLM
                       - normaliza timestamps
                       (GDPR / ENS España)
```

- [ ] Acumulador: mover CSV a cold storage tras consumo L1 (CsvRetentionManager)
- [ ] Anonimizador offline: pipeline separado, configurable por normativa
    - Campos a anonimizar: src_ip, dst_ip, src_port (en algunos contextos), timestamps
    - Pseudonimización reversible vs. anonimización irreversible (decisión legal)
- [ ] Dataset builder: transforma CSV anonimizados en formato fine-tuning
- [ ] Pipeline de entrenamiento: compatible con LLM local (llama.cpp / vLLM)
- [ ] Origen de datos: ml-detector + firewall-acl-agent (ambos)
- **Principio:** raw primero, anonimizar después — nunca al revés

### ENT-2 — Watcher de Configuración en Runtime
**Prioridad:** ALTA enterprise
**Descripción:** Cambiar comportamiento del pipeline sin reiniciar

- [ ] Watcher sobre etcd-server que monitoriza cambios en los JSON de configuración
- [ ] Hot-reload de parámetros: umbrales ML, retención CSV, rutas de archivo
- [ ] Notificación a componentes activos via ZeroMQ o señal interna
- [ ] Validación del nuevo JSON antes de aplicar (rollback si inválido)
- [ ] Audit log de cambios de configuración (quién, cuándo, qué cambió)
- [ ] Tests: cambio en runtime sin pérdida de eventos en curso

### ENT-3 — SecureBusNode (Cifrado sin etcd)
**Prioridad:** ALTA enterprise
**Referencia:** `docs/enterprise/ML Defender Enterprise Security Module: SecureBusNode.md`
**Descripción:** Cifrado independiente de etcd-server y etcd-client

- [ ] Implementar SecureBusNode según especificación del documento
- [ ] Soporte para USB encrypted storage como origen de clave raíz
- [ ] Soporte para Hardware Security Modules (HSM)
- [ ] Detección y recuperación ante compromiso de clave raíz
- [ ] Compatible con el resto del pipeline (CSV, ZeroMQ, FAISS)
- [ ] Tests de seguridad: escenarios de compromiso + recuperación

### ENT-4 — rag-world (Telemetría Global Federada)
**Prioridad:** MEDIA enterprise
**Descripción:** Agregador global de rag-local distribuidos

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
- [ ] Protocolo de telemetría: qué datos se comparten (anonimizados siempre)
- [ ] Caché L2 global: beneficio compartido de lo que ha visto cada instalación
- [ ] Privacy-preserving: ninguna instalación expone datos sin anonimizar
- [ ] Federación opt-in: cada instalación decide si participar
- [ ] Dataset global para fine-tuning: más rico que cualquier dataset local
- [ ] Gobernanza: quién administra rag-world, SLA, retención global

---

## 📊 Estado global del proyecto

```
Foundation + Thread-Safety:       ████████████████████ 100% ✅
Contract Validation:              ████████████████████ 100% ✅
Build System:                     ████████████████████ 100% ✅
HMAC Infrastructure (F1+F2):      ████████████████████ 100% ✅
CSV Pipeline (ml-detector):       ████████████████░░░░  80% 🟡 (E2E pendiente)
Test Suite:                       ████████████████░░░░  80% 🟡 (httplib pending)
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
├─ FASE 3 HMAC:        ⏳ rag-ingester validation
└─ SecureBusNode:      ⏳ enterprise only
```

---

### ENT-5 — Integración Threat Intelligence (MISP)
**Prioridad:** ALTA enterprise
**Descripción:** Correlación de eventos con IOCs globales

- [ ] Integración con MISP (open source) via API REST
- [ ] Consulta automática por src_ip/dst_ip en cada evento MALICIOUS
- [ ] Si hay match: elevar nivel de confianza del evento + enriquecer con contexto MISP
- [ ] Cache local de IOCs (TTL configurable) para no saturar MISP en eventos de alto volumen
- [ ] Feeds prioritarios: CERTs europeos, listas negras de ransomware conocido
- [ ] Integración en el proto NetworkSecurityEvent: campo threat_intel_context
- [ ] Compatible con OpenCTI como alternativa a MISP (misma API STIX/TAXII)
- [ ] Community: GeoLite2 (MaxMind gratuito) para geolocalización básica
- [ ] Enterprise: GeoIP2 de pago — precisión + ASN + tipo de conexión
  (datacenter vs residencial cambia el análisis de un evento radicalmente)
- **Principio:** exportar hacia SIEMs comerciales (Splunk, QRadar) via syslog/API,
  nunca depender de ellos — contradice la filosofía de democratizar seguridad enterprise
- **Valor inmediato:** convierte el sistema de reactivo a proactivo —
  una IP en lista negra de 50 CERTs europeos pasa de "sospechoso" a "confirmado"

### ENT-6 — Observabilidad OpenTelemetry + Grafana
**Prioridad:** MEDIA enterprise
**Descripción:** Exportar métricas internas en formato estándar OTEL

- [ ] Exportar métricas del pipeline (sniffer→ml-detector→rag-ingester) en formato OTEL
- [ ] Permite integración en el stack de observabilidad existente del cliente enterprise
- [ ] Dashboards Grafana predefinidos (latencia, throughput, eventos/seg, HMAC failures)
- [ ] Alertas: anomalías en volumen de eventos, fallos de HMAC, rotación de claves
- [ ] No construir dashboards propios desde cero — Grafana ya existe y es el estándar

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución |
|----------|------------|
| CSV cifrado | ❌ No — sin cifrado, con HMAC por fila |
| CSV compresión | ✅ gzip (archivo) / lz4 (streaming caliente) |
| CSV retención | Configurable desde etcd, nunca borrar en producción |
| Raw vs anonimizado | Acumular raw, anonimizar offline antes de L2 |
| JSONL deprecación | Tras validar CSV E2E — desactivar para eliminar fuga de memoria |
| S2 NetworkFeatures | ⚠️ Verificar en E2E si llegan poblados (condiciona CsvEventLoader) |
| FASE 4 Grace Period | Preparar hmac.version en FASE 3 para facilitar migración |
| SecureBusNode | Enterprise only — ver doc specs |

---
*Última actualización: Day 64 — 21 Feb 2026*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic)*