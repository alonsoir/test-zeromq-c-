# ML Defender (aRGUsEDR) — BACKLOG
## Via Appia Quality 🏛️

---

## 📐 Criterio de compleción (explícito para paper)

| Estado | Criterio |
|---|---|
| ✅ 100% | Implementado + probado en condiciones reales + resultado documentado |
| 🟡 80% | Implementado + compilando + smoke test pasado, sin validación E2E completa |
| 🟡 60% | Implementado parcialmente o con valores placeholder conocidos |
| ⏳ 0% | No iniciado |

Este criterio es intencionalmente conservador. Un componente que compila y pasa
sus unit tests pero no ha sido validado con tráfico real se considera 80%, no 100%.
La diferencia importa cuando se publican resultados.

---

## ✅ COMPLETADO

### Day 80 (9 Mar 2026) — Phase1-Day4-CRITICAL: Thresholds desde JSON ✅
- **JSON is the LAW — Phase1-Day4-CRITICAL CERRADO** tras 80 días de deuda técnica
- **4 capas de bug resueltas:**
  - `ring_consumer.cpp`: 8 literales hardcodeados → `config_.ml_defender.thresholds.*`
  - `main.cpp`: `ml_defender` nunca se mapeaba a `sniffer_config` → mapeo explícito
  - `config_types.h`: struct `ml_defender` añadido a `StrictSnifferConfig`
  - `config_types.cpp`: parser `ml_defender` desde `sniffer.json`
- **Evidencia:** `[ML Defender] Thresholds (JSON): DDoS=0.85 Ransomware=0.9 Traffic=0.8 Internal=0.85`
- **F1 post-thresholds:** F1=0.9934, Precision=0.9869, Recall=1.0000, FN=0 ✅
  - Mejora vs DAY 79 (0.9921): +0.0013 F1, +0.0025 Precision, FP absolutos 106→79
  - Nota: comparativa no limpia (distinto nº de eventos benignos entre replays)

### Day 79 (8 Mar 2026) — Sentinel Fix + Logging Standard + F1=0.9921
- **8× `return 0.5f` placeholder → `MISSING_FEATURE_SENTINEL`**
  - Funciones corregidas: geographical_concentration, io_intensity, resource_usage,
    file_operations, process_anomaly, temporal_pattern(else),
    behavior_consistency(iat_mean==0), packet_size_consistency(mean==0)
  - Distinción formal: sentinel fuera de dominio (-9999.0f) vs placeholder dentro
    de dominio vs valor semántico válido (0.5f TCP half-open)
  - Ver: docs/engineering_decisions/DAY79_sentinel_analysis.md
- **Logging estándar — deuda de 40 días liquidada**
  - 6 componentes escriben en /vagrant/logs/lab/ con nombre predecible
  - `make logs-all` + `make logs-lab-clean`
- **F1=0.9921 baseline CTU-13 Neris** ✅
  - Recall=1.0000 (FN=0), Precision=0.9844
  - TP=6676, FP=106, FN=0, TN=28, Total=6810 eventos
  - Ground truth: IP infectada 147.32.84.165

### Day 76 (5 Mar 2026) — Proto3 Sentinel Fix + Pipeline Estable
- **SIGSEGV ByteSizeLong eliminado definitivamente**
  - Root cause: Proto3 C++ 3.21 no serializa submensajes donde todos los floats == 0.0f
  - Fix: `init_embedded_sentinels()` helper — 40 campos, 4 submensajes
- **Pipeline 6/6 estable**: ml-detector VIVO tras 60s+ de operación continua

### Day 72 (Feb 2026) — Deterministic trace_id correlation
- SHA256 hashing de identificadores de red + temporal buckets
- 36K+ eventos procesados, 0 errores crypto
- ⚠️ 2 tests de trace_id fallan desde DAY 72 — preexistente, pendiente investigar

### Day 64 (21 Feb 2026) — CSV Pipeline + Test Suite
- CSV schema 127 columnas definido (FEATURE_SCHEMA.md)
- CsvEventWriter con HMAC por fila en ml-detector y firewall-acl-agent
- Tests unitarios: 127 cols, HMAC, rotación, zero-fill, concurrencia ✅
- ⚠️ Validación E2E con tráfico real pendiente → criterio 80%, no 100%

### Day 53 (9 Feb 2026) — HMAC Infrastructure
- SecretsManager + HTTP endpoints + key rotation
- Tests: 32/32 ✅

### Day 52 (8 Feb 2026) — Stress Testing + Config-Driven
- 364 events/sec, 54% CPU, 127MB RAM, 0 crypto errors @ 36K events

---

## 🔄 EN CURSO / INMEDIATO

### DAY 81 — FlowStatistics + F1 Comparativa Limpia + Dataset Balanceado

**P0 paper — Validación en tráfico balanceado**

CTU-13 Neris tiene 98% tráfico atacante. F1=0.9934 con Recall=1.0 no demuestra
comportamiento en tráfico mixto real. Gap científico señalado por todo el Consejo.

Datasets candidatos:
- CTU-13 otros escenarios (disponible ya, distinto ratio atacante/benigno)
- MAWI backbone (tráfico real de Internet, sin ataques conocidos)
- CICIDS2017 (Universidad de New Brunswick, ~7GB, referencia estándar literatura)
- UNSW-NB15 (UNSW Canberra, 9 categorías ataque + benigno real)

**P1 — Inspección FlowStatistics → features atacables**

```bash
grep -A 80 'struct FlowStatistics' sniffer/include/flow_manager.hpp
```

Features objetivo:
- `tcp_udp_ratio`: ¿hay `tcp_packets` y `udp_packets` separados?
- `protocol_variety`: ¿hay set de protocolos vistos?
- `flow_duration_std`: ¿hay timestamps inicio/fin por flujo?

Criterio: campo existe → implementar; no existe → SENTINEL + "requiere extensión FlowStatistics"

**P1 — Comparativa F1 limpia (mismo replay, ambas condiciones)**

Para el paper: mismo fichero PCAP, thresholds 0.85/0.90 vs 0.70/0.75.
El replay del DAY 80 tuvo distinto nº de eventos benignos (95 vs 134).

**Criterio de merge a main:**
- [ ] FlowStatistics inspeccionada y decisión documentada por feature
- [ ] F1 comparativa limpia documentada (mismo replay)
- [ ] Al menos 1 dataset balanceado validado (o protocolo de validación definido)
- [ ] Pipeline 6/6 RUNNING post-merge
- [ ] F1 ≥ 0.99 reproducible con `make test-replay-neris`

---

## 📋 BACKLOG — COMMUNITY

### Validación con datasets balanceados (post-merge DAY 81)
**Prioridad:** P0 paper — prerequisito para submission arXiv

CTU-13 Neris tiene 98% tráfico atacante. F1=0.9921/0.9934 con Recall=1.0 no
demuestra comportamiento en tráfico mixto real. Necesario para el paper.

Un clasificador dummy que dijera "todo es MALICIOUS" obtendría F1~0.99 en Neris.
Los reviewers de NDSS/RAID/USENIX lo saben y lo preguntarán.

Datasets candidatos:
- **CIC-IDS2017** — tráfico mixto balanceado, referencia estándar en literatura
- **UNSW-NB15** — 9 categorías de ataque + benigno real
- **MAWI Working Group** — tráfico real de backbone Internet, sin ataques
- **CTU-13 otros escenarios** — disponible ya, distinto ratio

Para el paper: tabla comparativa CTU-13 vs ≥1 dataset balanceado.

### CSV Pipeline E2E — validación con tráfico real
**Prioridad:** ALTA
**Estado actual:** 80% (implementado, compilando, unit tests ✅, E2E pendiente)

- [ ] Ejecutar `make test-replay-neris` y confirmar CSV generados correctamente
- [ ] Verificar HMAC por fila en CSV con tráfico real
- [ ] Confirmar trace_id une eventos ml-detector y firewall-acl-agent
- [ ] Investigar y resolver 2 fallos preexistentes test_trace_id (DAY 72)
- [ ] Verificar que rag-ingester consume CSV generados sin errores

**Criterio de compleción:** CSV generados, HMAC validado, trace_id correlacionado → 100% ✅

### FASE 3 — rag-ingester HMAC validation
**Prioridad:** ALTA
- [ ] EventLoader valida HMAC antes de descifrar
- [ ] Métricas: hmac_validation_success/failed, tampering_attempts
- [ ] Tests: 10+ escenarios

### CsvEventLoader — rag-ingester
**Prioridad:** ALTA
**Prerequisito:** CSV Pipeline E2E validado
- [ ] Parsear 127 cols, verificar HMAC, reconstruir vector 102 features
- [ ] Batch embedding hacia FAISS/SQLite
- [ ] Watcher de directorio: detecta nuevos CSV diarios

### simple-embedder — adaptación CSV
**Prioridad:** ALTA
**Prerequisito:** CsvEventLoader funcionando
- [ ] Consumir CSV en lugar de JSONL
- [ ] Una vez validado: desactivar JSONL (elimina fuga de memoria)

### CsvRetentionManager
**Prioridad:** MEDIA
- Rotación: ACTIVO → L1-CONSUMIDO → L2-ARCHIVO
- archive_path configurable desde etcd

### Unificar logs ml-detector (ADR pendiente)
**Prioridad:** MEDIA
Actualmente coexisten `detector.log` (spdlog interno) y `ml-detector.log`
(stdout Makefile). ADR: mover `log_file` al JSON de configuración de cada
componente. Hasta entonces Makefile es fuente de verdad sobre rutas de log.

### FASE 4 — Grace Period + Key Versioning
**Prioridad:** MEDIA
**Prerequisito:** FASE 3 completa
- [ ] KeyVersion struct + deque por key path
- [ ] Validador: current → previous dentro de grace period

### FASE 5 — Auto-Rotation de claves HMAC
**Prioridad:** BAJA
**Prerequisito:** FASE 4 completa
- [ ] Rotación automática programada + audit log + rollback

### rag-local — comandos adicionales
**Prioridad:** MEDIA
- [ ] Informes PDF desde consultas RAG
- [ ] Geolocalización GeoIP post-mortem
- [ ] Historial de consultas con timestamps

---

## 🏢 BACKLOG — ENTERPRISE

### ENT-1 — Federated Threat Intelligence (Inmunidad de Red)
**Prioridad:** ALTA enterprise

Arquitectura propuesta: anonimización local → contribución opt-in → reentrenamiento
federado → distribución como actualización binaria.

Referencias:
- McMahan et al., "Communication-Efficient Learning of Deep Networks
  from Decentralized Data" (FedAvg, AISTATS 2017)
- Nguyen et al., "Federated Learning for Intrusion Detection System",
  Computer Networks 2022

### ENT-2 — Attack Graph Generation (SOC Integration)
**Prioridad:** ALTA enterprise

GraphML + STIX 2.1 + streaming SOC/CAI/MITRE ATT&CK.
Hito mínimo viable: GraphML estático desde `make test-replay-neris` en Gephi.

### ENT-3 — P2P Seed Distribution via Protobuf (Eliminar MITM en etcd)
**Prioridad:** ALTA enterprise

Elimina V-001: sniffer genera semillas efímeras y las distribuye directamente
a ml-detector vía ZeroMQ cifrado. etcd queda como plano de control exclusivamente.
Perfect Forward Secrecy. Zero-downtime rotation.

### ENT-4 — Hot-Reload de Configuración en Runtime
**Prioridad:** ALTA enterprise

Watcher sobre etcd. Un hospital no puede reiniciar el pipeline para cambiar
un threshold. Relación con ENT-3: una vez implementado, etcd solo gestiona
configuración — nunca secretos.

### ENT-5 — rag-world (Telemetría Global Federada)
**Prioridad:** MEDIA enterprise
**Relación:** Infraestructura base para ENT-1

### ENT-6 — Integración Threat Intelligence (MISP/OpenCTI)
**Prioridad:** ALTA enterprise
- [ ] MISP via API REST + cache local IOCs
- [ ] Compatible con OpenCTI (STIX/TAXII)

### ENT-7 — Observabilidad OpenTelemetry + Grafana
**Prioridad:** MEDIA enterprise

### ENT-8 — SecureBusNode (HSM + USB Root Key)
**Prioridad:** MEDIA enterprise
**Prerequisito:** ENT-3 implementado

---

## 📊 Estado global del proyecto

```
                              [criterio: impl+test E2E+documentado = 100%]

Foundation + Thread-Safety:       ████████████████████ 100% ✅
Contract Validation:              ████████████████████ 100% ✅
Build System:                     ████████████████████ 100% ✅
HMAC Infrastructure (F1+F2):      ████████████████████ 100% ✅
Proto3 Pipeline Stability:        ████████████████████ 100% ✅
Logging Standard (6 components):  ████████████████████ 100% ✅  ← DAY 79
Sentinel Correctness:             ████████████████████ 100% ✅  ← DAY 79
F1-Score Validation (CTU-13):     ████████████████████ 100% ✅  ← DAY 79 F1=0.9921
Thresholds desde JSON:            ████████████████████ 100% ✅  ← DAY 80 F1=0.9934
CSV Pipeline ml-detector:         ████████████████░░░░  80% 🟡  impl+unit, E2E pendiente
CSV Pipeline firewall-acl-agent:  ████████████████░░░░  80% 🟡  impl+unit, E2E pendiente
trace_id correlación:             ████████████████░░░░  80% 🟡  impl, 2 fallos pendientes
Test Suite:                       ████████████████░░░░  80% 🟡  2 fallos trace_id
Ring Consumer Real Features:      ████████████░░░░░░░░  60% 🟡  ← DAY 79/80 (28/40)
rag-local (community):            ████░░░░░░░░░░░░░░░░  20% 🟡
F1-Score Validación (balanceado): ░░░░░░░░░░░░░░░░░░░░   0% ⏳  ← P0 paper DAY 81
FASE 3 rag-ingester HMAC:         ░░░░░░░░░░░░░░░░░░░░   0% ⏳
CsvEventLoader rag-ingester:      ░░░░░░░░░░░░░░░░░░░░   0% ⏳
simple-embedder CSV:              ░░░░░░░░░░░░░░░░░░░░   0% ⏳
Attack Graph Generation:          ░░░░░░░░░░░░░░░░░░░░   0% ⏳  ← ENT-2
Federated Threat Intelligence:    ░░░░░░░░░░░░░░░░░░░░   0% ⏳  ← ENT-1
P2P Seed Distribution:            ░░░░░░░░░░░░░░░░░░░░   0% ⏳  ← ENT-3
rag-world (enterprise):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳

Pipeline Security:
├─ Crypto-Transport:   ✅ ChaCha20-Poly1305 + LZ4
├─ HMAC (F1+F2):       ✅ SHA256 key management
├─ CSV Integrity:      ✅ HMAC por fila (unit tested, E2E pendiente)
├─ Proto3 Stability:   ✅ sentinel init — DAY 76
├─ Sentinel Quality:   ✅ 0.5f placeholders eliminados — DAY 79
├─ Thresholds JSON:    ✅ Phase1-Day4-CRITICAL CERRADO — DAY 80
├─ Seed Distribution:  ⚠️  etcd centralizado (V-001 documentado) → ENT-3
├─ FASE 3 HMAC:        ⏳ rag-ingester validation
└─ SecureBusNode:      ⏳ enterprise only (ENT-8)
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
| Sentinel correctness | -9999.0f fuera del dominio = determinista y auditable ✅ DAY 79 |
| 0.5f TCP half-open | Valor semántico válido — comentario protector en código ✅ DAY 79 |
| Thresholds ML | Desde JSON — Phase1-Day4-CRITICAL CERRADO ✅ DAY 80 |
| Log standard | /vagrant/logs/lab/COMPONENTE.log — un fichero por componente ✅ DAY 79 |
| GeoIP en critical path | ❌ Deliberadamente fuera — latencia inaceptable (100-500ms) |
| io_intensity/resource_usage | SENTINEL Phase 1 — requiere eBPF tracepoints Phase 2 |
| StrictSnifferConfig vs SnifferConfig | Mapeo explícito campo a campo — nunca asignación directa de structs distintos ✅ DAY 80 |
| Seed distribution (open source) | etcd-server — suficiente para demo y entornos controlados |
| Seed distribution (enterprise) | P2P via protobuf — PFS, sin etcd, elimina V-001 ← ENT-3 |
| Hot-reload configuración | Enterprise only — etcd watcher sin secretos ← ENT-4 |
| Federated learning | Opt-in, anonimización local obligatoria ← ENT-1 |
| Attack graphs | GraphML + STIX 2.1 + streaming SOC/CAI/MITRE ATT&CK ← ENT-2 |

---

*Última actualización: Day 80 — 9 Mar 2026*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic), Grok, ChatGPT5, DeepSeek, Qwen*