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
  - Nota honesta: FPR=79% en benigno por desequilibrio dataset (98% tráfico atacante)

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

### DAY 80 — Thresholds desde JSON + Features + F1 post-fix

**P0 — Thresholds desde JSON (Phase1-Day4-CRITICAL)**

Thresholds hardcodeados en ring_consumer.cpp violan "JSON is the law":
- DDoS: `0.7f`, Ransomware: `0.75f`, Traffic: `0.7f`, Internal: `0.00000000065f`

```cpp
struct MLThresholds {
    float ddos       = 0.7f;   // fallback EXPLÍCITO, nunca silencioso
    float ransomware = 0.75f;
    float traffic    = 0.7f;
    float internal   = 0.00000000065f;
};
// Leer en initialize() desde ml_detector_config.json
// Si falta la clave → excepción. Nunca silencio.
```

**P1 — Inspección FlowStatistics → features atacables**
- `tcp_udp_ratio`: viable si hay conteo TCP/UDP por protocolo
- `flow_duration_std` / `connection_duration_std`: solo con multi-flow
- `protocol_variety`: solo con multi-flow

**Criterio de merge a main:**
- [ ] Thresholds desde JSON implementados y testeados
- [ ] Pipeline 6/6 RUNNING post-merge
- [ ] F1 ≥ 0.99 reproducible con `make test-replay-neris`
- [ ] `make logs-all` funciona

---

## 📋 BACKLOG — COMMUNITY

### Validación con datasets balanceados (post-merge DAY 80)
**Prioridad:** ALTA — prerequisito paper

CTU-13 Neris tiene 98% tráfico atacante. F1=0.9921 con Recall=1.0 no
demuestra comportamiento en tráfico mixto real. Necesario para el paper.

Datasets candidatos:
- **CIC-IDS2017** (Universidad de New Brunswick) — tráfico mixto balanceado,
  múltiples tipos de ataque, referencia estándar en literatura
- **UNSW-NB15** (UNSW Canberra) — 9 categorías de ataque + benigno real
- **MAWI Working Group** — tráfico real de backbone, representativo de Internet real

Métricas objetivo: F1, Precision, Recall por clase + matriz de confusión completa.
Para el paper: tabla comparativa CTU-13 vs CIC-IDS2017 vs UNSW-NB15.

### CSV Pipeline E2E — validación con tráfico real
**Prioridad:** ALTA
**Estado actual:** 80% (implementado, compilando, unit tests ✅, E2E pendiente)

- [ ] Ejecutar `make test-replay-neris` y confirmar que los CSV de ml-detector
  y firewall-acl-agent se generan correctamente con trace_id correlacionado
- [ ] Verificar HMAC por fila en los CSV generados con tráfico real
- [ ] Confirmar que trace_id une eventos de ml-detector y firewall-acl-agent
  para el mismo flujo
- [ ] Investigar y resolver los 2 fallos preexistentes de test_trace_id (DAY 72)
- [ ] Verificar que rag-ingester consume los CSV generados sin errores

**Criterio de compleción:** CSV generados, HMAC validado, trace_id correlacionado
entre los dos componentes, demostrable con un replay real → 100% ✅

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

### FASE 4 — Grace Period + Key Versioning
**Prioridad:** MEDIA
**Prerequisito:** FASE 3 completa
- [ ] KeyVersion struct + deque por key path
- [ ] Validador: current → previous dentro de grace period

### Unificar logs ml-detector
**Prioridad:** MEDIA
Actualmente coexisten `detector.log` (spdlog interno) y `ml-detector.log`
(stdout Makefile). Mover `log_file` al JSON de configuración de cada componente.

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
**Motivación:** Una instalación que detecta una variante nueva contribuye al
conocimiento colectivo sin exponer datos sensibles. Mecanismo inmunológico
distribuido: cada instalación desarrolla "anticuerpos", toda la red se beneficia.

**Arquitectura propuesta:**
```
[Instalación detecta variante nueva]
         ↓
[Anonimización local obligatoria]
  - Eliminar IPs, MACs, puertos de aplicación
  - Normalizar timestamps (offsets relativos)
  - Preservar patrones de comportamiento (IAT, entropía, flags)
         ↓
[Contribución a servidor central — opt-in]
  rag-world recibe vectores de features anonimizados
         ↓
[Reentrenamiento del ensemble central]
  Nuevo RandomForest entrenado con datos federados
         ↓
[Distribución como actualización binaria]
  Nuevo ml-detector embebe el modelo actualizado
```

**Garantías de privacidad requeridas:**
- Contribución 100% opt-in
- Ninguna instalación expone IPs, MACs ni contenido de tráfico
- Modelo central no puede reconstruir datos originales

**Referencias:**
- McMahan et al., "Communication-Efficient Learning of Deep Networks
  from Decentralized Data" (FedAvg, AISTATS 2017)
- Nguyen et al., "Federated Learning for Intrusion Detection System",
  Computer Networks 2022

### ENT-2 — Attack Graph Generation (SOC Integration)
**Prioridad:** ALTA enterprise
**Motivación:** Los CSV de ml-detector y firewall-acl-agent contienen toda la
información para construir grafos de ataque en tiempo real, consumibles por
herramientas SOC modernas (CAI framework, SIEM, OpenCTI, MITRE ATT&CK).

**Modelo del grafo:**
```
Nodos:
  - IPNode:       src/dst IP con atributos (reputation, geo, ASN)
  - PortNode:     servicio/puerto con contexto de protocolo
  - EventNode:    detección o bloqueo con timestamp y score ML
  - CampaignNode: cluster de eventos relacionados temporalmente

Aristas:
  - COMMUNICATES_WITH: IP → IP (flujo de red)
  - TARGETS:           IP → Port (intento de conexión)
  - DETECTED_AS:       EventNode → tipo de ataque
  - BLOCKED_BY:        EventNode → firewall rule
  - PART_OF:           EventNode → CampaignNode
  - FOLLOWS:           EventNode → EventNode (secuencia temporal)
```

**Formatos de salida:**
- GraphML / GEXF: compatible con Gephi, análisis offline
- STIX 2.1: estándar threat intelligence, compatible con MISP/OpenCTI
- Cypher (Neo4j): consultas relacionales sobre el grafo
- Streaming (WebSocket/SSE): grafos que crecen en tiempo real

**Hito mínimo viable:** GraphML estático desde `make test-replay-neris`
abierto en Gephi mostrando el grafo de comunicaciones de 147.32.84.165.

### ENT-3 — P2P Seed Distribution via Protobuf (Eliminar MITM en etcd)
**Prioridad:** ALTA enterprise
**Motivación:** etcd-server es actualmente el punto único de compromiso
criptográfico (V-001 documentado). Distribuye la semilla ChaCha20 compartida
a todos los componentes en el arranque. Un atacante que comprometa etcd obtiene
capacidad completa de descifrado y suplantación de cualquier componente.

**Mecanismo propuesto:**
El sniffer genera semillas efímeras y las distribuye directamente a ml-detector
dentro del canal ZeroMQ ya cifrado, embebidas en el contrato protobuf.
etcd-server deja de ser autoridad criptográfica y queda como plano de control
exclusivamente (discovery + configuración JSON).

```
Flujo actual — open source (etcd centralizado):
  etcd-server → semilla compartida → todos los componentes
  Riesgo: comprometer etcd = comprometer todo el pipeline

Flujo propuesto — enterprise (P2P via protobuf):
  1. sniffer genera nueva semilla efímera (CSPRNG)
  2. sniffer cifra la nueva semilla con la semilla actual
  3. sniffer embebe en NetworkEvent protobuf:
       CryptoHandoff {
         next_seed:      bytes  // cifrado con semilla actual
         activate_at_ts: uint64 // Unix timestamp de activación
         seed_version:   uint32 // para audit trail
       }
  4. ml-detector recibe, descifra con semilla actual
  5. ml-detector activa nueva semilla en activate_at_ts
  6. Transición transparente: comunicación nunca interrumpida
  7. etcd-server no participa en ningún paso
```

**Propiedades de seguridad obtenidas:**
- Elimina V-001: comprometer etcd ya no expone semillas de comunicación
- Perfect Forward Secrecy: semillas efímeras con rotación periódica
- Aislamiento por par: sniffer↔ml-detector tienen su propia semilla,
  independiente del resto del pipeline
- Anti-replay: timestamp de activación impide re-inyección de handoffs anteriores
- Zero-downtime rotation: el canal sigue cifrado durante toda la transición

**Campo protobuf requerido (NetworkEvent):**
```protobuf
message CryptoHandoff {
  bytes  next_seed       = 1;  // cifrado con semilla actual
  uint64 activate_at_ts  = 2;  // Unix timestamp de activación
  uint32 seed_version    = 3;  // para audit trail
}
        optional CryptoHandoff crypto_handoff = 99;
```

**Extensión natural a todos los pares del pipeline:**
- sniffer → ml-detector (par principal)
- ml-detector → firewall-acl-agent
- ml-detector → rag-ingester

**Nota para el paper:** Documentar como arquitectura objetivo de producción.
La versión open source usa etcd para distribución de semilla (suficiente para
demostración y entornos controlados). Enterprise elimina esa dependencia.

### ENT-4 — Hot-Reload de Configuración en Runtime
**Prioridad:** ALTA enterprise
**Motivación:** Un hospital no puede reiniciar el pipeline para cambiar un
threshold. Los parámetros deben ser modificables en caliente sin downtime.

- [ ] Watcher sobre etcd que monitoriza cambios en JSON de configuración
- [ ] Hot-reload sin reiniciar: thresholds ML, retención CSV, rutas de archivo
- [ ] Validación del nuevo JSON antes de aplicar (rollback si inválido)
- [ ] Notificación a componentes activos via ZeroMQ o señal interna
- [ ] Audit log: quién cambió qué, cuándo, valor anterior y nuevo
- **Relación con ENT-3:** Una vez implementado ENT-3, etcd solo gestiona
  configuración — nunca secretos criptográficos. El watcher es seguro.

### ENT-5 — rag-world (Telemetría Global Federada)
**Prioridad:** MEDIA enterprise
**Relación:** Infraestructura base para ENT-1

```
[Instalación A]    [Instalación B]    [Instalación C]
  rag-local  ──┐     rag-local  ──┤     rag-local  ──┐
               ↓                  ↓                  ↓
          ┌─────────────────────────────────────────┐
          │              rag-world                   │
          │  telemetría global + modelo federado     │
          └─────────────────────────────────────────┘
```

### ENT-6 — Integración Threat Intelligence (MISP/OpenCTI)
**Prioridad:** ALTA enterprise
- [ ] Integración MISP via API REST — consulta por src_ip/dst_ip en MALICIOUS
- [ ] Cache local de IOCs (TTL configurable)
- [ ] Feeds: CERTs europeos, listas negras ransomware conocido
- [ ] Compatible con OpenCTI (misma API STIX/TAXII)
- **Relación con ENT-2:** Los grafos exportables como STIX 2.1 bundles
  directamente hacia MISP/OpenCTI

### ENT-7 — Observabilidad OpenTelemetry + Grafana
**Prioridad:** MEDIA enterprise
- [ ] Métricas pipeline en formato OTEL
- [ ] Dashboards Grafana: latencia, throughput, eventos/seg, HMAC failures
- [ ] Alertas: anomalías en volumen, fallos HMAC, rotación de claves

### ENT-8 — SecureBusNode (HSM + USB Root Key)
**Prioridad:** MEDIA enterprise
**Prerequisito:** ENT-3 implementado
- [ ] USB encrypted storage como origen de clave raíz
- [ ] Hardware Security Modules (HSM) — IRootKeyProvider interface
- [ ] Detección y recuperación ante compromiso de clave raíz

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
CSV Pipeline ml-detector:         ████████████████░░░░  80% 🟡  impl+unit, E2E pendiente
CSV Pipeline firewall-acl-agent:  ████████████████░░░░  80% 🟡  impl+unit, E2E pendiente
trace_id correlación:             ████████████████░░░░  80% 🟡  impl, 2 fallos pendientes
Test Suite:                       ████████████████░░░░  80% 🟡  2 fallos trace_id
Ring Consumer Real Features:      ████████████░░░░░░░░  60% 🟡  ← DAY 79
rag-local (community):            ████░░░░░░░░░░░░░░░░  20% 🟡
F1-Score Validación (balanceado): ░░░░░░░░░░░░░░░░░░░░   0% ⏳  ← post-merge
Thresholds desde JSON:            ░░░░░░░░░░░░░░░░░░░░   0% ⏳  ← DAY 80 P0
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
| Thresholds ML | Desde JSON — nunca hardcodeados. "JSON is the law" ⏳ DAY 80 |
| Log standard | /vagrant/logs/lab/COMPONENTE.log — un fichero por componente ✅ DAY 79 |
| GeoIP en critical path | ❌ Deliberadamente fuera — latencia inaceptable (100-500ms) |
| io_intensity/resource_usage | SENTINEL Phase 1 — requiere eBPF tracepoints Phase 2 |
| Seed distribution (open source) | etcd-server — suficiente para demo y entornos controlados |
| Seed distribution (enterprise) | P2P via protobuf — PFS, sin etcd, elimina V-001 ← ENT-3 |
| Hot-reload configuración | Enterprise only — etcd watcher sin secretos ← ENT-4 |
| Federated learning | Opt-in, anonimización local obligatoria ← ENT-1 |
| Attack graphs | GraphML + STIX 2.1 + streaming SOC/CAI/MITRE ATT&CK ← ENT-2 |

---

*Última actualización: Day 79 — 8 Mar 2026*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic), Grok, ChatGPT5, DeepSeek, Qwen*