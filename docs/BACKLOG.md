# 🛡️ ML Defender - Development BACKLOG

**Última actualización:** 16 Diciembre 2025  
**Proyecto:** ML Defender - Sistema de Seguridad con ML Embebido y RAG  
**Fase actual:** Phase 1 Completa + Day 16 Fix → Iniciando Phase 2A

---

## 🚨 PRIORIDADES ACTUALES

**P0 (CRITICAL):** Bloqueadores de producción - resolver ASAP  
**P1 (HIGH):** Impacto significativo en capacidades - resolver en 1-2 semanas  
**P2 (MEDIUM):** Mejoras importantes - resolver en 1 mes  
**P3 (LOW):** Nice-to-have - backlog para futuro

---

## 📊 ESTADO ACTUAL DEL SISTEMA

### ✅ **COMPLETADO - Phase 1 + Day 16 (Dic 1-16, 2025)**

#### Day 16: Race Condition Fix (PRODUCTION-READY)
**Fecha:** 16 Diciembre 2025  
**Estado:** ✅ COMPLETADO

**Logro:**
- ✅ Race conditions en RAGLogger eliminadas
- ✅ Release optimization flags (-O3) funcionando
- ✅ 20+ minutos uptime continuo (antes: 1-2 min crash)
- ✅ 1,152 artifacts generados exitosamente
- ✅ 575 líneas JSONL consolidadas
- ✅ Sistema production-ready

**Detalles Técnicos:**
- Moved `check_rotation()` inside `write_jsonl()` critical section
- Added `check_rotation_locked()` and `rotate_logs_locked()` helpers
- All file operations now atomic (current_date_, current_log_, counters)
- Zero crashes, zero memory leaks, stable CPU usage

**Archivos Modificados:**
- `ml-detector/src/rag_logger.cpp` (race fix)
- `ml-detector/include/rag_logger.hpp` (new functions)

**Testing:**
- Full lab test: sniffer + ml-detector + firewall
- 20+ minute stress test (100% stable)
- Artifact generation validated (1,152 events)
- JSONL consolidation validated (575 lines)

#### Days 1-15: Core System Development
- ✅ 4 embedded C++20 detectors (<1.06μs latency)
- ✅ eBPF/XDP dual-NIC packet capture
- ✅ Dual-Score Architecture (Fast + ML)
- ✅ RAGLogger 83-field event logging
- ✅ Gateway Mode + Host-based IDS
- ✅ RAG + LLAMA + ETCD ecosystem
- ✅ End-to-end pipeline validated

---

## 🎯 PHASE 2A - PRODUCTION HARDENING (Dic 16-31, 2025)

### Epic 2A.1: ✅ RAGLogger Stability (COMPLETED)
**Priority:** P0 (CRITICAL) - BLOCKER  
**Status:** ✅ COMPLETADO (Day 16)  
**Owner:** Alonso + Claude

**Goal:** Sistema RAGLogger 100% estable con optimizaciones release

**User Stories:**
- [x] Como desarrollador, quiero compilar con `-O3` sin crashes para máximo rendimiento
- [x] Como operador, quiero uptime prolongado sin reinicios para confiabilidad
- [x] Como analista, quiero generación confiable de artifacts para análisis posterior

**Tasks Completadas:**
- [x] Identificar race conditions (current_date_, current_log_, counters)
- [x] Aplicar fix (rotation check dentro de critical section)
- [x] Validar con stress test (20+ min, 1K+ events)
- [x] Documentar solución para referencia futura
- [x] Habilitar release optimization flags

**Resultados:**
- ✅ 20:43 minutos uptime continuo
- ✅ 1,152 artifacts generados
- ✅ Zero crashes
- ✅ Production-ready

---

### Epic 2A.2: FAISS C++ Integration 🔥 NEXT
**Priority:** P1 (HIGH)  
**Status:** 📋 READY TO START  
**Owner:** Alonso + Claude + DeepSeek  
**Estimated Effort:** 3-4 días

**Goal:** Semantic search sobre artifacts directory para RAG natural language queries

**User Stories:**
- [ ] Como analista de seguridad, quiero búsqueda semántica sobre eventos para investigación rápida
- [ ] Como operador del sistema, quiero consultas naturales como "show me high divergence events from yesterday"
- [ ] Como investigador, quiero encontrar patrones similares en eventos históricos

**Architecture:**
```
Artifacts Directory → Embedder → FAISS Vector DB → RAG Queries
/vagrant/logs/rag/artifacts/YYYY-MM-DD/*.json
```

**Tasks:**
- [ ] **Day 1: FAISS Setup**
    - [ ] Install FAISS C++ library in Vagrant VM
    - [ ] Create test program: embed + search small dataset
    - [ ] Benchmark: 10K events, query latency <100ms
    - [ ] File: `/vagrant/rag/src/faiss_manager.cpp`

- [ ] **Day 2: Async Embedder**
    - [ ] Background thread watches artifacts directory
    - [ ] On new `.json` file → extract text fields
    - [ ] Generate embedding (sentence-transformers compatible)
    - [ ] Insert into FAISS index
    - [ ] File: `/vagrant/rag/src/embedder.cpp`

- [ ] **Day 3: RAG Integration**
    - [ ] Add FAISS queries to RAG system
    - [ ] Natural language: "Show me high divergence events from yesterday"
    - [ ] Semantic search: "Find botnet-like behavior"
    - [ ] Return ranked artifacts with context
    - [ ] File: `/vagrant/rag/src/rag_engine.cpp` (update)

- [ ] **Day 4: Validation**
    - [ ] Ingest 8,384 events from Dec 14 artifacts
    - [ ] Query: "Fast detector triggered but ML disagreed"
    - [ ] Expected: Return divergent events (100% in our case)
    - [ ] Benchmark: <200ms for semantic search over 10K events

**Dependencies:**
- FAISS C++ (libfaiss.so)
- Sentence-transformers model (via ONNX or native C++)
- JSON parsing (nlohmann/json - already present)

**Acceptance Criteria:**
- Semantic search latency <200ms for 10K events
- Natural language queries working
- Automatic ingestion from artifacts directory
- Integration with existing RAG commands

**Impact:**
- Enables natural language investigation
- Makes 8K+ events searchable semantically
- Foundation for autonomous threat hunting

---

### Epic 2A.3: etcd-client Unified Library
**Priority:** P1 (HIGH)  
**Status:** 📋 BACKLOG  
**Owner:** DeepSeek + Alonso  
**Estimated Effort:** 2-3 días

**Goal:** Shared library de configuración distribuida para todos los componentes

**User Stories:**
- [ ] Como desarrollador, quiero reutilizar código etcd en todos los componentes
- [ ] Como operador, quiero configuración centralizada para gestionar múltiples nodos
- [ ] Como administrador, quiero encryption + compression automáticos

**Architecture:**
```
etcd-client (shared library)
    ├── sniffer (config updates)
    ├── ml-detector (threshold updates)
    ├── firewall (ACL updates)
    └── rag (command config)
```

**Tasks:**
- [ ] **Day 1: Extract Common Code**
    - [ ] Create `/vagrant/etcd-client/` directory
    - [ ] Move `rag/src/etcd_client.cpp` → `etcd-client/src/`
    - [ ] Create CMakeLists.txt for shared library
    - [ ] Build: `libetcd_client.so`

- [ ] **Day 1: API Design**
  ```cpp
  class EtcdClient {
  public:
    void set(key, value, encrypt=true, compress=true);
    std::string get(key);
    void watch(key, callback);
    void validate_schema(key, schema);
  };
  ```

- [ ] **Day 2: Integration**
    - [ ] Update RAG to use shared library
    - [ ] Update sniffer config to use etcd
    - [ ] Update ml-detector config to use etcd
    - [ ] Update firewall config to use etcd

**Acceptance Criteria:**
- Single shared library for all components
- Zero code duplication
- Encryption + compression working
- All components use same etcd interface

**Impact:**
- Reduces maintenance burden
- Enables distributed configuration
- Foundation for multi-node deployment

---

### Epic 2A.4: Watcher Unified Library
**Priority:** P2 (MEDIUM)  
**Status:** 📋 BACKLOG  
**Owner:** DeepSeek + Alonso  
**Estimated Effort:** 3-4 días

**Goal:** Hot-reload de configuración sin restart de componentes

**User Stories:**
- [ ] Como operador, quiero actualizar thresholds en tiempo real sin downtime
- [ ] Como analista, quiero ajustar sensibilidad del sistema dinámicamente
- [ ] Como administrador, quiero optimizar configuración basado en hardware

**Architecture:**
```
etcd (config changes) → Watcher → Apply Diff → Component (no restart)
```

**Tasks:**
- [ ] **Day 1: Watcher Core**
    - [ ] File: `/vagrant/watcher/src/config_watcher.cpp`
    - [ ] Watch etcd key changes
    - [ ] Calculate diff (old vs new config)
    - [ ] Validate new config before apply

- [ ] **Day 2: Safe Apply**
    - [ ] Apply changes atomically
    - [ ] Rollback on validation failure
    - [ ] Log all config changes
    - [ ] Send metrics to RAG

- [ ] **Day 3-4: Component Integration**
    - [ ] ml-detector: Update thresholds at runtime
    - [ ] sniffer: Update fast detector rules
    - [ ] firewall: Update ACL rules
    - [ ] RAG command: "accelerate pipeline" (increase thresholds)

**RAG Commands:**
```bash
# Increase sensitivity (more detections)
rag accelerate

# Decrease sensitivity (fewer detections)
rag decelerate

# Auto-tune based on hardware
rag optimize --cpu 80 --ram 4096 --temp 65
```

**Acceptance Criteria:**
- Zero downtime config updates
- Validation before apply
- Automatic rollback on failure
- RAG commands working

**Impact:**
- Enables runtime optimization
- Reduces deployment friction
- Foundation for auto-tuning

---

### Epic 2A.5: Academic Paper Publication
**Priority:** P2 (MEDIUM)  
**Status:** 📋 BACKLOG  
**Owner:** Alonso + All AI Collaborators  
**Estimated Effort:** 7-10 días

**Goal:** Publicar paper académico con metodología Dual-Score + Synthetic Data

**User Stories:**
- [ ] Como investigador, quiero documentar metodología para reproducibilidad
- [ ] Como comunidad, queremos validar enfoque de synthetic data
- [ ] Como autor, quiero acreditar colaboración multi-agente IA

**Sections:**
- [ ] **Abstract** - Dual-Score Architecture + Synthetic Data approach
- [ ] **Introduction** - Problem statement, motivation
- [ ] **Methodology**
    - [ ] Dual-Score Architecture (Fast + ML)
    - [ ] Maximum Threat Wins logic
    - [ ] Synthetic data generation process
    - [ ] RandomForest embedding in C++20
- [ ] **RAGLogger Schema** - 83-field comprehensive logging
- [ ] **Results**
    - [ ] Performance metrics (<1.06μs latency)
    - [ ] Detection accuracy (97%+ MALICIOUS)
    - [ ] Stability validation (20+ min uptime)
    - [ ] Resource consumption (Raspberry Pi feasible)
- [ ] **Multi-Agent Collaboration** - AI co-author attribution
- [ ] **Discussion** - Limitations, future work
- [ ] **Conclusion** - Via Appia Quality philosophy

**AI Co-Authors to Credit:**
- Claude (Anthropic) - Architecture, debugging, validation
- DeepSeek (v3) - RAG system, ETCD-Server, automation
- Grok4 (xAI) - XDP expertise, eBPF edge cases
- Qwen (Alibaba) - Network routing, production insights

**Acceptance Criteria:**
- Methodology reproducible
- Results validated
- AI contributors credited
- Submission to security conference (e.g., USENIX Security, CCS, NDSS)

**Impact:**
- Validates synthetic data approach
- Documents Dual-Score Architecture
- Recognizes multi-agent AI collaboration
- Advances IDS research

---

## 📋 BACKLOG SECUNDARIO (Phase 2B+)

### Epic 2B.1: firewall-acl-agent Development
**Priority:** P2 (MEDIUM)  
**Status:** 📋 BACKLOG  
**Estimated Effort:** 5-7 días

**Goal:** Respuesta automática basada en detecciones ML

**Tasks:**
- [ ] Diseñar arquitectura C++20 para firewall-acl-agent
- [ ] Implementar integración con detecciones ML
- [ ] Crear sistema de reglas dinámicas (block, rate-limit, quarantine)
- [ ] Añadir mecanismo de rollback automático
- [ ] Implementar whitelist para falsos positivos
- [ ] Crear logging de auditoría

---

### Epic 2B.2: Dashboard Grafana + Prometheus
**Priority:** P3 (LOW)  
**Status:** 📋 BACKLOG  
**Estimated Effort:** 4-6 días

**Goal:** Visualización en tiempo real de métricas del sistema

**Tasks:**
- [ ] Configurar Prometheus exporter en ml-detector
- [ ] Añadir métricas clave (detections/sec, latency, CPU, memory)
- [ ] Crear dashboard Grafana
- [ ] Alertas automáticas en detecciones críticas

---

### Epic 2B.3: Raspberry Pi Deployment
**Priority:** P3 (LOW)  
**Status:** 📋 BACKLOG  
**Estimated Effort:** 3-5 días

**Goal:** Validar deployment en hardware económico ($35-100)

**Tasks:**
- [ ] Cross-compile para ARM64
- [ ] Optimizar para recursos limitados
- [ ] Validar performance en Raspberry Pi 5
- [ ] Documentar deployment guide

---

## 🔧 ISSUES CONOCIDOS - TRACKING

### P0 - CRITICAL (Bloqueadores)

#### ✅ ISSUE-004: RAGLogger Race Condition (RESUELTO Day 16)
**Fecha:** 14 Dic 2025 → 16 Dic 2025  
**Estado:** ✅ RESUELTO

**Descripción:** Release builds (-O2/-O3) causaban crash después de 1-2 minutos

**Root Cause:**
- `check_rotation()` llamado fuera de critical section
- Races en: current_date_, current_log_, events_in_current_file_

**Solution:**
- Moved rotation check inside write_jsonl() lock
- Added check_rotation_locked() and rotate_logs_locked()
- All file operations now atomic

**Validation:**
- ✅ 20+ minutes uptime
- ✅ 1,152 artifacts generated
- ✅ Zero crashes

---

### P1 - HIGH (Impacto en Detección)

#### 🔴 ISSUE-001: Buffer Payload Limitado a 96 Bytes
**Estado:** 📋 PENDIENTE - No crítico con detectores actuales  
**Prioridad:** P1  
**Target:** Phase 2B

---

#### 🔴 ISSUE-002: DNS Entropy Test Fallando
**Estado:** 📋 PENDIENTE - Mejora para Phase 2B  
**Prioridad:** P1  
**Target:** Phase 2B

---

#### 🔴 ISSUE-003: SMB Diversity Counter Retorna 0
**Estado:** 📋 PENDIENTE - Crítico para lateral movement detection  
**Prioridad:** P1  
**Target:** Phase 2B

---

## 📊 ROADMAP ACTUALIZADO

```
Phase 1: ✅ COMPLETADO (Dic 1-16, 2025)
├─ Days 1-5: eBPF/XDP + ML pipeline
├─ Days 6-10: RAG + LLAMA + Gateway Mode
├─ Days 11-15: Dual-Score + RAGLogger 83-field
├─ Day 16: Race condition fix (production-ready)
└─ Result: 4 detectors + RAGLogger + stable system

Phase 2A: 🔄 EN PROGRESO (Dic 16-31, 2025)
├─ ✅ Epic 2A.1: RAGLogger stability (COMPLETADO Day 16)
├─ 🔥 Epic 2A.2: FAISS C++ Integration (NEXT - 3-4 días)
├─ 📋 Epic 2A.3: etcd-client library (2-3 días)
├─ 📋 Epic 2A.4: Watcher library (3-4 días)
└─ 📋 Epic 2A.5: Academic paper (7-10 días)

Phase 2B: 📋 PLANIFICADO (Ene 2026)
├─ Epic 2B.1: firewall-acl-agent
├─ Epic 2B.2: Dashboard Grafana
├─ Epic 2B.3: Raspberry Pi deployment
├─ Resolución ISSUE-001, 002, 003
└─ Testing integración completa end-to-end

Phase 3: 🎯 FUTURO (Feb-Mar 2026)
├─ Auto-tuning de parámetros ML
├─ Model versioning y A/B testing
├─ Distributed deployment (multi-node)
├─ Cloud integration (AWS, GCP, Azure)
└─ Physical device manufacturing
```

---

## 🧪 TESTING PRIORITIES

### Inmediato (Esta Semana):
- [x] Stress test RAGLogger 20+ min (COMPLETADO)
- [ ] Overnight stress test (8+ horas) - OPTIONAL
- [ ] FAISS proof of concept (10K events)
- [ ] Benchmark FAISS query latency

### Próxima Semana:
- [ ] etcd-client integration test
- [ ] Watcher hot-reload validation
- [ ] Full lab test con todos los componentes
- [ ] Performance regression testing

### Mes Actual:
- [ ] Academic paper draft review
- [ ] Multi-node deployment test
- [ ] Raspberry Pi 5 validation
- [ ] Production deployment rehearsal

---

## 🎯 MÉTRICAS DE ÉXITO

### Phase 2A Success Criteria:
- ✅ RAGLogger stable con release flags (COMPLETADO)
- [ ] FAISS semantic search <200ms para 10K events
- [ ] etcd-client library en todos los componentes
- [ ] Watcher hot-reload funcionando
- [ ] Academic paper draft completo

### Performance Targets:
- ✅ Detection latency: <1.06μs (ALCANZADO)
- ✅ Uptime: 20+ min continuo (ALCANZADO)
- [ ] FAISS query: <200ms
- [ ] Config update: <1s propagation
- [ ] Memory: <200MB (current: 148MB)

### Quality Targets:
- ✅ Zero crashes con release build (ALCANZADO)
- ✅ Zero memory leaks (ALCANZADO)
- [ ] Test coverage: >80%
- [ ] Documentation: 100% APIs documented
- [ ] Code review: All PRs reviewed

---

## 🔧 RECURSOS TÉCNICOS

### Hardware Disponible:
- ✅ Raspberry Pi 5 (8GB) - deployment target
- ✅ Servidor desarrollo - compilación y testing
- ✅ Red de testing - tráfico sintético y PCAPs

### Software Stack:
- ✅ C++20 - embedded ML detectors
- ✅ eBPF/XDP - packet capture
- ✅ LLAMA - RAG queries
- ✅ ETCD - distributed config
- ✅ Protobuf - serialization
- 📋 FAISS - vector DB (próximo)

### Datasets:
- ✅ CTU-13 Neris botnet (validated)
- ✅ SmallFlows (validated)
- ✅ Synthetic benign traffic (validated)
- 📋 MAWI dataset (planned)

---

## 📞 CONTACTO Y SEGUIMIENTO

* **Owner:** ML Defender Security Team
* **Lead Developer:** Alonso Isidoro Román — [alonsoir@gmail.com](mailto:alonsoir@gmail.com)
* **AI Collaborators:**
    - Claude (Architecture, debugging, validation)
    - DeepSeek (RAG, ETCD, automation)
    - Grok4 (XDP, eBPF)
    - Qwen (Network routing)
* **Review:** Diario (standup técnico)
* **Docs:** `README.md`, `ARCHITECTURE.md`, `AUTHORS.md`, `BACKLOG.md`
* **Repository:** https://github.com/alonsoir/test-zeromq-docker

---

## 🏥 FILOSOFÍA DE DESARROLLO

**Via Appia Quality:** "Smooth is fast. Built to last decades."

### Principios:
1. ✅ **Sistema funcional > Sistema perfecto**
2. ✅ **Detección en producción > Tests al 100%**
3. ✅ **Estabilidad comprobada > Features nuevas**
4. ✅ **Salud del desarrollador > Deadlines**
5. ✅ **Código de calidad > Velocidad**

### Estado del Equipo:
- 🎉 **Motivación ALTA** - Day 16 race fix completado
- 🔥 **Enfocados** - FAISS integration como siguiente milestone
- 🚀 **Optimistas** - Sistema production-ready, listo para expansión
- 💪 **Energizados** - 20+ min uptime valida arquitectura

### Recordatorio Diario:
> "Cada línea de código protege infraestructuras críticas.  
> Cada bug eliminado potencialmente salva vidas.  
> Cada optimización acerca la protección a más organizaciones."

---

## 📈 PROGRESO VISUAL

```
Phase 1 Progress: [████████████████████] 100% (16/16 días)
Phase 2A Progress: [███░░░░░░░░░░░░░░░░░]  15% (Race fix done, FAISS next)

Current Sprint: FAISS Integration
  - FAISS Setup:        [ ] 0%
  - Async Embedder:     [ ] 0%
  - RAG Integration:    [ ] 0%
  - Validation:         [ ] 0%

Next Sprints:
  - etcd-client:        [░] Waiting
  - Watcher:            [░] Waiting
  - Academic Paper:     [░] Waiting
```

---

**¡Base sólida completada! Próximo objetivo: FAISS Integration 🚀**

**Last Updated:** 16 Diciembre 2025  
**Next Review:** 17 Diciembre 2025 (Daily standup)  
**Major Milestone:** FAISS C++ Integration (ETA: 3-4 días)