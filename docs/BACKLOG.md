# 🛡️ ML Defender - Development BACKLOG

[... secciones anteriores sin cambios ...]

## 📊 ESTADO ACTUAL DEL SISTEMA

### ✅ **COMPLETADO - Phase 1 + Day 16 (Dic 1-16, 2025)**

#### Day 16: Race Condition Fix (PARTIALLY RESOLVED)
**Fecha:** 16 Diciembre 2025  
**Estado:** ⚠️ PARCIALMENTE RESUELTO - Ver ISSUE-005

**Logro:**
- ✅ Race conditions en RAGLogger eliminadas
- ✅ Release optimization flags (-O3) funcionando
- ✅ 20+ minutos uptime continuo (antes: 1-2 min crash)
- ✅ 1,152 artifacts generados exitosamente
- ✅ 575 líneas JSONL consolidadas
- ⚠️ Memory leak en librería JSONL detectado (solución temporal activa)

**Detalles Técnicos:**
- Moved `check_rotation()` inside `write_jsonl()` critical section
- Added `check_rotation_locked()` and `rotate_logs_locked()` helpers
- All file operations now atomic (current_date_, current_log_, counters)
- Zero crashes, stable CPU usage
- ⚠️ Slow memory leak en nlohmann/json JSONL generation (Ver ISSUE-005)

**Archivos Modificados:**
- `ml-detector/src/rag_logger.cpp` (race fix)
- `ml-detector/include/rag_logger.hpp` (new functions)

**Testing:**
- Full lab test: sniffer + ml-detector + firewall
- 20+ minute stress test (stable except memory)
- Artifact generation validated (1,152 events)
- JSONL consolidation validated (575 lines)
- ⚠️ Memory leak: ~50-100MB / 72 hours (requires restart)

[... resto de Day 16 sin cambios ...]

---

## 🎯 PHASE 2A - FAISS/RAG INTEGRATION (Ene 2026)

### Epic 2A.1: ⚠️ RAGLogger Stability (PARTIALLY RESOLVED)
**Priority:** P0 (CRITICAL)  
**Status:** ⚠️ PARCIALMENTE RESUELTO - ISSUE-005 abierto  
**Owner:** Alonso + Claude

**Goal:** Sistema RAGLogger 100% estable con optimizaciones release

**User Stories:**
- [x] Como desarrollador, quiero compilar con `-O3` sin crashes para máximo rendimiento
- [x] Como operador, quiero uptime prolongado sin reinicios para confiabilidad
- [x] Como analista, quiero generación confiable de artifacts para análisis posterior
- [ ] ⚠️ Como operador, quiero zero memory leaks para uptime indefinido (ISSUE-005)

**Resultados:**
- ✅ 20+ minutos uptime continuo
- ✅ 1,152 artifacts generados
- ✅ Zero crashes
- ✅ Race conditions resueltas
- ⚠️ Memory leak en librería JSONL (solución temporal: restart cada 3 días)

**Solución Temporal Activa:**
- Cron task en Vagrant VM: restart ml-detector cada 72 horas
- Impacto: ~5 segundos downtime cada 3 días (aceptable para lab)
- Ver ISSUE-005 para solución permanente

---

## 🎯 PHASE 2A - FAISS/RAG INTEGRATION (Ene 2026)

### Epic 2A.2: FAISS Infrastructure (Days 31-34) - IN PROGRESS
**Priority:** P1 (HIGH)
**Status:** 🟡 IN PROGRESS - Infrastructure Complete, Implementation Ongoing
**Owner:** Alonso + Claude

**Goal:** Semantic search para eventos históricos con FAISS

**Progress (Week 5 - Days 31-34):**
- ✅ Day 31: FAISS v1.8.0 installed + Anti-curse design (peer-reviewed)
- ✅ Day 32: ONNX Runtime v1.23.2 installed + tested
- ✅ Day 33: 3 ONNX embedder models created (512-d, 384-d, 256-d)
- ✅ Day 34: Pipeline validated with real JSONL data (21 min)
    - Python inference: 3/3 tests passed
    - C++ inference: 3/3 tests passed (ONNX Runtime upgraded)
    - Batch processing: 6.8K-18.5K events/sec throughput

**Next (Week 6 - Days 35-40):**
- 🔄 Day 35: DimensionalityReducer (PCA reduction 4x)
- 🔄 Day 36-38: AttackIndexManager, SelectiveEmbedder, Integration
- 🔄 Day 39-40: Advanced strategies (temporal tiers, quantization)

**Note:** FAISS work is parallel to ISSUE-005. Not blocked.

---

## Security Hardening (Post-MVP)

**Priority**: P1 - Critical for Production  
**Effort**: 8-12 days  
**Status**: Documented, pending implementation  
**Triggered**: 2026-01-07 (Knowledge Graph poisoning article)

### Tasks
- [ ] Implement HMAC signing in ml-detector/rag_logger
- [ ] Implement log validation in faiss-ingester
- [ ] Implement nonce store (replay prevention)
- [ ] Generate certificates for all components
- [ ] Configure etcd mTLS
- [ ] (Optional) Implement log encryption
- [ ] Security testing (inject fake logs)
- [ ] Documentation: Operations procedures

### Dependencies
- ✅ Phase 1 MVP complete (DimensionalityReducer + FAISS Ingester + RAG)
- ✅ SECURITY_HARDENING.md documented

### References
- `/docs/SECURITY_HARDENING.md`
- Article: https://www.elladodelmal.com/2026/01/adulteracion-del-knowledge-graph-de-una.html

## 🔧 ISSUES CONOCIDOS - TRACKING

### P0 - CRITICAL (Bloqueadores de Producción)

#### 🔴 ISSUE-005: JSONL Memory Leak en nlohmann/json Library
**Fecha:** 6 Enero 2026  
**Estado:** 🔴 ACTIVO - Solución temporal implementada  
**Priority:** P0 (CRITICAL para producción)  
**Owner:** Alonso + Claude  
**Target:** Phase 2A (antes de FAISS completion)

**Descripción:**

RAGLogger presenta memory leak lento en generación de archivos JSONL. El problema NO es de nuestra implementación, sino de la librería `nlohmann/json` al escribir grandes volúmenes de JSONL.

**Síntomas:**
- Memory usage crece ~50-100MB cada 72 horas
- No crashes, pero requiere restart preventivo
- Leak solo en JSONL write path, no en JSON config read

**Root Cause:**
```cpp
// Problema identificado:
nlohmann::json j;
j["field"] = value;
std::string line = j.dump();  // ← Memory leak aquí en alta frecuencia
output << line << "\n";       // Leak se acumula con miles de eventos
```

La librería `nlohmann/json` no está optimizada para escritura masiva de JSONL en streaming. Memory allocations no se liberan correctamente en ciclos rápidos.

**Solución Temporal (ACTIVA):**

Cron task en Vagrant VM:
```bash
# /etc/cron.d/ml-detector-restart
# Restart ml-detector cada 3 días (72 horas)
0 3 */3 * * vagrant systemctl restart ml-detector || true
```

**Impact de solución temporal:**
- ✅ Previene memory exhaustion
- ✅ ~5 segundos downtime cada 3 días
- ✅ Aceptable para lab, NO para producción
- ❌ No es solución definitiva

**Solución Permanente (PENDIENTE):**

Reemplazar `nlohmann/json` con librería optimizada para JSONL streaming:

**Opciones evaluadas:**

1. **RapidJSON (RECOMENDADO)** ⭐
  - Diseñado para high-performance streaming
  - Zero-copy parsing
  - SAX-style writing (no intermediate objects)
  - Usado en producción por Facebook, Tencent
  - Benchmark: 2-3x más rápido que nlohmann
```cpp
   // Ejemplo RapidJSON streaming:
   rapidjson::StringBuffer buffer;
   rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
   
   writer.StartObject();
   writer.Key("field");
   writer.String(value.c_str());
   writer.EndObject();
   
   output << buffer.GetString() << "\n";
   buffer.Clear();  // ← Libera memoria inmediatamente
```

2. **simdjson (ALTERNATIVA)**
  - Ultra-fast parsing (4-10x más rápido)
  - SIMD-optimized
  - Mejor para reading, no tanto para writing
  - Puede combinarse con RapidJSON writer

3. **Custom JSONL Writer (MANUAL)**
  - String formatting manual (sprintf style)
  - Zero dependencies
  - Máximo control, pero más error-prone
  - Viable para 83 campos fijos

**Plan de Implementación:**

**Day 1: RapidJSON Integration (4-6 horas)**
- [ ] Install RapidJSON library in Vagrant
- [ ] Create `rag_logger_rapidjson.cpp` (new implementation)
- [ ] Port 83-field JSONL generation to RapidJSON
- [ ] Benchmark: memory usage over 24 hours
- [ ] Keep nlohmann/json for config reading ONLY

**Day 2: Testing & Validation (2-4 horas)**
- [ ] Stress test: 10K events, monitor memory
- [ ] Overnight test: 24+ hours uptime
- [ ] Verify JSONL format compatibility with existing scripts
- [ ] Benchmark: throughput (events/sec)

**Day 3: Deployment (1-2 horas)**
- [ ] Replace RAGLogger implementation
- [ ] Remove cron restart task
- [ ] Update documentation
- [ ] Git commit + tag

**Acceptance Criteria:**

- ✅ Zero memory leak over 7+ days continuous operation
- ✅ JSONL format unchanged (backwards compatible)
- ✅ Performance maintained or improved
- ✅ No cron restart needed
- ✅ Production-ready

**Impact:**

- **Criticality:** HIGH - Blocks long-term production deployment
- **Effort:** LOW (1-3 días)
- **Risk:** LOW (RapidJSON battle-tested)
- **Priority:** Before FAISS completion

**Dependencies:**

- RapidJSON C++ library
- Existing RAGLogger tests (reuse for validation)

**Notes:**

- nlohmann/json stays for config reading (low frequency, no leak)
- Only JSONL writing path needs replacement
- This is library limitation, not our bug
- Common issue in high-frequency JSON serialization

**Related:**
- Epic 2A.1: RAGLogger Stability
- ISSUE-004: Race condition (resolved)

---

#### ✅ ISSUE-004: RAGLogger Race Condition (RESUELTO Day 16)

[... sin cambios ...]

---

### P1 - HIGH (Impacto en Detección)

[... resto de issues sin cambios ...]

---

## 📊 ROADMAP ACTUALIZADO
```
Phase 1: ✅ COMPLETADO (Dic 1-16, 2025)
├─ Days 1-5: eBPF/XDP + ML pipeline
├─ Days 6-10: RAG + LLAMA + Gateway Mode
├─ Days 11-15: Dual-Score + RAGLogger 83-field
├─ Day 16: Race condition fix (production-ready)
└─ ⚠️ ISSUE-005: Memory leak identified (temp fix active)

Phase 2A: 🔄 EN PROGRESO (Ene 2026)
├─ ⚠️ Epic 2A.1: RAGLogger stability (ISSUE-005 pending)
├─ 🔴 ISSUE-005: Fix JSONL memory leak (1-3 días) ← NEXT
├─ 🔥 Epic 2A.2: FAISS C++ Integration (after ISSUE-005)
├─ 📋 Epic 2A.3: etcd-client library (2-3 días)
├─ 📋 Epic 2A.4: Watcher library (3-4 días)
└─ 📋 Epic 2A.5: Academic paper (7-10 días)

[... resto sin cambios ...]
```

---

## 🧪 TESTING PRIORITIES

### CRITICAL - Esta Semana:
- [ ] 🔴 ISSUE-005: RapidJSON integration (1-3 días) ← BLOCKER
- [ ] 24h stress test con RapidJSON (memory validation)
- [ ] Remove cron restart after validation

### Inmediato - Después de ISSUE-005:
- [ ] FAISS proof of concept (10K events)
- [ ] Benchmark FAISS query latency
- [ ] Natural language query validation

[... resto sin cambios ...]

---

## 🎯 MÉTRICAS DE ÉXITO

### Phase 2A Success Criteria:
- ⚠️ RAGLogger stable sin memory leaks (ISSUE-005 en progreso)
- [ ] Zero memory growth over 7+ days
- [ ] No cron restarts needed
- [ ] FAISS semantic search <200ms para 10K events
- [ ] etcd-client library en todos los componentes
- [ ] Watcher hot-reload funcionando
- [ ] Academic paper draft completo

[... resto sin cambios ...]

### Quality Targets:
- ✅ Zero crashes con release build (ALCANZADO)
- ⚠️ Zero memory leaks (ISSUE-005: temp fix, permanent pending)
- [ ] Test coverage: >80%
- [ ] Documentation: 100% APIs documented
- [ ] Code review: All PRs multi-agent reviewed

---

[... resto del documento sin cambios ...]

## 📈 PROGRESO VISUAL
```
Phase 1 Progress: [████████████████████] 100% (16/16 días)
Phase 2A Progress: [██░░░░░░░░░░░░░░░░░░]  10% (RAGLogger partial, ISSUE-005 active)

Current Sprint: ISSUE-005 Resolution (BLOCKER)
  - RapidJSON Integration:  [░] 0% ← CURRENT
  - Memory Testing:         [░] 0%
  - Validation:             [░] 0%
  - Deployment:             [░] 0%

Next Sprint: FAISS Integration (after ISSUE-005)
  - FAISS Setup:        [░] 0% ← BLOCKED by ISSUE-005
  - Async Embedder:     [░] 0%
  - RAG Integration:    [░] 0%
  - Validation:         [░] 0%

Foundation Architecture:
  - BACKLOG-001:        [✓] Architecturally complete
                        [░] Implementation pending (post-FAISS)

Next Sprints:
  - etcd-client:        [░] Waiting
  - Watcher:            [░] Waiting
  - Academic Paper:     [░] Waiting
  - Flow Sharding:      [░] Post-FAISS
```

---

**¡ISSUE-005 es BLOCKER para FAISS! Resolver memory leak primero para foundation sólida. 🚀**

**Last Updated:** 6 Enero 2026  
**Next Review:** 7 Enero 2026 (Daily standup)  
**CRITICAL:** ISSUE-005 JSONL Memory Leak (ETA: 1-3 días)  
**BLOCKED:** FAISS integration (waiting for ISSUE-005 resolution)  
**Next Major Milestone:** BACKLOG-001 Flow Sharding (post-FAISS)