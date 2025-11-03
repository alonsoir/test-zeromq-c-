---

# 🏥 Hospital Ransomware Detection System - BACKLOG

**Última actualización:** 1 Noviembre 2025
**Proyecto:** Sistema de Detección de Ransomware Hospitalario
**Fase actual:** Phase 1B - Integration

---

## 🚨 PRIORIDADES

**P0 (CRITICAL):** Bloqueadores de producción - resolver ASAP
**P1 (HIGH):** Impacto significativo en detección - resolver en 1-2 semanas
**P2 (MEDIUM):** Mejoras importantes - resolver en 1 mes
**P3 (LOW):** Nice-to-have - backlog para futuro

---

## 📊 ISSUES CONOCIDOS

### P0 - CRITICAL (Bloqueadores)

> *Actualmente ninguno - Phase 1A completada al 67%*

---

### P1 - HIGH (Impacto en Detección)

#### 🔴 ISSUE-001: Buffer Payload Limitado a 96 Bytes

**Fecha:** 30 Oct 2025
**Impacto:** Alto - puede perder información crítica de DNS/HTTP
**Descripción:**
El buffer de payload en `SimpleEvent` está hardcodeado a 96 bytes:

```c
// sniffer.bpf.c
struct SimpleEvent {
    // ...
    __u8 payload[96];  // ← LIMITACIÓN
};
```

**Problemas identificados:**

* ✅ DGA domains pueden ser >50 caracteres
* ✅ HTTP headers con C&C info pueden exceder 96 bytes
* ✅ DNS TXT records (exfiltración) pueden ser >200 bytes
* ✅ Familias de ransomware varían en tamaño de payload

**Impacto en features:**

* `dns_query_entropy`: Puede calcular entropy sobre domain truncado
* `http_header_anomaly`: No captura headers completos
* `data_exfiltration_bytes`: Subestima volumen real

**Estrategias propuestas:**

**Opción A: Buffer Fijo Mayor (Quick Win)**

```c
__u8 payload[256];  // o 512 bytes
```

* ✅ Pros: Fácil, rápido, funciona para 90% casos
* ❌ Contras: Desperdicia memoria si packet es pequeño
* **Recomendación:** Implementar PRIMERO (Phase 2)

**Opción B: Multiple Ring Buffers por Tamaño**

```c
// Small packets (<128B) → ring_buffer_small
// Medium packets (128-512B) → ring_buffer_medium  
// Large packets (>512B) → ring_buffer_large
```

* ✅ Pros: Eficiente en memoria, escalable
* ❌ Contras: Complejidad en kernel, 3 consumers en userspace
* **Recomendación:** Phase 3 si Opción A no es suficiente

**Opción C: Payload Hash + Deduplication**

```c
struct SimpleEvent {
    __u64 payload_hash;      // Blake2b hash
    __u16 payload_size_full; // Tamaño real (puede ser >96)
    __u8 payload[96];        // Primeros 96 bytes
};
```

* ✅ Pros: Detecta payloads únicos sin almacenar todo
* ❌ Contras: No reconstruye payload completo
* **Recomendación:** Complemento a Opción A

**Opción D: Payload Dinámico con BPF_MAP_TYPE_PERF_EVENT_ARRAY**

```c
// Payload variable usando perf event array
bpf_perf_event_output(ctx, &events, flags, &event, sizeof(event));
```

* ✅ Pros: Payloads de tamaño arbitrario
* ❌ Contras: Mayor complejidad, overhead en kernel
* **Recomendación:** Phase 4 (optimización avanzada)

**Plan de acción:**

1. **Phase 2:** Implementar Opción A (256-512 bytes fijos)
2. **Phase 3:** Evaluar con datos reales si necesitamos Opción B o C
3. **Phase 4:** Considerar Opción D si el volumen de tráfico lo justifica

**Asignado:** Backlog
**Target:** Phase 2 (post-MVP)

---

#### 🔴 ISSUE-002: DNS Entropy Test Fallando (Esperado >6.0, Actual 3.64)

**Fecha:** 31 Oct 2025
**Impacto:** Medio - falso negativo en detección DGA
**Descripción:**
El test de DNS entropy malicioso falla porque los dominios sintéticos no son suficientemente random:

```cpp
// Test actual (demasiado estructurado)
"xjf8dk2jf93.com"  // Entropy: 3.64
"9fj3kd8s2df.com"

// DGA real (más random)
"ajkdh3kdjf93kdjf83kdnf83kd.com"  // Entropy esperada: >6.0
```

**Causa raíz:**

* Dominios de test tienen longitud fija ~15 caracteres
* Mezcla predecible de números y letras
* DGA reales usan dominios más largos (30-60 chars) con distribución uniforme

**Plan de acción:**

1. Generar dominios con `std::mt19937` (random uniforme)
2. Longitud variable 20-50 caracteres
3. Solo lowercase + números (como DGA real)
4. Validar entropy calculada vs Locky/TeslaCrypt conocidos

**Asignado:** Backlog
**Target:** Phase 2 (después de validar con tráfico real)

---

#### 🔴 ISSUE-003: SMB Diversity Counter Retorna 0 (Esperado >5)

**Fecha:** 31 Oct 2025
**Impacto:** Alto - falso negativo en lateral movement
**Descripción:**
El test de SMB diversity malicioso retorna 0 cuando debería contar 15 destinos únicos:

```cpp
// Test inyecta 15 eventos SMB a IPs diferentes
for (int i = 1; i <= 15; i++) {
    TimeWindowEvent event(src_ip, target_ip, port, 445, TCP, size);
    extractor.add_event(event);
}

// Resultado: smb_diversity = 0 (❌ debería ser 15)
```

**Causa probable:**

* Bug en `extract_smb_connection_diversity()`
* Eventos no se están agregando al `TimeWindowAggregator`
* Filtro de puerto 445 no funciona correctamente

**Plan de acción:**

1. Añadir logging en `add_event()` para verificar recepción
2. Debuggear `extract_smb_connection_diversity()` con GDB
3. Validar que `dst_port == 445` se detecta correctamente
4. Testear con PCAP real de lateral movement (Mimikatz + PsExec)

**Asignado:** Backlog
**Target:** Phase 2 (crítico para detección)

---

### P2 - MEDIUM (Mejoras Importantes)

#### 🟡 ISSUE-004: Falta Integración con main.cpp

**Fecha:** 31 Oct 2025
**Descripción:** Phase 1B pendiente - integrar `RansomwareFeatureProcessor` en el sniffer principal
**Target:** Phase 1B (HOY)

---

#### 🟡 ISSUE-005: Sin Serialización Protobuf

**Fecha:** 31 Oct 2025
**Descripción:** Features extraídas no se serializan a protobuf para envío
**Target:** Phase 1B (HOY)

---

#### 🟡 ISSUE-006: Sin Envío ZMQ

**Fecha:** 31 Oct 2025
**Descripción:** Features no se envían al `ml-detector` por ZMQ
**Target:** Phase 1B (HOY)

---

#### 🟡 ISSUE-007: Timer de Extracción Hardcodeado (30s)

**Fecha:** 31 Oct 2025
**Descripción:** El intervalo de extracción está hardcodeado, debería ser configurable por JSON
**Target:** Phase 2

---

#### 🟡 ISSUE-008: Sin Whitelist de IPs Internas

**Fecha:** 30 Oct 2025
**Descripción:** `IPWhitelist` cuenta TODAS las IPs externas, debería filtrar IPs confiables (Google DNS, CDNs, etc.)
**Target:** Phase 2

---

### P3 - LOW (Nice-to-Have)

#### 🟢 ISSUE-009: DNS Parsing Usa Pseudo-Domain por IP

**Fecha:** 30 Oct 2025
**Descripción:** Si no hay payload DNS real, se genera `192-168-1-1.pseudo.dns` - es funcional pero no ideal
**Target:** Phase 3 (cuando buffer payload sea mayor)

---

#### 🟢 ISSUE-010: Sin Métricas de Performance

**Fecha:** 31 Oct 2025
**Descripción:** No hay métricas de latencia de extracción, throughput de features, CPU usage
**Target:** Phase 3

---

#### 🟢 ISSUE-011: Sin Dashboard de Monitoreo

**Fecha:** 31 Oct 2025
**Descripción:** Falta dashboard para visualizar features en tiempo real (Grafana/Prometheus)
**Target:** Phase 4

---

#### 🟢 ISSUE-012: Extensión Firmada de Tabla de Protocolos (Futuro)

**Fecha:** 1 Nov 2025
**Impacto:** Bajo (mejora arquitectónica, no bloqueante)
**Descripción:**
Actualmente, la tabla de protocolos (`protocol_numbers.hpp`) está **compilada y embebida a fuego**, garantizando integridad, rendimiento y seguridad en producción.
En futuras versiones puede ser útil permitir la **extensión controlada y firmada** de esta tabla (por ejemplo, si IANA introduce nuevos números de protocolo o se requieren protocolos internos).

**Requisitos de diseño:**

* Tabla compilada permanece **inmutable por defecto**.
* Extensión opcional mediante JSON **firmado con ECDSA**.
* Verificación previa: hash y firma válidos.
* Fallback seguro: si la verificación falla, se ignora la extensión.
* Compatible con el `constexpr` map original (`protocol_number_to_string`).

**Plan de acción:**

1. Fase actual: mantener compilado (✅ filosofía *smooth is fast*).
2. Fase 3-4: diseñar prototipo `ProtocolMapExtender` con validación de firmas.
3. Fase 4+: integrar en modo “opt-in” para producción controlada.

**Asignado:** Backlog
**Target:** Phase 4 (2026)
**Estado:** Pendiente de decisión de arquitectura (Claude + Alonso + GPT review)

---

## 📈 ROADMAP SUGERIDO

```
Phase 1A: ✅ COMPLETADO (31 Oct 2025)
├─ Componentes compilados (6/6)
├─ Tests unitarios (67% passing)
└─ Binary optimizado (877KB)

Phase 1B: ⏳ EN PROGRESO (1 Nov 2025)
├─ Integración main.cpp
├─ Timer thread (30s)
├─ Serialización protobuf
└─ Envío ZMQ

Phase 2: 🔜 SIGUIENTE (Nov 2025)
├─ ISSUE-001: Buffer payload 256-512 bytes
├─ ISSUE-002: Fix DNS entropy test
├─ ISSUE-003: Fix SMB diversity counter
├─ ISSUE-008: Whitelist de IPs confiables
└─ Testing con tráfico real (PCAP replay)

Phase 3: 📋 BACKLOG (Dic 2025)
├─ DNS parsing mejorado
├─ Métricas de performance
├─ Multiple ring buffers (si necesario)
└─ Optimizaciones AVX2/SIMD

Phase 4: 🎯 FUTURO (2026)
├─ Dashboard Grafana
├─ ML model integration
├─ A/B testing de thresholds
├─ Auto-tuning de parámetros
└─ Extensión firmada de tabla de protocolos
```
# Product Backlog - Enhanced Network Sniffer

## 🎯 Vision
Build an enterprise-grade, ML-powered network security monitoring system with real-time ransomware detection capabilities.

---

## ✅ DONE - Phase 1: Foundation & Detection (Nov 1, 2025)

### Sprint 1A: Protocol Numbers Standardization ✅
- [x] Create protocol_numbers.hpp with IANA standards
- [x] Implement IPProtocol enum with 30+ protocols
- [x] Add protocol_to_string() helper functions
- [x] Replace all magic numbers in codebase
- [x] Update feature_logger, flow_tracker, ransomware_feature_processor

### Sprint 1B: FastDetector Implementation ✅
- [x] Design FastDetector class with 4 heuristics
- [x] Implement 10-second sliding window
- [x] Add thread_local storage for zero contention
- [x] Create comprehensive test suite (5 tests)
- [x] Validate <1μs latency per event

### Sprint 1C: Two-Layer Integration ✅
- [x] Integrate FastDetector in RingBufferConsumer
- [x] Implement send_fast_alert() with protobuf
- [x] Implement send_ransomware_features() with protobuf
- [x] Add statistics tracking (3 new counters)
- [x] Fix namespace and compilation issues

### Sprint 1D: Live Traffic Validation ✅
- [x] Run sniffer with real network traffic
- [x] Generate 150+ alerts over 271 seconds
- [x] Validate zero crashes and memory leaks
- [x] Measure performance (229μs avg processing)
- [x] Test graceful shutdown

### Sprint 1E: Documentation ✅
- [x] Update README.md with architecture
- [x] Create CHANGELOG.md with version history
- [x] Document all features and usage
- [x] Add performance metrics

**Status:** ✅ COMPLETE - MVP validated with live traffic

---

## 🔄 IN PROGRESS - Phase 2: Enhanced Detection

### Epic 2.1: Payload Analysis (Priority: HIGH)
**Goal:** Analyze first 512 bytes of each packet for PE headers, encryption patterns, suspicious strings

**User Stories:**
- [ ] As a security analyst, I want PE header detection so I can identify ransomware executables in transit
- [ ] As a SOC operator, I want encryption pattern recognition so I can detect file encryption activity
- [ ] As a threat hunter, I want string-based heuristics so I can catch ransomware ransom notes

**Tasks:**
- [ ] Extend SimpleEvent with `uint8_t payload[512]`
- [ ] Implement PE header parser (DOS stub, PE signature)
- [ ] Add entropy calculation for payload
- [ ] Create string matching for common ransom note patterns
- [ ] Write unit tests for payload analysis
- [ ] Integrate with FastDetector

**Acceptance Criteria:**
- Payload captured in eBPF without performance impact
- PE headers detected with >95% accuracy
- Entropy calculation <10μs per packet
- Zero false positives on benign executables

**Estimated Effort:** 2-3 days

---

### Epic 2.2: Threshold Tuning (Priority: MEDIUM)
**Goal:** Reduce false positives by tuning detection thresholds based on real traffic patterns

**User Stories:**
- [ ] As a security analyst, I want configurable thresholds so I can adapt to my network
- [ ] As a SOC operator, I want baseline learning so the system adapts to normal behavior
- [ ] As a DevOps engineer, I want threshold persistence so settings survive restarts

**Tasks:**
- [ ] Add threshold configuration to sniffer.json
- [ ] Implement dynamic threshold adjustment
- [ ] Create baseline learning mode (7-day window)
- [ ] Add whitelist for known-good DNS patterns
- [ ] Persist learned baselines to disk
- [ ] Create threshold tuning dashboard

**Acceptance Criteria:**
- False positive rate <5% on production traffic
- Thresholds configurable per network environment
- Baseline learning converges in 7 days
- Settings persist across restarts

**Estimated Effort:** 3-4 days

---

## 📋 BACKLOG - Phase 3: ML Integration

### Epic 3.1: Random Forest Model Integration
**Goal:** Deploy trained RF model (8 features, 98.61% accuracy) for real-time inference

**User Stories:**
- [ ] As a data scientist, I want model loading so I can deploy trained models
- [ ] As a security analyst, I want real-time predictions so I can respond immediately
- [ ] As a MLOps engineer, I want model versioning so I can A/B test models

**Tasks:**
- [ ] Create model loader (ONNX or TensorFlow Lite)
- [ ] Implement inference pipeline
- [ ] Add feature vector construction
- [ ] Integrate with RansomwareFeatureProcessor
- [ ] Create model versioning system
- [ ] Add A/B testing framework
- [ ] Implement model performance metrics

**Acceptance Criteria:**
- Inference latency <100μs per event
- Model accuracy ≥98% on validation set
- Support for multiple concurrent models
- Zero-downtime model updates

**Estimated Effort:** 5-7 days

---

### Epic 3.2: Feature Pipeline Optimization
**Goal:** Optimize feature extraction for ML model compatibility

**User Stories:**
- [ ] As a data scientist, I want normalized features so models perform optimally
- [ ] As a system architect, I want feature caching so we avoid redundant calculations
- [ ] As a performance engineer, I want batch processing so we maximize throughput

**Tasks:**
- [ ] Implement feature normalization (z-score, min-max)
- [ ] Add feature caching layer
- [ ] Create batch processing for ML features
- [ ] Optimize memory layout for SIMD
- [ ] Add feature importance tracking
- [ ] Create feature drift detection

**Acceptance Criteria:**
- Feature extraction <500μs for 83 features
- Cache hit rate >80%
- SIMD optimizations applied
- Feature drift alerts when distribution changes

**Estimated Effort:** 4-5 days

---

## 📋 BACKLOG - Phase 4: Production Readiness

### Epic 4.1: Containerization & Orchestration
**Goal:** Package sniffer as container for easy deployment

**Tasks:**
- [ ] Create Dockerfile with multi-stage build
- [ ] Create Kubernetes manifests
- [ ] Implement health checks
- [ ] Add liveness/readiness probes
- [ ] Create Helm chart
- [ ] Document deployment procedures

**Estimated Effort:** 3-4 days

---

### Epic 4.2: Observability & Monitoring
**Goal:** Comprehensive monitoring and alerting

**Tasks:**
- [ ] Implement Prometheus metrics exporter
- [ ] Create Grafana dashboards
- [ ] Add distributed tracing (Jaeger)
- [ ] Implement structured logging
- [ ] Create alert rules
- [ ] Add SLO/SLI tracking

**Estimated Effort:** 4-5 days

---

### Epic 4.3: Alert Management System
**Goal:** Intelligent alert aggregation and routing

**Tasks:**
- [ ] Implement alert deduplication
- [ ] Create severity classification
- [ ] Add alert routing (email, Slack, PagerDuty)
- [ ] Implement alert suppression rules
- [ ] Create incident correlation
- [ ] Add playbook automation

**Estimated Effort:** 5-7 days

---

## 🧊 ICE BOX - Future Ideas

### Nice-to-Have Features
- [ ] Web-based UI for real-time monitoring
- [ ] Packet replay for offline analysis
- [ ] Integration with SIEM systems
- [ ] Support for encrypted traffic analysis (TLS inspection)
- [ ] Multi-tenancy support
- [ ] Distributed deployment across multiple nodes
- [ ] Automatic threat intelligence feed integration
- [ ] Behavioral anomaly detection
- [ ] Custom rule engine for SOC analysts
- [ ] Cloud storage integration (S3, Azure Blob)

---

## 📊 Metrics & KPIs

### Performance Targets
- Throughput: >1M events/sec
- Latency: <1ms p99
- CPU usage: <50% at peak load
- Memory: <2GB resident set size

### Detection Targets
- True positive rate: >95%
- False positive rate: <5%
- Time to detect: <30s
- Model accuracy: >98%

---

## 🎯 Current Sprint

**Sprint Goal:** Enhanced Detection - Payload Analysis

**Duration:** Nov 2-15, 2025

**Capacity:** 40 story points

**Sprint Backlog:**
1. Epic 2.1: Payload Analysis (Priority: HIGH)
    - Extend SimpleEvent with payload buffer
    - Implement PE header detection
    - Add encryption entropy calculation
    - Create comprehensive tests

---

**Last Updated:** November 1, 2025  
**Next Review:** November 15, 2025

---

## 🏥 NOTAS DE DESARROLLO

**Filosofía:** "Smooth is fast. Via Appia no se construyó en un día."

**Prioridades:**

1. ✅ Sistema funcional > Sistema perfecto
2. ✅ Detección en producción > Tests al 100%
3. ✅ Salud del desarrollador > Deadlines
4. ✅ Código de calidad > Velocidad

**Cada línea de código protege vidas reales.**

---

## 📞 CONTACTO Y SEGUIMIENTO

* **Owner:** Hospital Security Team
* **Lead Developer:** Alonso Isidoro Román — [alonsoir@gmail.com](mailto:alonsoir@gmail.com)
* **Review:** Semanal (Viernes)
* **Docs:** `/vagrant/STATUS.md`, `/vagrant/BACKLOG.md`


