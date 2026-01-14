¡Claro! Aquí está el **BACKLOG actualizado completo**:

```markdown
# RAG Ingester - Development Backlog

**Last Updated:** 2026-01-14 (Day 38 Parcial)  
**Current Phase:** 2A - Foundation  
**Next Session:** Day 38 Completion - Execute Generator + ONNX Embedders

---

## 🔒 CRITICAL SECURITY DECISION: Mandatory Encryption

**ADR-001: Encryption is NOT Optional**

**Decision:** Encryption and compression are HARDCODED in the pipeline, NOT configurable.

**Rationale:**
- **Poison Log Prevention:** Attacker could disable encryption to inject malicious events
- **Data Integrity:** Compressed + encrypted data has built-in tamper detection
- **Compliance:** Enterprise security requires encryption at rest
- **No Backdoors:** No "debug mode" that bypasses security

**Implementation (COMPLETED Day 37):**
```cpp
// ml-detector (rag_logger.cpp)
void RAGLogger::save_artifacts(const NetworkSecurityEvent& event,
                               const nlohmann::json& json_record) {
    // Serialize
    std::string serialized;
    event.SerializeToString(&serialized);
    
    // Compress (ALWAYS - not configurable)
    auto compressed = crypto_manager_->compress_with_size(serialized);
    
    // Encrypt (ALWAYS - not configurable)
    auto encrypted = crypto_manager_->encrypt(compressed);
    
    // Write .pb.enc (encrypted extension)
    std::ofstream file(pb_path + ".enc", std::ios::binary);
    file.write(encrypted.data(), encrypted.size());
}

// rag-ingester (event_loader.cpp)
Event EventLoader::load(const std::string& filepath) {
    auto encrypted = read_file(filepath);
    
    // Decrypt (ALWAYS - no fallback to plaintext)
    auto decrypted = crypto_->decrypt(encrypted);
    if (!decrypted) {
        throw SecurityException("Decryption failed - rejecting event");
    }
    
    // Decompress (ALWAYS)
    auto decompressed = crypto_->decompress(decrypted);
    
    return parse_protobuf(decompressed);
}
```

### 🔄 Day 38 - Synthetic Data + ONNX Embedders (2026-01-14 PARCIAL)

**Status:** 40% Complete (Generator compiled, execution pending)

**Completado:**
- [x] Tools infrastructure created (`/vagrant/tools/`)
- [x] `generate_synthetic_events.cpp` implemented (850 lines)
- [x] Config file: `synthetic_generator_config.json`
- [x] CMakeLists.txt: Protobuf + etcd-client linking fixed
- [x] Makefile integration: `make tools-build` working
- [x] Binary compiled successfully (2.5MB)
- [x] 100% compliance architecture (etcd + RAGLogger)

**Pendiente (Tomorrow Morning):**
- [ ] etcd-server setup with encryption_seed
- [ ] Execute generator (100-200 events)
- [ ] Verify .pb.enc artifacts generated
- [ ] Update ChronosEmbedder (103 → 512-d)
- [ ] Update SBERTEmbedder (103 → 384-d)
- [ ] Update AttackEmbedder (103 → 256-d)
- [ ] End-to-end smoke test

**Architecture Decisions:**
```
Via Appia Quality:
1. No hardcoded keys → Uses etcd (same as ml-detector)
2. Zero drift → Reuses production RAGLogger directly
3. Realistic data → 101 features + ADR-002 provenance
4. Security first → Encryption from etcd, not config
```

**Compilation Fixes Applied:**
- ❌ `-lnetwork_security_proto` → ✅ Compile `.pb.cc` directly
- ❌ Missing etcd symbols → ✅ Added `etcd_client.cpp`
- ❌ Missing OpenSSL/CURL → ✅ Added to CMakeLists
- ❌ Reason code enums → ✅ Use strings directly

**Success criteria:**
- ✅ Generator compiles cleanly
- ⏳ 100+ .pb.enc files with provenance
- ⏳ Embedders process 103 features
- ⏳ End-to-end pipeline functional

**Via Appia Milestones:**
- 🏛️ Reuse over reinvent: Production RAGLogger used directly
- 🏛️ Security by design: etcd integration, no hardcoded secrets
- 🏛️ Test tools = production quality

---

## 🐛 TECHNICAL DEBT REGISTER

### ISSUE-008: etcd-server encryption_seed Bootstrap (NEW - Day 38)

**Severity:** Medium  
**Impact:** Generator can't run without encryption_seed  
**Discovered:** 2026-01-14 (Day 38)

**Problem:**
- Generator requires `/crypto/ml-detector/tokens/encryption_seed` in etcd
- Currently manual setup required
- No bootstrap script exists

**Solution:**
Create `/vagrant/scripts/bootstrap_etcd_encryption.sh`:
```bash
#!/bin/bash
# Generate and store encryption seed in etcd
SEED=$(openssl rand -hex 32)
ETCDCTL_API=3 etcdctl put /crypto/ml-detector/tokens/encryption_seed $SEED
echo "✅ Encryption seed created: ${SEED:0:16}..."
```

**Affected Components:**
- generate_synthetic_events
- ml-detector (also requires this key)

**Estimación:** 15 minutes  
**Priority:** HIGH (blocks Day 38 execution)  
**Assigned:** Tomorrow morning

---
## 📚 KEY DOCUMENTS

### Day 38 Files (NEW)
- `/vagrant/tools/generate_synthetic_events.cpp` - Synthetic data generator (850 lines)
- `/vagrant/tools/config/synthetic_generator_config.json` - Generator config
- `/vagrant/tools/CMakeLists.txt` - Build system (corrected)
- `/vagrant/tools/build/generate_synthetic_events` - Compiled binary
**Status:** ✅ IMPLEMENTED (Day 37)

---

## 🔍 ADR-002: Multi-Engine Provenance & Situational Intelligence

**Date:** 13 Enero 2026  
**Status:** ✅ IMPLEMENTED  
**Decision:** Extend protobuf contract to capture multiple engine verdicts  
**Proposer:** Gemini (peer reviewer)  
**Implementation:** Day 37 COMPLETE

---

### Context

Previously, the protobuf contract only stored the **final classification** result. This discarded valuable information:

- **Sniffer verdict:** Fast-path decision (eBPF/XDP)
- **RandomForest verdict:** ML primary classification
- **CNN verdict:** ML secondary classification (if enabled)

**The discrepancy between engines IS the signal** for detecting:
- APTs (Advanced Persistent Threats) camouflaged as benign traffic
- 0-days (unknown attacks)
- False positives (industrial equipment with unusual patterns)

### Decision

Add `DetectionProvenance` message to protobuf contract with:
1. **Array of verdicts** - All engine opinions
2. **Reason codes** - WHY each engine decided
3. **Discrepancy score** - Measure of disagreement
4. **Final decision** - What action was taken

### Protobuf Extension (IMPLEMENTED)

```protobuf
// network_security.proto

message EngineVerdict {
  string engine_name = 1;      // "fast-path-sniffer", "random-forest", "cnn-secondary"
  string classification = 2;   // "Benign", "DDoS", "Ransomware", etc.
  float confidence = 3;        // 0.0 - 1.0
  string reason_code = 4;      // WHY (see table below)
  uint64 timestamp_ns = 5;     // When this engine decided
}

message DetectionProvenance {
  repeated EngineVerdict verdicts = 1;  // ALL opinions
  uint64 global_timestamp_ns = 2;       // When event was logged
  string final_decision = 3;            // "ALLOW", "DROP", "ALERT"
  float discrepancy_score = 4;          // 0.0 (agree) - 1.0 (disagree)
  string logic_override = 5;            // If human/RAG forced decision
  string discrepancy_reason = 6;        // Explanation if engines disagree
}

// Add to NetworkSecurityEvent:
message NetworkSecurityEvent {
  // ... existing fields (101 features) ...
  
  DetectionProvenance provenance = 35;  // NEW
}
```

### Reason Codes (Gemini's Table)

**Created:** `/vagrant/common/include/reason_codes.hpp`

| Code | Meaning | Value for RAG/LLM |
|------|---------|-------------------|
| `SIG_MATCH` | Exact signature match (blacklist) | **Immediate block priority** |
| `STAT_ANOMALY` | Statistical deviation (Z-score high) | Unusual behavior, not necessarily malicious |
| `PCA_OUTLIER` | Vector outside normal cloud (latent space) | **Critical for 0-day detection** 🎯 |
| `PROT_VIOLATION` | Protocol malformation (impossible TCP flags) | Low-level technical attack |
| `ENGINE_CONFLICT` | Sniffer vs ML disagree significantly | **High-priority alert for LLM analysis** 🚨 |

### Use Cases

**Case 1: APT Camouflaged**
```
Sniffer: "Benign" (95%) - reason: SIG_MATCH (whitelist)
RF:      "Exfiltration" (55%) - reason: STAT_ANOMALY (timing)
CNN:     "Suspicious" (62%) - reason: PCA_OUTLIER (latent)

→ discrepancy_score: 0.85 (HIGH)
→ reason_code: ENGINE_CONFLICT + PCA_OUTLIER
→ RAG flags for human review
→ LLM: "Sniffer fooled by TLS, but timing+PCA suspicious"
→ Action: ALERT + monitor
```

**Case 2: Industrial False Positive**
```
Sniffer: "Benign" (99%) - reason: SIG_MATCH (known IoT)
RF:      "Anomaly" (40%) - reason: STAT_ANOMALY (Z-score)
CNN:     "Benign" (90%) - reason: (normal pattern)

→ discrepancy_score: 0.35 (MEDIUM)
→ Campus A: LLM learns "RF always alerts on this equipment"
→ Action: Auto-whitelist campus A, monitor campus B
```

**Case 3: 0-Day Detection**
```
Sniffer: "Benign" (80%) - reason: (no signature)
RF:      "Unknown" (50%) - reason: STAT_ANOMALY
CNN:     "Malicious" (70%) - reason: PCA_OUTLIER (never seen)

→ discrepancy_score: 0.75 (HIGH)
→ reason_code: PCA_OUTLIER (critical!)
→ RAG: "No engine is sure, but CNN detects novelty"
→ Action: Escalate to human (possible 0-day)
→ If confirmed → Global vaccine
```

### Implementation Status (Day 37 COMPLETE)

**Phase 1: ml-detector (writes provenance)** ✅
```cpp
void ZMQHandler::process_event(const std::string& message) {
    // ... existing classification logic ...
    
    // NEW: Fill provenance
    auto* provenance = event.mutable_provenance();
    
    // Check if sniffer already added verdict
    bool sniffer_verdict_exists = (provenance->verdicts_size() > 0);
    
    // Add RandomForest verdict
    auto* rf_verdict = provenance->add_verdicts();
    rf_verdict->set_engine_name("random-forest-level1");
    rf_verdict->set_classification(label_l1 == 0 ? "Benign" : "Attack");
    rf_verdict->set_confidence(confidence_l1);
    rf_verdict->set_reason_code(
        label_l1 == 1 
            ? ml_defender::to_string(ml_defender::ReasonCode::STAT_ANOMALY)
            : ml_defender::to_string(ml_defender::ReasonCode::UNKNOWN)
    );
    rf_verdict->set_timestamp_ns(now_ns());
    
    // Calculate discrepancy
    float discrepancy = 0.0f;
    if (sniffer_verdict_exists) {
        const auto& sniffer_v = provenance->verdicts(0);
        discrepancy = std::abs(sniffer_v.confidence() - ml_score);
    }
    
    provenance->set_discrepancy_score(discrepancy);
    provenance->set_final_decision(final_score >= 0.70 ? "DROP" : "ALLOW");
    
    // ... rest of processing ...
}
```

**Phase 2: rag-ingester (reads provenance)** ✅
```cpp
Event EventLoader::parse_protobuf(const std::vector<uint8_t>& data) {
    // ... existing feature extraction ...
    
    // NEW: Parse provenance
    if (proto_event.has_provenance()) {
        const auto& prov = proto_event.provenance();
        
        for (int i = 0; i < prov.verdicts_size(); i++) {
            const auto& v = prov.verdicts(i);
            
            EngineVerdict verdict;
            verdict.engine_name = v.engine_name();
            verdict.classification = v.classification();
            verdict.confidence = v.confidence();
            verdict.reason_code = v.reason_code();
            verdict.timestamp_ns = v.timestamp_ns();
            
            event.verdicts.push_back(verdict);
        }
        
        event.discrepancy_score = prov.discrepancy_score();
        event.final_decision = prov.final_decision();
        
        // Log high discrepancy
        if (event.discrepancy_score > 0.30f) {
            std::cout << "[INFO] High discrepancy: " << event.event_id 
                      << " (score=" << event.discrepancy_score << ")" << std::endl;
        }
    }
    
    return event;
}
```

**Phase 3: Embedders use discrepancy** ⏳ (Day 38)
```cpp
std::vector<float> ChronosEmbedder::embed(const Event& event) {
    std::vector<float> input = event.features;  // 101 features
    
    // NEW: Add meta-features from ADR-002
    input.push_back(event.discrepancy_score);   // Feature 102
    input.push_back(event.verdicts.size());     // Feature 103
    
    // ONNX inference: 103 features → 512-d
    return run_inference(input);
}
```

### Benefits

1. **Situational Intelligence** - Not just "what" but "why" and "how sure"
2. **0-day Detection** - PCA_OUTLIER + ENGINE_CONFLICT = red flag
3. **Adaptive Learning** - LLM learns campus-specific patterns
4. **Reduced False Positives** - Context helps distinguish noise from threats
5. **Forensics** - Complete audit trail of all engine decisions
6. **Vaccine Quality** - Discrepancies help prioritize which events to analyze

## 📊 Success Metrics

### Phase 2A (Week 5)
- ✅ Compilation successful (Days 35-37)
- ✅ All tests passing (Days 35-37)
- ✅ Dependencies resolved (Days 35-37)
- ✅ Binary functional (Days 36-37)
- ✅ ADR-002 implemented (Day 37)
- ✅ ADR-001 hardened (Day 37)
- ✅ Generator compiled (Day 38 parcial) ← NEW
- [ ] Synthetic data generation (Day 38 completion)
- [ ] ONNX Embedders updated (Day 38 completion)
- [ ] End-to-end pipeline working (Day 40)

---
## 📈 Progress Visual
```
Phase 1:  [████████████████████] 100% COMPLETE
Phase 2A: [██████████░░░░░░░░░░]  50% (Days 35-38/40) ← Updated
Phase 2B: [░░░░░░░░░░░░░░░░░░░░]   0%
Phase 3:  [░░░░░░░░░░░░░░░░░░░░]   0%
```

**Day 38 Progress:**
- Structure:    [████] 100% ✅
- Compilation:  [████] 100% ✅ (NEW)
- Execution:    [░░░░]   0% ← Tomorrow
- Embedders:    [░░░░]   0% ← Tomorrow
- Integration:  [█░░░]  25%

---

## 🐛 TECHNICAL DEBT REGISTER

### ISSUE-007: Magic Numbers in ml-detector (NEW - Day 37)

**Severity:** Medium  
**Impact:** Code readability, maintainability  
**Discovered:** 2026-01-13 (Day 37)

**Problem:**
```cpp
// zmq_handler.cpp lines 332, 365
if (score_divergence > 0.30) {  // ← Magic number
    event.set_authoritative_source(protobuf::DETECTOR_SOURCE_DIVERGENCE);
}

event.set_final_classification(final_score >= 0.70 ? "MALICIOUS" : "BENIGN");  // ← Magic number
```

**Solution:**
Move to `ml_detector_config.json`:
```json
{
  "ml": {
    "thresholds": {
      "divergence_alert": 0.30,
      "classification_threshold": 0.70
    }
  }
}
```

**Affected Files:**
- `/vagrant/ml-detector/src/zmq_handler.cpp`
- `/vagrant/ml-detector/config/ml_detector_config.json`

**Estimación:** 30 minutes  
**Priority:** Medium (not blocking)  
**Assigned:** Day 39

---

### ISSUE-006: Log Files Not Persisted (NEW - Day 37)

**Severity:** Medium  
**Impact:** Operational monitoring  
**Discovered:** 2026-01-13 (Day 37)

**Problem:**
- Logs currently only to stdout
- Monitor scripts expect log files for `tail -f`
- No persistent logs for troubleshooting

**Solution:**
Configure spdlog with rotating file sinks:
```cpp
// main.cpp
auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
    "/vagrant/logs/ml-detector/ml-detector.log", 
    1024 * 1024 * 10,  // 10 MB
    3                   // 3 rotated files
);

auto logger = std::make_shared<spdlog::logger>("ml-detector", 
    spdlog::sinks_init_list{console_sink, file_sink});
```

**Affected Components:**
- ml-detector
- sniffer
- rag-ingester

**Estimación:** 1 hour (all components)  
**Priority:** Medium  
**Assigned:** Day 39

---

### ISSUE-005: RAGLogger Memory Leak (Known)

**Status:** Documented, pendiente  
**Impact:** Restart cada 3 días  
**Root Cause:** nlohmann/json allocations  
**Solution:** RapidJSON migration  
**Priority:** Medium  
**Estimación:** 2-3 days

---

### ISSUE-003: Thread-Local FlowManager Bug (Known)

**Status:** Documented, pendiente  
**Impact:** Solo 11/102 features capturadas  
**Workaround:** PCA entrenado con datos sintéticos  
**Solution:** Fix thread-local storage  
**Priority:** HIGH (pero no bloqueante para Day 38)  
**Estimación:** 1-2 days

---

### MISSING: Acceptance Tests for Protobuf Contract (NEW - Day 37)

**Severity:** HIGH  
**Impact:** Contract validation  
**Discovered:** 2026-01-13 (Day 37)

**Necesidad:**
- Validar nuevo contrato end-to-end
- Verificar que provenance se preserva: sniffer → ml-detector → rag-ingester
- Tests para todos los reason codes

**Solution:**
Create `/vagrant/tests/test_protobuf_contract.py`:
- test_provenance_structure()
- test_multiple_verdicts()
- test_reason_codes()
- test_end_to_end_preservation()

**Estimación:** 2 hours  
**Priority:** HIGH  
**Assigned:** Day 38

---

## 🌍 Vision: GAIA System - Hierarchical Immune Network

ML Defender no es solo un IDS - es un **sistema inmunológico jerárquico distribuido** para redes empresariales globales.

### Arquitectura Jerárquica Multi-Nivel
```
                    ┌─────────────────────────────────────┐
                    │   GLOBAL RAG-MASTER (Nivel 3)      │
                    │   etcd-server (HA cluster)          │
                    │   "Cerebro - Visión global"         │
                    └──────────────┬──────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
    ┌───────▼───────┐      ┌──────▼──────┐      ┌───────▼───────┐
    │ Campus-A      │      │ Campus-B    │      │ Campus-C      │
    │ RAG-Master L2 │      │ RAG-Master  │      │ RAG-Master    │
    │ etcd-server   │      │ etcd-server │      │ etcd-server   │
    │ "Ganglio"     │      │ "Ganglio"   │      │ "Ganglio"     │
    └───────┬───────┘      └──────┬──────┘      └───────┬───────┘
            │                     │                      │
    ┌───────┼──────┐      ┌──────┼──────┐       ┌──────┼──────┐
    │       │      │      │      │      │       │      │      │
┌───▼─┐ ┌──▼──┐┌──▼─┐ ┌──▼─┐ ┌──▼──┐┌──▼─┐  ┌──▼─┐ ┌──▼──┐┌──▼─┐
│Edif │ │Edif ││Edif│ │Edif│ │Edif ││Edif│  │Edif│ │Edif ││Edif│
│  1  │ │  2  ││  3 │ │  4 │ │  5  ││  6 │  │  7 │ │  8  ││  9 │
│Local│ │Local││Loc │ │Loc │ │Local││Loc │  │Loc │ │Local││Loc │
│RAG-M│ │RAG-M││RAG │ │RAG │ │RAG-M││RAG │  │RAG │ │RAG-M││RAG │
│etcd │ │etcd ││etcd│ │etcd│ │etcd ││etcd│  │etcd│ │etcd ││etcd│
└──┬──┘ └──┬──┘└──┬─┘ └──┬─┘ └──┬──┘└──┬─┘  └──┬─┘ └──┬──┘└──┬─┘
   │       │      │       │       │      │       │       │      │
 Plant  Plant  Plant   Plant   Plant  Plant   Plant   Plant  Plant
  1-1    1-2    1-3     2-1     2-2    2-3     3-1     3-2    3-3
```

### Principios de Jerarquía

**Nivel 1 (Local) - Edificio/Planta:**
```
- 1 etcd-server por edificio
- 1 RAG-master local
- N RAG-clients (1 por planta/zona)
- M ml-detectors (1:1 con RAG-clients)
- Decisiones: Locales, inmediatas
- Propagación: Hacia arriba (anomalías críticas)
- Awareness: Solo su edificio
```

**Nivel 2 (Campus) - Grupo de Edificios:**
```
- 1 etcd-server (HA) por campus
- 1 RAG-master campus
- Agrega: 5-10 edificios
- Decisiones: Campus-wide policies
- Propagación: Bidireccional (arriba/abajo)
- Awareness: Su campus, NO otros campus
- Sincroniza: Con nivel superior
```

**Nivel 3 (Global) - Organización:**
```
- 1 etcd-server (HA cluster, multi-region)
- 1 RAG-master global
- Agrega: Todos los campus
- Decisiones: Global threat response
- Propagación: Top-down (vacunas globales)
- Awareness: Visión completa, NO lateral entre campus
- Autoridad: Máxima, override local si crítico
```

---

## 🧬 Flujo de "Vacunación" Jerárquico

### Escenario 1: Amenaza Local (Edificio 1, Planta 2)
```
1. RAG-client (Edificio-1, Planta-2) detecta ransomware variant
   → Divergence score > 0.8 (nunca visto)
   → NEW: reason_code: PCA_OUTLIER
   
2. RAG-master Local (Edificio-1) recibe alerta
   → LLM analiza provenance: 2 engines, alta discrepancia
   → Operador local valida: ✅ Confirma ransomware
   → Genera "vacuna local": nuevo embedding signature
   
3. Decisión: Propagación limitada
   → Distribuye a TODAS las plantas del Edificio-1
   → Tiempo: <30 segundos
   → NO propaga a otros edificios (autonomía local)
   
4. Edificio-1 inmunizado
   → Plantas 1-5 detectan variant instantáneamente
   → Otros edificios: sin conocimiento (aún)
```

### Escenario 2: Amenaza Campus (Campus-A)
```
1. RAG-master Local (Edificio-3) detecta patrón recurrente
   → Mismo ransomware en 3 plantas diferentes
   → NEW: Multiple verdicts con ENGINE_CONFLICT
   → Severity escalation → Informa a RAG-master Campus-A
   
2. RAG-master Campus-A analiza
   → Correlaciona con Edificio-1 (mismo campus)
   → LLM Campus-level: Patrón confirmado en 2 edificios
   → Provenance analysis: Consistent reason codes
   → Operador Campus valida: ✅ Amenaza campus-wide
   → Genera "vacuna campus": embedding + metadata
   
3. Decisión: Propagación campus
   → Distribuye a RAG-masters de Edificios 1-5 (Campus-A)
   → Cada RAG-master local distribuye a sus plantas
   → Tiempo: <5 minutos (cascada)
   → NO propaga a Campus-B ni Campus-C (no awareness lateral)
   
4. Campus-A inmunizado
   → 5 edificios, 25 plantas protegidas
   → Otros campus: sin conocimiento
```

### Escenario 3: Amenaza Global (APT detectado)
```
1. RAG-master Campus-A detecta APT sofisticado
   → Mismo actor en múltiples edificios
   → Técnicas avanzadas (zero-day exploit)
   → NEW: PCA_OUTLIER + ENGINE_CONFLICT en múltiples eventos
   → Severity: CRITICAL → Escala a RAG-master Global
   
2. RAG-master Global analiza
   → Correlaciona Campus-A + Campus-B (misma firma)
   → LLM Global: APT campaign confirmada
   → Provenance patterns: Consistent discrepancy signatures
   → Operador Global valida: ✅ Threat actor nation-state
   → Genera "vacuna global": complete threat profile
   
3. Decisión: Propagación global
   → Override authority: Distribuye a TODOS los campus
   → Cada RAG-master campus → sus edificios
   → Cada RAG-master edificio → sus plantas
   → Tiempo: <15 minutos (cascada global)
   → Priority: MÁXIMA (bypasses local queues)
   
4. Organización completa inmunizada
   → Todos los campus, edificios, plantas
   → Detección instantánea del APT
   → Threat intelligence global aplicada
```

---

## 📅 Phase 2A - Foundation (Week 5: Days 35-40)

### ✅ Day 35 - Skeleton Complete (2026-01-11)

**Completado:**
- [x] Directory structure (18 directories, 12 files)
- [x] CMakeLists.txt with dependency detection
- [x] Configuration parser (JSON → Config struct)
- [x] Main loop with signal handling
- [x] All stub files created (embedders, indexers, etc.)
- [x] Test suite passing (test_config_parser)
- [x] Binary compiling and running
- [x] Dependencies verified

**Via Appia Milestones:**
- 🏛️ Foundation first: Estructura completa antes de funcionalidad
- 🏛️ Dependency clarity: Todas las librerías verificadas
- 🏛️ Test-driven: Test suite desde día 1
- 🏛️ Raspberry Pi target: Diseñado para hardware barato (~310MB RAM)

---

### ✅ Day 36 - Crypto Integration & Compilation (2026-01-12)

**Completado:**
- [x] Integrar crypto-transport API real
- [x] Corregir event_loader.cpp con crypto-transport
- [x] Actualizar CMakeLists.txt (protobuf desde build/proto)
- [x] Integrar rag-ingester en Makefile raíz
- [x] Compilación exitosa: `[100%] Built target rag-ingester`
- [x] Binario funcional: Arranca y espera eventos
- [x] 101-feature extraction implementada

**Decisión de Seguridad (ADR-001):**
- 🔒 Cifrado y compresión HARDCODED (no configurables)
- 🔒 Prevención de poison log attacks

**Via Appia Milestones:**
- 🏛️ Security first: Encryption mandatory, not optional
- 🏛️ Real APIs: crypto-transport integrated correctly
- 🏛️ Clean compilation: 0 errors

---

### ✅ Day 37 - ADR-002 Provenance + ADR-001 Encryption (2026-01-13)

**MAJOR MILESTONE COMPLETADO:**

**ADR-002 Implementation:**
- [x] Protobuf contract extendido: `DetectionProvenance` + `EngineVerdict`
- [x] Created `/vagrant/common/include/reason_codes.hpp` (5 codes)
- [x] Sniffer: Fills verdict in fast-path detection
- [x] ml-detector: Adds RF verdict + calculates discrepancy_score
- [x] rag-ingester: Parses complete provenance
- [x] Event struct: Extended with verdicts + discrepancy
- [x] CMakeLists.txt: Updated for 3 components

**ADR-001 BONUS:**
- [x] RAGLogger encrypts artifacts: `.pb.enc`, `.json.enc`
- [x] Pipeline: Serialize → Compress → Encrypt
- [x] crypto_manager integrated through component chain
- [x] save_artifacts() rewritten with crypto-transport

**Infrastructure:**
- [x] Vagrantfile: Permanent ONNX Runtime lib64 symlinks fix
- [x] All components compile cleanly
- [x] Functional binaries: sniffer (1.4M), ml-detector, rag-ingester

**Technical Debt Identified:**
- [ ] ISSUE-007: Magic numbers in ml-detector → JSON config
- [ ] ISSUE-006: Log files not persisted → spdlog file sinks
- [ ] MISSING: Acceptance tests for protobuf contract
- [ ] ISSUE-005: RAGLogger memory leak (known)
- [ ] ISSUE-003: Thread-local FlowManager bug (known)

**Via Appia Milestones:**
- 🏛️ Situational Intelligence: From binary to multi-engine provenance
- 🏛️ 0-day Detection: PCA_OUTLIER + ENGINE_CONFLICT signals
- 🏛️ Security Hardened: Encryption at code level, not config
- 🏛️ Forensics Ready: Complete audit trail with reason codes

---

### 📋 Day 38 - Synthetic Data + ONNX Embedders (103 features)

**Goals:**
- [ ] Verify clean compilation from scratch (VM rebuild)
- [ ] Create `generate_synthetic_events.py` script
- [ ] Generate 100+ .pb.enc files with provenance
- [ ] Acceptance tests for protobuf contract
- [ ] Update ChronosEmbedder (103 → 512-d)
- [ ] Update SBERTEmbedder (103 → 384-d)
- [ ] Update AttackEmbedder (103 → 256-d)
- [ ] Test end-to-end: synthetic data → rag-ingester → embeddings

**Synthetic Data Strategy:**
```python
# generate_synthetic_events.py
event.features = [101 features]  # Synthetic realistic data
event.provenance.verdicts = [
    {"engine": "sniffer", "confidence": 0.9, "reason": "STAT_ANOMALY"},
    {"engine": "rf", "confidence": 0.85, "reason": "STAT_ANOMALY"}
]
event.provenance.discrepancy_score = 0.15  # Low (agreement)
```

**ONNX Embedders Update:**
```cpp
// 103 features: 101 original + 2 meta
std::vector<float> input;
input.insert(input.end(), event.features.begin(), event.features.end());  // 101
input.push_back(event.discrepancy_score);  // 102
input.push_back(static_cast<float>(event.verdicts.size()));  // 103

// ONNX inference: 103 → 512/384/256
```

**Success criteria:**
- ✅ 100+ .pb.enc files with provenance generated
- ✅ Acceptance tests PASS
- ✅ Embedders process 103 features correctly
- ✅ End-to-end pipeline functional
- ✅ No crashes with synthetic data

---

### 📋 Day 39 - Technical Debt Cleanup

**Goals:**
- [ ] Fix ISSUE-007: Magic numbers → JSON config
- [ ] Fix ISSUE-006: Log files persistence
- [ ] Analysis of ISSUE-003: Thread-local FlowManager bug
- [ ] Decision on fix strategy for FlowManager

**Success criteria:**
- ✅ ml-detector config has thresholds in JSON
- ✅ All components write logs to files
- ✅ Monitor scripts can tail logs
- ✅ FlowManager bug root cause identified

---

### 📋 Day 40 - Bug Fixes + Integration Testing

**Goals:**
- [ ] Fix ISSUE-003: Thread-local FlowManager (if feasible)
- [ ] Alternative: Document workaround + synthetic data
- [ ] End-to-end integration tests
- [ ] Performance benchmarking
- [ ] etcd registration (if time permits)

**Success criteria:**
- ✅ Either FlowManager fixed OR workaround documented
- ✅ Integration tests PASS
- ✅ <500ms latency per event
- ✅ Memory usage <500MB (target)

---

## 📅 Phase 2B - Optimization (Week 6: Days 41-45)

### Day 41 - Multi-Threading

**Goals:**
- [ ] Enable parallel mode in config
- [ ] ThreadPool for embeddings (3 workers)
- [ ] ThreadPool for indexing (4 workers)
- [ ] Performance benchmarking (target: 500 events/sec)

### Day 42 - Persistence & Checkpointing

**Goals:**
- [ ] FAISS index save/load
- [ ] Checkpoint every 1000 events
- [ ] Graceful shutdown with persistence
- [ ] Recovery from crash (load last checkpoint)

### Day 43 - Advanced Strategies

**Goals:**
- [ ] Temporal tiers (hot/warm/cold)
- [ ] Metadata-first search
- [ ] Quantization (int8)

### Day 44 - Integration Testing

**Goals:**
- [ ] End-to-end pipeline test (sniffer → ingester → search)
- [ ] Performance benchmarks (10K events)
- [ ] Memory profiling (target: <500MB)
- [ ] Load testing (continuous 24h)

### Day 45 - Documentation & Hardening

**Goals:**
- [ ] API documentation
- [ ] Deployment guide (systemd service)
- [ ] Troubleshooting guide
- [ ] Operational runbook

---

## 📅 Phase 3 - GAIA System (Weeks 7-8)

### RAG-Master Development (Local Level)

**Components:**
- [ ] Orchestrator service
- [ ] LLM validator (TinyLlama)
- [ ] Vaccine distributor (to local RAG-clients)
- [ ] Multi-client coordination
- [ ] Health aggregator

**Features:**
- [ ] Anomaly validation (reduce false positives)
- [ ] Vaccine generation (embedding signatures)
- [ ] Distribution to all plants in building
- [ ] Decision authority (local scope)
- [ ] NEW: Provenance analysis for 0-day detection

### RAG-Master Campus (Nivel 2)

**Components:**
- [ ] Campus-level orchestrator
- [ ] Multi-building aggregation
- [ ] Upward sync to Global
- [ ] Downward distribution to buildings
- [ ] NO lateral sync (isolated campus)

**Features:**
- [ ] Campus-wide threat correlation
- [ ] Policy enforcement
- [ ] Model update distribution
- [ ] Building health monitoring
- [ ] NEW: Cross-building provenance correlation

### RAG-Master Global (Nivel 3)

**Components:**
- [ ] Global orchestrator
- [ ] Multi-campus aggregation
- [ ] Threat intelligence APIs
- [ ] Global policy engine
- [ ] Override authority

**Features:**
- [ ] APT detection (cross-campus correlation)
- [ ] Global vaccine distribution
- [ ] Compliance enforcement
- [ ] Organization-wide visibility
- [ ] NEW: Global provenance pattern analysis

---

## 📚 KEY DOCUMENTS

### Current Component
- `/vagrant/rag-ingester/docs/BACKLOG.md` - This file
- `/vagrant/rag-ingester/docs/ADR_002_MULTI_ENGINE_PROVENANCE.md` - Full ADR-002 spec
- `/vagrant/rag-ingester/README.md` - Build & run instructions
- `/vagrant/rag-ingester/config/rag-ingester.json` - Configuration

### Shared Resources
- `/vagrant/common/include/reason_codes.hpp` - 5 reason codes (ADR-002)
- `/vagrant/protobuf/network_security.proto` - THE LAW (updated Day 37)

### Related Components
- `/vagrant/sniffer/` - ml-detector (produces encrypted .pb.enc files)
- `/vagrant/ml-detector/` - ML pipeline (adds RF verdict + encryption)
- `/vagrant/etcd-client/` - Service discovery library
- `/vagrant/crypto-transport/` - Encryption/compression library

### Bug Reports
- `/vagrant/docs/bugs/2025-01-10_thread_local_flowmanager_bug.md`
- `/vagrant/docs/bugs/2026-01-13_magic_numbers_ml_detector.md` (NEW)

---

## 🎓 Lessons Learned

### Day 35

1. ✅ Library naming matters
2. ✅ Forward declarations need full headers
3. ✅ System vs local libs order
4. ✅ Log permissions (use /tmp)
5. ✅ Via Appia: Skeleton first
6. ✅ Test-driven from day 1

### Day 36

1. ✅ Real APIs over invented
2. ✅ Config parser exists - don't reinvent
3. ✅ API consistency matters
4. ✅ Security by design
5. ✅ Poison log prevention critical
6. ✅ Fix headers first, then source

### Day 37 (NEW)

1. ✅ **Protobuf is THE LAW** - All components must honor contract
2. ✅ **Shared headers strategy** - Create `/vagrant/common/include` for cross-component
3. ✅ **CMakeLists propagation** - All 3 components need same include paths
4. ✅ **Crypto-transport chain** - Pass crypto_manager through component hierarchy
5. ✅ **Factory functions** - Update signatures when adding parameters
6. ✅ **Technical debt tracking** - Document magic numbers immediately
7. ✅ **Via Appia Quality** - Complete one feature before next (ADR-002 complete before ONNX)

---

## 📊 Success Metrics

### Phase 2A (Week 5)
- ✅ Compilation successful (Days 35-37)
- ✅ All tests passing (Days 35-37)
- ✅ Dependencies resolved (Days 35-37)
- ✅ Binary functional (Days 36-37)
- ✅ ADR-002 implemented (Day 37)
- ✅ ADR-001 hardened (Day 37)
- [ ] Synthetic data generation (Day 38)
- [ ] ONNX Embedders updated (Day 38)
- [ ] End-to-end pipeline working (Day 40)
- [ ] <500ms latency per event

### Phase 2B (Week 6)
- [ ] Multi-threading operational
- [ ] Memory usage <500MB (100K events)
- [ ] CV metrics stable >0.20
- [ ] 10+ hours continuous operation

### Phase 3 (Weeks 7-8)
- [ ] RAG-master Local operational
- [ ] Vaccine distribution <30 sec (local)
- [ ] RAG-master Campus operational
- [ ] Vaccine distribution <5 min (campus)
- [ ] RAG-master Global operational
- [ ] Vaccine distribution <15 min (global)

---

## 📈 Progress Visual
```
Phase 1:  [████████████████████] 100% COMPLETE
Phase 2A: [████████░░░░░░░░░░░░]  40% (Days 35-37/40)
Phase 2B: [░░░░░░░░░░░░░░░░░░░░]   0%
Phase 3:  [░░░░░░░░░░░░░░░░░░░░]   0%
```

**Days 35-37 Completion:**
- Structure:    [████] 100% ✅
- Dependencies: [████] 100% ✅
- Tests:        [████] 100% ✅
- Compilation:  [████] 100% ✅
- Crypto:       [████] 100% ✅
- Provenance:   [████] 100% ✅ (NEW)
- Embedders:    [░░░░]   0% ← Day 38
- Integration:  [█░░░]  25% ← Days 38-40

---

## 🏛️ Via Appia Quality Checkpoints

**Foundation (Week 5):**
- [x] Estructura antes que funcionalidad
- [x] Dependencias verificadas antes de código
- [x] Tests desde día 1
- [x] Compilación limpia antes de features
- [x] Security by design (encryption mandatory)
- [x] Provenance contract complete (ADR-002)
- [x] Generator compiled with production compliance (Day 38) ← NEW
- [ ] Synthetic data execution validated (Day 38 completion)
- [ ] ONNX embedders updated (Day 38 completion)
- [ ] End-to-end validation antes de expansión (Day 40)

**Expansion (Week 6):**
- [ ] Multi-threading solo cuando single funciona
- [ ] Optimización solo con profiling real
- [ ] Persistencia antes de distribución

**Production (Weeks 7-8):**
- [ ] GAIA hierarchy incremental (local → campus → global)
- [ ] Failover tested en cada nivel
- [ ] Disaster recovery procedures documented

---

**End of Backlog**

**Last Updated:** 2026-01-14 (Day 38 Parcial - Generator Compiled)  
**Next Update:** 2026-01-15 (Day 38 Complete - Execution + ONNX)  
**Vision:** Sistema inmunológico jerárquico global 🌍  
**Security:** Multi-engine provenance + Encryption mandatory 🔒  
**Quality:** Via Appia - Generator ready, execution tomorrow 🏛️