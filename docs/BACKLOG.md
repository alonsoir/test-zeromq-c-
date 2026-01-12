# RAG Ingester - Development Backlog

**Last Updated:** 2026-01-12  
**Current Phase:** 2A - Foundation (Day 36 Complete)  
**Next Session:** Day 37 - ONNX Runtime Embedders

---

## 🔒 CRITICAL SECURITY DECISION: Mandatory Encryption

**ADR-001: Encryption is NOT Optional**

**Decision:** Encryption and compression are HARDCODED in the pipeline, NOT configurable.

**Rationale:**
- **Poison Log Prevention:** Attacker could disable encryption to inject malicious events
- **Data Integrity:** Compressed + encrypted data has built-in tamper detection
- **Compliance:** Enterprise security requires encryption at rest
- **No Backdoors:** No "debug mode" that bypasses security

**Implementation:**
```cpp
// ml-detector (rag_logger.cpp)
void RAGLogger::log_event(const NetworkSecurityEvent& event) {
    // 1. Serialize protobuf
    std::string serialized;
    event.SerializeToString(&serialized);
    
    // 2. Compress (ALWAYS - not configurable)
    auto compressed = crypto_transport::compress(data);
    
    // 3. Encrypt (ALWAYS - not configurable)
    auto encrypted = crypto_transport::encrypt(compressed, key_from_etcd);
    
    // 4. Write
    write_to_file(encrypted);
}

// rag-ingester (event_loader.cpp)
Event EventLoader::load(const std::string& filepath) {
    auto encrypted = read_file(filepath);
    
    // Decrypt (ALWAYS - no fallback to plaintext)
    auto decrypted = crypto_transport::decrypt(encrypted, key_);
    if (!decrypted) {
        throw SecurityException("Decryption failed - rejecting event");
    }
    
    // Decompress (ALWAYS)
    auto decompressed = crypto_transport::decompress(decrypted);
    
    return parse_protobuf(decompressed);
}
```

**Config Fields REMOVED:**
```json
// ❌ BEFORE (insecure):
{
  "ingester": {
    "input": {
      "encrypted": true,  // ← REMOVED - always true
      "compressed": true  // ← REMOVED - always true
    }
  }
}

// ✅ AFTER (secure):
{
  "ingester": {
    "input": {
      "directory": "/vagrant/logs/rag/events",
      "pattern": "*.pb"
      // encryption/compression implicit, not configurable
    }
  }
}
```

**Threat Model:**
- Attacker gains access to config file
- Sets `encrypted: false`
- Injects poisoned events in plaintext
- RAG ingester accepts them → FAISS poisoned
- System compromised

**Mitigation:**
- Encryption is code-level contract, not config-level
- Any plaintext event is rejected with SecurityException
- Config only controls paths, patterns, NOT security primitives

---

## 🔍 ADR-002: Multi-Engine Provenance & Situational Intelligence

**Date:** 12 Enero 2026  
**Status:** APPROVED  
**Decision:** Extend protobuf contract to capture multiple engine verdicts  
**Proposer:** Gemini (peer reviewer)  
**Implementation:** Day 37 (before ONNX embedders)

---

### Context

Currently, the protobuf contract only stores the **final classification** result. This discards valuable information:

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

### Protobuf Extension

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
}

// Add to NetworkSecurityEvent:
message NetworkSecurityEvent {
  // ... existing fields (101 features) ...
  
  DetectionProvenance provenance = 20;  // NEW
}
```

### Reason Codes (Gemini's Table)

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

### Benefits

1. **Situational Intelligence** - Not just "what" but "why" and "how sure"
2. **0-day Detection** - PCA_OUTLIER + ENGINE_CONFLICT = red flag
3. **Adaptive Learning** - LLM learns campus-specific patterns
4. **Reduced False Positives** - Context helps distinguish noise from threats
5. **Forensics** - Complete audit trail of all engine decisions
6. **Vaccine Quality** - Discrepancies help prioritize which events to analyze

### Implementation Strategy

**Phase 1: ml-detector (writes provenance)**
```cpp
void RAGLogger::log_event(
    const PacketAnalysis& sniffer_verdict,
    const RandomForestResult& rf_result,
    const CNNResult& cnn_result
) {
    NetworkSecurityEvent event;
    
    // ... existing feature extraction ...
    
    // NEW: Fill provenance
    auto* prov = event.mutable_provenance();
    
    // Add all verdicts with reason codes
    // Calculate discrepancy score
    // Record final decision
    
    // Encrypt, compress, write...
}
```

**Phase 2: rag-ingester (reads provenance)**
```cpp
Event EventLoader::load(const std::string& filepath) {
    // Decrypt, decompress, parse protobuf...
    
    Event event;
    event.features = extract_101_features(proto_event);
    
    // NEW: Parse provenance
    if (proto_event.has_provenance()) {
        for (const auto& verdict : prov.verdicts()) {
            event.verdicts.push_back(verdict);
        }
        event.discrepancy_score = prov.discrepancy_score();
    }
    
    return event;
}
```

**Phase 3: Embedders use discrepancy**
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

### Success Metrics

- ✅ Reduced false positive rate (target: -20%)
- ✅ 0-day detection improved (measure: time to detection)
- ✅ LLM can explain decisions (qualitative: operator feedback)
- ✅ Vaccine quality improved (measure: global vs local vaccine ratio)

### Implementation Timeline

**Day 37 Morning (2-3 hours):**
- Update `network_security.proto`
- Regenerate protobuf (make proto-unified)
- Update ml-detector (fill provenance)
- Add encryption to ml-detector (BONUS)
- Test with synthetic data

**Day 37 Afternoon (3-4 hours):**
- Update rag-ingester (parse provenance)
- Update embedders (use discrepancy as feature)
- Implement ONNX embedders
- Test end-to-end

---

**Full Documentation:** See `/vagrant/rag-ingester/docs/ADR_002_MULTI_ENGINE_PROVENANCE.md`

**Status:** APPROVED - Ready for implementation Day 37  
**Proposed by:** Gemini (peer reviewer)  
**Co-authors:** Alonso, Claude, Gemini

🏛️ **Via Appia:** This transforms the system from binary decision to situational intelligence - exactly the kind of architectural decision that builds systems for 2000 years.

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
   
2. RAG-master Local (Edificio-1) recibe alerta
   → LLM analiza: ¿Es verdadero positivo?
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
   → Severity escalation → Informa a RAG-master Campus-A
   
2. RAG-master Campus-A analiza
   → Correlaciona con Edificio-1 (mismo campus)
   → LLM Campus-level: Patrón confirmado en 2 edificios
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
   → Severity: CRITICAL → Escala a RAG-master Global
   
2. RAG-master Global analiza
   → Correlaciona Campus-A + Campus-B (misma firma)
   → LLM Global: APT campaign confirmada
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

## 🔐 Sincronización etcd-server Jerárquica

### Modelo de Sincronización

**Upward Sync (Bottom-Up):**
```
Local etcd → Campus etcd → Global etcd

Qué sube:
- Anomalías críticas (divergence > 0.7)
- Health metrics agregados
- Threat signatures locales (candidates)

Frecuencia:
- Real-time: Alertas críticas
- Periodic: Cada 5 min (health)
```

**Downward Sync (Top-Down):**
```
Global etcd → Campus etcd → Local etcd

Qué baja:
- Vacunas globales (threat signatures)
- Policy updates (compliance)
- Model updates (new ML models)

Frecuencia:
- Real-time: Vacunas críticas
- Periodic: Cada 1 hora (policies)
```

**NO Lateral Sync:**
```
Campus-A etcd ⇿ Campus-B etcd  ❌ PROHIBIDO

Razón:
- Blast radius control
- Performance (avoid mesh complexity)
- Security (lateral movement prevention)
- Autonomy (campus independence)

Excepción:
- Solo via Global etcd (explicit authorization)
```

### Tolerancia a Fallos

**Local etcd-server falla:**
```
1. RAG-master local sigue operando (cached policies)
2. No puede sync upward (queued)
3. Downward sync buffered en Campus etcd
4. Auto-reconnect cuando etcd-server recovered
5. Sync backlog (últimos 24h)
```

**Campus etcd-server falla (HA cluster):**
```
1. Failover automático (Raft consensus)
2. Standby replica promoted a leader
3. Local etcd-servers re-connect
4. Zero data loss (Raft log)
```

**Global etcd-server falla:**
```
1. Campus etcd-servers operan autónomos
2. Local decisions continue
3. Global vacunas queued
4. Manual intervention si >1 hora
5. Disaster recovery plan activated
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
- [x] Dependencies verified:
  - ✅ etcd_client: `/usr/local/lib/libetcd_client.so`
  - ✅ crypto_transport: `/usr/local/lib/libcrypto_transport.so`
  - ✅ common-rag-ingester: `/vagrant/common-rag-ingester/build/`
  - ✅ FAISS: `/usr/local/lib/libfaiss.so`
  - ✅ ONNX Runtime: `/usr/local/lib/libonnxruntime.so`

**Via Appia Milestones:**
- 🏛️ Foundation first: Estructura completa antes de funcionalidad
- 🏛️ Dependency clarity: Todas las librerías verificadas
- 🏛️ Test-driven: Test suite desde día 1
- 🏛️ Raspberry Pi target: Diseñado para hardware barato (~310MB RAM)

---

### ✅ Day 36 - Crypto Integration & Compilation (2026-01-12)

**Completado:**
- [x] Integrar crypto-transport API real (`crypto.hpp`, `compression.hpp`)
- [x] Corregir event_loader.cpp con crypto-transport
- [x] Actualizar CMakeLists.txt (protobuf desde build/proto)
- [x] Integrar rag-ingester en Makefile raíz
- [x] Corregir main.cpp (ConfigParser::load, FileWatcher API)
- [x] Compilación exitosa: `[100%] Built target rag-ingester`
- [x] Binario funcional: Arranca y espera eventos
- [x] 101-feature extraction implementada (event_loader)

**Problemas Resueltos:**
1. ✅ Headers crypto-transport inventados → API real
2. ✅ config.hpp faltante → ConfigParser::load() existente
3. ✅ Campos config incorrectos → threading.mode, input.pattern
4. ✅ API FileWatcher incorrecta → start(callback)
5. ✅ Protobuf no encontrado → Copiado a build/proto
6. ✅ Clave cifrado en config → Preparado para etcd-client

**Decisión de Seguridad (ADR-001):**
- 🔒 Cifrado y compresión HARDCODED (no configurables)
- 🔒 Prevención de poison log attacks
- 🔒 Sin "modo debug" que bypass seguridad

**Output del Binario:**
```bash
vagrant@bookworm:/vagrant/rag-ingester/build$ ./rag-ingester
[INFO] RAG Ingester starting...
[INFO] Configuration loaded
[INFO] EventLoader: Crypto initialized (ChaCha20-Poly1305 + LZ4)
[INFO] FileWatcher started: /vagrant/logs/rag/events/ (*.pb)
[INFO] ✅ RAG Ingester ready and waiting for events
```

**Via Appia Milestones:**
- 🏛️ Security first: Encryption mandatory, not optional
- 🏛️ Real APIs: crypto-transport integrated correctly
- 🏛️ Clean compilation: 0 errors, warnings ignorables (stubs)
- 🏛️ Functional binary: Waits for encrypted .pb files

**Estado:**
- ✅ Compila limpiamente
- ✅ Tests pasando (14/14)
- ✅ Binario arranca sin errores
- ⏳ Necesita .pb cifrados para testing completo

---

### 📋 Day 37 - Embedders (ONNX Runtime)

**Goals:**
- [ ] Download/prepare ONNX models
- [ ] Implement `ChronosEmbedder` (83 → 512-d)
- [ ] Implement `SBERTEmbedder` (83 → 384-d)
- [ ] Implement `AttackEmbedder` (83 → 256-d)
- [ ] ONNX Runtime session initialization
- [ ] Batch inference support

**Models strategy:**
```bash
# Option 1: Use existing PCA embedder as placeholder
cp /vagrant/contrib/claude/pca_pipeline/models/pca_embedder.onnx \
   /vagrant/rag-ingester/models/onnx/chronos.onnx

# Option 2: Download pre-trained from HuggingFace
# Option 3: Train custom embedders (Week 6)
```

**Implementation:**
```cpp
class ChronosEmbedder {
    Ort::Session* session_;
    Ort::MemoryInfo memory_info_;
    
    std::vector<float> embed(const Event& event) {
        // Prepare input tensor (83 features)
        std::vector<float> input = event.features;
        
        // Run inference
        auto output_tensor = session_->Run(...);
        
        // Extract 512-d embedding
        return std::vector<float>(output_data, output_data + 512);
    }
};
```

**Success criteria:**
- ✅ ONNX models loaded successfully
- ✅ Inference <10ms per event
- ✅ Correct output dimensions (512, 384, 256)
- ✅ Batch processing functional
- ✅ Thread-safe (multiple inference sessions)

---

### 📋 Day 38 - PCA & Multi-Index Manager

**Goals:**
- [ ] Integrate `common-rag-ingester` PCA library
- [ ] Dimensionality reduction (512→128, 384→96, 256→64)
- [ ] Implement `MultiIndexManager`
- [ ] Create 4 FAISS indices
- [ ] Implement eventual consistency logic

**PCA Integration:**
```cpp
#include <dimensionality_reducer.hpp>

DimensionalityReducer reducer(512, 128);
reducer.load("/vagrant/rag-ingester/models/pca/chronos_512_128.faiss");

auto chronos_emb = chronos_embedder->embed(event);  // 512-d
auto reduced = reducer.transform(chronos_emb);       // 128-d
```

**Multi-Index Architecture:**
```cpp
class MultiIndexManager {
    std::unique_ptr<faiss::IndexFlatL2> chronos_index_;         // 128-d
    std::unique_ptr<faiss::IndexFlatL2> sbert_index_;           // 96-d
    std::unique_ptr<faiss::IndexFlatL2> entity_benign_index_;   // 64-d
    std::unique_ptr<faiss::IndexFlatL2> entity_malicious_index_;// 64-d
    
    CommitResult add_event(const Event& event, 
                          const Embeddings& embeddings) {
        // Best-effort commit (eventual consistency)
        CommitResult result;
        
        try { 
            chronos_index_->add(1, embeddings.chronos.data());
            result.successful_commits++;
        } catch (...) { 
            result.failed_commits++; 
        }
        
        // Same for sbert, entity_benign, entity_malicious
        return result;
    }
};
```

**Success criteria:**
- ✅ PCA reduces dimensions correctly
- ✅ Variance retained >95%
- ✅ All 4 indices operational
- ✅ Best-effort commit working
- ✅ Partial failures handled gracefully
- ✅ Health metrics tracked

---

### 📋 Day 39 - Health Monitoring

**Goals:**
- [ ] Implement `IndexHealthMonitor`
- [ ] CV (Coefficient of Variation) calculation
- [ ] Alert when CV < 0.20
- [ ] etcd health reporting

**Health Monitoring:**
```cpp
struct HealthMetrics {
    double CV;              // Target: >0.20
    double mean_distance;
    double std_distance;
    size_t num_vectors;
    
    bool is_healthy() const { return CV > 0.2; }
    bool is_degrading() const { return CV < 0.25; }
};

class IndexHealthMonitor {
    HealthMetrics compute_health(faiss::Index* index) {
        // Sample 1000 random vectors
        // Compute k-NN distances
        // Calculate statistics
        return { CV, mean, std, ntotal };
    }
    
    void monitor_loop() {
        while (running_) {
            auto chronos_health = compute_health(chronos_index_);
            
            if (!chronos_health.is_healthy()) {
                spdlog::warn("Chronos CV={:.3f} < 0.20", chronos_health.CV);
                trigger_alert("chronos_degradation");
            }
            
            report_to_etcd(chronos_health);
            std::this_thread::sleep_for(std::chrono::seconds(60));
        }
    }
};
```

**Success criteria:**
- ✅ CV calculated correctly
- ✅ Alerts trigger at thresholds
- ✅ Health reported to etcd every 60s
- ✅ Dashboard-ready metrics

---

### 📋 Day 40 - etcd Integration & Symbiosis

**Goals:**
- [ ] Register in etcd with `partner_detector`
- [ ] Heartbeat every 10s
- [ ] Subscribe to ml-detector status
- [ ] Alert if partner fails
- [ ] Test coordinated shutdown

**etcd Registration:**
```cpp
void register_service() {
    nlohmann::json metadata = {
        {"type", "rag-ingester"},
        {"location", config_.service.location},
        {"partner_detector", config_.service.etcd.partner_detector},
        {"faiss_indices", {
            {"chronos", {
                {"vectors", chronos_index_->ntotal},
                {"cv", chronos_health.CV}
            }},
            {"sbert", {...}},
            {"entity_benign", {...}},
            {"entity_malicious", {...}}
        }},
        {"health", {
            {"status", "healthy"},
            {"last_heartbeat", iso_timestamp()}
        }}
    };
    
    etcd_client_->put(
        "/ml-defender/services/rag-ingester-" + config_.service.location,
        metadata.dump(),
        10  // TTL seconds
    );
}

void heartbeat_loop() {
    while (running_) {
        register_service();  // Refresh TTL
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
}

void watch_partner() {
    etcd_client_->watch(
        "/ml-defender/services/" + config_.service.etcd.partner_detector,
        [this](const etcd::Response& response) {
            if (response.is_expired()) {
                spdlog::error("Partner detector {} expired!", 
                             config_.service.etcd.partner_detector);
                // Trigger alert, pause ingestion
            }
        }
    );
}
```

**Success criteria:**
- ✅ Service visible in etcd
- ✅ Heartbeat maintains TTL
- ✅ Partner detection working
- ✅ Coordinated shutdown tested
- ✅ Symbiosis observable

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

---

## 📅 Phase 4 - Post-Hardening (Future)

### Model Re-training

**Capabilities:**
- [ ] Continual learning from new threats
- [ ] A/B testing of model versions
- [ ] Automatic rollback on degradation
- [ ] Federated learning (privacy-preserving)

### Advanced Features

- [ ] GPU acceleration (CUDA)
- [ ] Distributed FAISS (cluster)
- [ ] Real-time model updates
- [ ] Threat intelligence APIs (STIX/TAXII)
- [ ] Integration with SOC/SIEM

---

## 🎓 Lessons Learned

### Day 35

1. ✅ **Library naming matters**: `libetcd_client.so` not `libetcd-client.so`
2. ✅ **Forward declarations**: Need full headers in `.cpp` for `unique_ptr<T>`
3. ✅ **System vs local libs**: Check `/usr/local/lib` first, then `/vagrant`
4. ✅ **Log permissions**: Use `/tmp` instead of `/var/log` to avoid sudo
5. ✅ **Via Appia principle**: Skeleton first, functionality incremental
6. ✅ **Dependency verification**: Always verify libraries exist before linking
7. ✅ **Test-driven**: Test suite from day 1 catches issues early

### Day 36

1. ✅ **Real APIs over invented**: Always check existing library headers first
2. ✅ **Config parser exists**: Don't reinvent - use existing ConfigParser::load()
3. ✅ **API consistency**: FileWatcher uses start(callback), not on_file_created()
4. ✅ **Security by design**: Encryption/compression hardcoded, not configurable
5. ✅ **Poison log prevention**: No config option to bypass security
6. ✅ **Compilation errors cascade**: Fix headers first, then source files
7. ✅ **Integration testing needs data**: Can't test without encrypted .pb files

---

## 📊 Success Metrics

### Phase 2A (Week 5)
- ✅ Compilation successful (Days 35-36)
- ✅ All tests passing (Days 35-36)
- ✅ Dependencies resolved (Days 35-36)
- ✅ Binary functional (Day 36)
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
Phase 2A: [████░░░░░░░░░░░░░░░░]  20% (Days 35-36/40)
Phase 2B: [░░░░░░░░░░░░░░░░░░░░]   0%
Phase 3:  [░░░░░░░░░░░░░░░░░░░░]   0%
```

**Days 35-36 Completion:**
- Structure:    [████] 100% ✅
- Dependencies: [████] 100% ✅
- Tests:        [████] 100% ✅
- Compilation:  [████] 100% ✅
- Crypto:       [████] 100% ✅
- Functionality:[█░░░]  25% ← Days 37-40

---

## 🏛️ Via Appia Quality Checkpoints

**Foundation (Week 5):**
- [x] Estructura antes que funcionalidad
- [x] Dependencias verificadas antes de código
- [x] Tests desde día 1
- [x] Compilación limpia antes de features
- [x] Security by design (encryption mandatory)
- [ ] End-to-end validation antes de expansión

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

**Last Updated:** 2026-01-12 (Day 36 Complete)  
**Next Update:** 2026-01-13 (Day 37 - ONNX Embedders)  
**Vision:** Sistema inmunológico jerárquico global - De edificios a planetas 🌍
**Security:** Encryption mandatory - Poison log prevention 🔒