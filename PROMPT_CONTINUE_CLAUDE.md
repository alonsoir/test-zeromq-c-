# RAG Ingester - Continuation Prompt
**Last Updated:** 12 Enero 2026 - Day 36 Complete  
**Phase:** 2A - Foundation  
**Status:** ✅ Compilation successful, crypto integrated, ready for Day 37

---

## 📍 CURRENT STATE (12 Enero 2026)

### ✅ Day 36 Achievements (TODAY)

**Integración crypto-transport:**
- ✅ API real integrada (`crypto.hpp`, `compression.hpp`)
- ✅ event_loader.cpp con ChaCha20-Poly1305 + LZ4
- ✅ 101-feature extraction implementada
- ✅ CMakeLists.txt con protobuf desde build/proto
- ✅ rag-ingester integrado en Makefile raíz

**Compilación Exitosa:**
```bash
[100%] Built target rag-ingester
```

**Binario Funcional:**
```bash
vagrant@bookworm:/vagrant/rag-ingester/build$ ./rag-ingester
[INFO] RAG Ingester starting...
[INFO] Configuration loaded
[INFO] EventLoader: Crypto initialized (ChaCha20-Poly1305 + LZ4)
[INFO] FileWatcher started: /vagrant/logs/rag/events/ (*.pb)
[INFO] ✅ RAG Ingester ready and waiting for events
```

**Decisión de Seguridad Crítica (ADR-001):**
```
🔒 ENCRYPTION IS NOT OPTIONAL

Cifrado y compresión son HARDCODED en el pipeline.
NO son configurables.

Razón: Poison log prevention
- Attacker NO puede deshabilitar encryption vía config
- Todos los .pb files DEBEN estar cifrados
- Si falla decryption → SecurityException (rechazado)
- Sin "modo debug" que bypass seguridad
```

**Correcciones Aplicadas:**
1. ✅ Headers crypto-transport reales (no inventados)
2. ✅ `ConfigParser::load()` en vez de `load_config()`
3. ✅ FileWatcher API: `start(callback)` en vez de `on_file_created()`
4. ✅ Campos config: `threading.mode`, `input.pattern`
5. ✅ `#include <fstream>` agregado a main.cpp
6. ✅ Protobuf copiado a build/proto por Makefile

**Tests Pasando:**
```bash
./test_file_watcher    # 7/7 ✅
./test_event_loader    # 7/7 ✅
```

---

### ✅ Day 35 (Previous Session)

**Estructura Completa:**
```
/vagrant/rag-ingester/
├── src/          # 12 source files
├── include/      # 12 header files
├── config/       # rag-ingester.json
├── tests/        # Unit tests ✅
├── docs/         # BACKLOG.md, design docs
├── models/       # onnx/, pca/ (empty, ready for Day 37)
└── CMakeLists.txt
```

**Dependencies Verificadas:**
- ✅ `libetcd_client.so` → `/usr/local/lib/`
- ✅ `libcrypto_transport.so` → `/usr/local/lib/`
- ✅ `libcommon-rag-ingester.so` → `/vagrant/common-rag-ingester/build/`
- ✅ `libfaiss.so` → `/usr/local/lib/`
- ✅ `libonnxruntime.so` → `/usr/local/lib/`

---

## 🎯 DAY 37 OBJECTIVES (Immediate Next)

### ONNX Runtime Embedders Implementation

**Goal:** Implementar 3 embedders para generar vectores densos

**Modelos:**
1. **ChronosEmbedder** - 83 features → 512 dimensions
2. **SBERTEmbedder** - 83 features → 384 dimensions
3. **AttackEmbedder** - 83 features → 256 dimensions

**Implementation Pattern:**
```cpp
// include/embedders/chronos_embedder.hpp
class ChronosEmbedder {
public:
    ChronosEmbedder(const std::string& onnx_path);
    std::vector<float> embed(const Event& event);
    
private:
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    std::vector<int64_t> input_shape_;   // [1, 83]
    std::vector<int64_t> output_shape_;  // [1, 512]
};
```

**ONNX Runtime Setup:**
```cpp
// src/embedders/chronos_embedder.cpp
ChronosEmbedder::ChronosEmbedder(const std::string& onnx_path) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ChronosEmbedder");
    Ort::SessionOptions session_options;
    
    session_ = std::make_unique<Ort::Session>(
        env, 
        onnx_path.c_str(), 
        session_options
    );
    
    memory_info_ = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, 
        OrtMemTypeDefault
    );
}

std::vector<float> ChronosEmbedder::embed(const Event& event) {
    // 1. Prepare input (83 features)
    std::vector<float> input_data = event.features;
    
    // 2. Create input tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        input_data.data(),
        input_data.size(),
        input_shape_.data(),
        input_shape_.size()
    );
    
    // 3. Run inference
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );
    
    // 4. Extract output (512-d vector)
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    return std::vector<float>(output_data, output_data + 512);
}
```

**Models Strategy:**
```bash
# Option 1: Use placeholder ONNX (for compilation/testing)
# Create simple identity network that outputs fixed dimensions

# Option 2: Download pre-trained from HuggingFace
# Search for time-series and sentence embedding models

# Option 3: Train custom (Week 6)
# Use synthetic data + RandomForest features
```

**AttackEmbedder Special Feature:**
```cpp
// Benign sampling (only 10% of benign events)
bool AttackEmbedder::should_embed(const Event& event) const {
    if (event.classification == "Benign") {
        return (rand() % 100) < 10;  // 10% sampling
    }
    return true;  // All malicious events
}
```

**Success Criteria:**
- ✅ ONNX models load successfully
- ✅ Inference <10ms per event (single-threaded)
- ✅ Correct output dimensions (512, 384, 256)
- ✅ Memory-efficient (reuse tensors)
- ✅ Thread-safe initialization

---

## 🔒 CRITICAL SECURITY DECISION (ADR-001)

### Encryption is NOT Optional

**Problem:** Config files could be modified to disable encryption:
```json
// ❌ INSECURE:
{
  "ingester": {
    "input": {
      "encrypted": false,  // ← Attacker sets to false
      "compressed": false
    }
  }
}
```

**Attack Scenario:**
1. Attacker gains write access to config file
2. Sets `encrypted: false`
3. Injects poisoned events in plaintext
4. rag-ingester accepts them
5. FAISS indices poisoned
6. System compromised

**Solution:** Remove config options, hardcode encryption:
```cpp
// ml-detector (ALWAYS encrypts)
void RAGLogger::log_event(const NetworkSecurityEvent& event) {
    auto serialized = serialize(event);
    auto compressed = compress(serialized);  // NOT optional
    auto encrypted = encrypt(compressed);    // NOT optional
    write(encrypted);
}

// rag-ingester (ALWAYS decrypts)
Event EventLoader::load(const std::string& filepath) {
    auto data = read_file(filepath);
    auto decrypted = decrypt(data);  // If fails → SecurityException
    auto decompressed = decompress(decrypted);
    return parse(decompressed);
}
```

**Config Fields REMOVED:**
```json
// ✅ SECURE (after refactor):
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

**Enforcement:**
- Code-level contract, not config-level
- Plaintext events rejected with SecurityException
- No "debug mode" bypassing security
- Compliance: Encryption at rest mandatory

---

## 🏗️ ARCHITECTURE CONTEXT

### Pipeline Flow (End-to-End)

```
┌────────────────┐
│   ml-detector  │
│   (sniffer)    │
└───────┬────────┘
        │
        │ 1. Capture packet
        │ 2. Extract 83 features
        │ 3. RandomForest classify
        │ 4. Serialize protobuf
        │ 5. Compress (LZ4)
        │ 6. Encrypt (ChaCha20)
        │ 7. Write .pb file
        ↓
┌────────────────────────────┐
│ /vagrant/logs/rag/events/  │
│   event_12345.pb           │ ← Encrypted + Compressed
└───────┬────────────────────┘
        │
        │ inotify detects
        ↓
┌────────────────┐
│ rag-ingester   │
│  (this)        │
└───────┬────────┘
        │
        │ 8. FileWatcher callback
        │ 9. EventLoader decrypt
        │ 10. EventLoader decompress
        │ 11. Parse protobuf (83 features)
        ↓
┌────────────────┐
│  Event struct  │
│  {             │
│    id: 12345   │
│    features: [83 floats]
│    class: "Ransomware"
│  }             │
└───────┬────────┘
        │
        │ Day 37 (NEXT)
        ↓
┌────────────────┐
│ ChronosEmbedder│ → 512-d vector
│ SBERTEmbedder  │ → 384-d vector
│ AttackEmbedder │ → 256-d vector
└───────┬────────┘
        │
        │ Day 38
        ↓
┌────────────────┐
│ PCA Reduction  │ → 128-d, 96-d, 64-d
└───────┬────────┘
        │
        │ Day 38
        ↓
┌────────────────┐
│ FAISS Indices  │
│  - Chronos     │ 128-d
│  - SBERT       │ 96-d
│  - Entity-B    │ 64-d
│  - Entity-M    │ 64-d
└────────────────┘
```

---

## 🌍 GAIA VISION (Context for Future)

### Hierarchical Architecture

**Nivel 1 (Local) - Edificio:**
- 1 etcd-server + 1 RAG-master
- N RAG-clients (plantas)
- M ml-detectors (1:1 con RAG-clients)
- Decisions: Immediate, local
- Propagation: Upward (to Campus)

**Nivel 2 (Campus) - Grupo de Edificios:**
- 1 etcd-server (HA) + 1 RAG-master
- Aggregates: 5-10 edificios
- Decisions: Campus-wide policies
- Propagation: Bidirectional (up/down)
- NO lateral awareness (isolated campus)

**Nivel 3 (Global) - Organización:**
- 1 etcd-server (HA cluster) + 1 RAG-master
- Aggregates: All campus
- Decisions: Global threat response
- Propagation: Top-down (global vaccines)
- Authority: Maximum, override local if critical

### Vaccine Distribution Flow

**Local Threat:**
```
Planta 2 (Edificio 1) detecta ransomware
  → RAG-master Local valida
  → Distribute to all plantas (Edificio 1)
  → Time: <30 seconds
  → Scope: Local only
```

**Campus Threat:**
```
2 edificios (Campus A) mismo ransomware
  → RAG-master Campus correlaciona
  → Distribute to 5 edificios (Campus A)
  → Time: <5 minutes
  → Scope: Campus only (NO lateral to Campus B)
```

**Global Threat (APT):**
```
Multiple campus, same actor
  → RAG-master Global correlaciona
  → Override authority: Distribute ALL
  → Time: <15 minutes
  → Scope: Global (all campus, edificios, plantas)
```

---

## 🔧 TECHNICAL DEBT & KNOWN ISSUES

### Thread-Local FlowManager Bug (ml-detector)

**Status:** Documented, fix postponed  
**Impact:** Only 11/102 features captured currently  
**Workaround:** PCA trained for 102-feature schema (synthetic data)  
**Plan:** Fix in Week 6, re-train PCA with real data

**NOT blocking rag-ingester:**
- Can process 11-feature OR 102-feature events
- Zero-padding strategy for missing features
- Will scale automatically when sniffer fixed

### ISSUE-005: RAGLogger Memory Leak (ml-detector)

**Status:** Identified, not fixed  
**Impact:** ml-detector requires restart every ~3 days  
**Root Cause:** nlohmann/json allocations  
**Solution:** Replace with RapidJSON (2-3 days work)  
**Priority:** Medium (does NOT block FAISS work)

---

## 📚 KEY DOCUMENTS

### Current Component
- `/vagrant/rag-ingester/docs/BACKLOG.md` - Vision & roadmap (UPDATED Day 36)
- `/vagrant/rag-ingester/README.md` - Build & run instructions
- `/vagrant/rag-ingester/config/rag-ingester.json` - Configuration
- `/vagrant/rag-ingester/docs/design/` - Architecture diagrams

### Related Components
- `/vagrant/sniffer/` - ml-detector (produces encrypted .pb files)
- `/vagrant/etcd-client/` - Service discovery library
- `/vagrant/crypto-transport/` - Encryption/compression library
- `/vagrant/common-rag-ingester/` - PCA dimensionality reduction
- `/vagrant/protobuf/network_security.proto` - Event schema (THE LAW)

### Compilation Guides
- `/mnt/user-data/outputs/GUIA_ETCD_CRYPTO_INTEGRATION.md` - Crypto integration
- `/mnt/user-data/outputs/EJEMPLO_ML_DETECTOR_CRYPTO.md` - ml-detector example

### Bug Reports
- `/vagrant/docs/bugs/2025-01-10_thread_local_flowmanager_bug.md`

---

## 💡 COLLABORATION CONTEXT

**Philosophy:** Via Appia Quality - Build to last 2000 years

**Working with:**
- Claude (AI Co-author) - Architecture & implementation
- DeepSeek, Grok, Qwen, ChatGPT (Peer review)
- ALL credited as co-authors in academic papers

**Goal:** Democratize enterprise-grade security
- Hospitals: Protect patient data
- Schools: Safe learning environments
- Small businesses: Affordable cybersecurity

**Design Constraints:**
- Must run on Raspberry Pi 4 (4GB RAM)
- Must scale to 64-core servers
- Memory target: <500MB (100K events)
- Latency target: <500ms per event

**Transparent AI Collaboration:**
- AI systems credited as co-authors (not just "acknowledged")
- Methodology documented for reproducibility
- Open source (GPL v3)

---

## 🎯 SUCCESS CRITERIA

### Day 37 (ONNX Embedders - NEXT)
- [ ] ONNX Runtime integrated
- [ ] ChronosEmbedder: 83→512-d (<10ms inference)
- [ ] SBERTEmbedder: 83→384-d (<10ms inference)
- [ ] AttackEmbedder: 83→256-d + benign sampling
- [ ] Unit tests for embedders
- [ ] Memory efficient (reuse tensors)
- [ ] Thread-safe initialization

### Week 5 (Phase 2A - Days 35-40)
- [x] Structure complete (Day 35)
- [x] Compilation successful (Day 36)
- [x] Crypto integration (Day 36)
- [ ] Embedders functional (Day 37)
- [ ] FAISS indices operational (Day 38)
- [ ] Health monitoring (Day 39)
- [ ] etcd registration (Day 40)
- [ ] <500ms latency per event

---

## 🚀 COMMANDS FOR NEXT SESSION

```bash
# Navigate to project
cd /vagrant/rag-ingester

# Verify current status
./build/rag-ingester ../config/rag-ingester.json
# Should show: ✅ RAG Ingester ready and waiting for events

# Create ONNX models directory
mkdir -p models/onnx models/pca

# Check ONNX Runtime
pkg-config --modversion onnxruntime
# Should show: 1.15.1 (or similar)

# Compile after Day 37 changes
cd build
make -j$(nproc)

# Run tests
./test_file_watcher   # 7/7 ✅
./test_event_loader   # 7/7 ✅
# TODO Day 37: ./test_chronos_embedder

# Generate test .pb file (when ml-detector updated)
cd /vagrant/sniffer/build
sudo ./sniffer --test-mode
```

---

## 📊 PROGRESS TRACKER

```
Phase 2A: Foundation (Week 5)
├── Day 35: Skeleton        [████] 100% ✅
├── Day 36: Crypto          [████] 100% ✅
├── Day 37: Embedders       [░░░░]   0% ← NEXT
├── Day 38: Multi-Index     [░░░░]   0%
├── Day 39: Health Monitor  [░░░░]   0%
└── Day 40: etcd Integration[░░░░]   0%

Overall Phase 2A: 33% complete (2/6 days)
```

**Compilation Status:**
```
[████] Structure      100% ✅ (Day 35)
[████] Dependencies   100% ✅ (Day 35)
[████] Tests          100% ✅ (Day 35-36)
[████] Crypto         100% ✅ (Day 36)
[░░░░] Embedders        0% ← Day 37
[░░░░] FAISS            0%   Day 38
[░░░░] Health           0%   Day 39
[░░░░] etcd             0%   Day 40
```

---

## 🏛️ VIA APPIA REMINDERS

1. **Foundation first, always**
    - Days 35-36: Structure + Crypto ✅
    - Day 37: Processing (Embedders)
    - Days 38-40: Integration (FAISS + etcd)

2. **Security by design**
    - Encryption mandatory (ADR-001) ✅
    - No config-level bypasses ✅
    - Poison log prevention ✅

3. **Test-driven development**
    - Every component has unit tests
    - Integration tests before moving on
    - 14/14 tests passing currently ✅

4. **Raspberry Pi as baseline**
    - If it works on Pi, it works anywhere
    - Memory-conscious design from day 1
    - <500MB target

5. **Measure before optimize**
    - Single-threaded first (Days 36-40)
    - Multi-threaded only when needed (Week 6)
    - Profile with real data

6. **Document exhaustively**
    - ADRs for every architectural decision
    - Future maintainers thank us
    - AI collaboration transparent

---

## 📝 IMPORTANT FILES FOR DAY 37

**To Implement:**
- `src/embedders/chronos_embedder.cpp` - ONNX inference (83→512)
- `src/embedders/sbert_embedder.cpp` - ONNX inference (83→384)
- `src/embedders/attack_embedder.cpp` - ONNX inference (83→256) + sampling

**To Test:**
- `tests/test_chronos_embedder.cpp` - Unit tests for embedders
- Verify output dimensions
- Verify inference speed <10ms

**Models Needed:**
- `models/onnx/chronos.onnx` - Time-series embedder
- `models/onnx/sbert.onnx` - Semantic embedder
- `models/onnx/attack.onnx` - Attack-specific embedder

---

**End of Continuation Prompt**

**Ready for Day 37:** ONNX Runtime Embedders  
**Dependencies:** ONNX Runtime (installed), ONNX models (need to prepare)  
**Expected Duration:** 4-6 hours  
**Blockers:** None (crypto working, compilation clean)

🏛️ Via Appia: Days 35-36 complete, crypto integrated, security hardened, ready for embedders.
🔒 Security: Encryption mandatory, poison log prevention enforced.