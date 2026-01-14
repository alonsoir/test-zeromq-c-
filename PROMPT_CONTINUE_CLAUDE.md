# RAG Ingester - Continuation Prompt
**Last Updated:** 14 Enero 2026 - Day 38 (Parcial Complete)  
**Phase:** 2A - Foundation + Synthetic Data Generation  
**Status:** ✅ Generator Compiled | ⏳ ONNX Embedders Pending

---

## 📍 CURRENT STATE (14 Enero 2026 - Evening)

### ✅ Day 38 Achievements (TODAY) - Synthetic Event Generator

**Tools Infrastructure - COMPLETADO:**
- ✅ `/vagrant/tools/` directory structure established
- ✅ `generate_synthetic_events.cpp` implemented (850 lines)
- ✅ Config: `synthetic_generator_config.json` created
- ✅ CMakeLists.txt: Correct protobuf + etcd-client linking
- ✅ Makefile integration: `make tools-build` functional
- ✅ Binary compiled: `/vagrant/tools/build/generate_synthetic_events`

**100% Compliance Architecture:**
```
generate_synthetic_events
├─> etcd-client (get encryption_seed from etcd)
├─> crypto_manager (SAME key as ml-detector)
├─> RAGLogger (SAME code as production)
└─> Output: IDENTICAL to ml-detector (.pb.enc)
```

**Key Design Decisions:**
1. **No hardcoded keys** - Uses etcd like ml-detector
2. **Zero drift** - Reuses production RAGLogger directly
3. **101 features + provenance** - Full ADR-002 compliance
4. **Realistic distributions:**
    - 20% malicious, 80% benign
    - Discrepancy: 78% low, 12% medium, 10% high
    - Reason codes: SIG_MATCH (40%), STAT_ANOMALY (35%), etc.

**Features Generated:**
```cpp
// 101 features: 61 basic + 40 embedded
features.basic_flow = [61];    // TCP/IP statistics
features.ddos = [10];          // DDoS signatures
features.ransomware = [10];    // Ransomware patterns
features.traffic = [10];       // Traffic classification
features.internal = [10];      // Internal anomaly

// Provenance (ADR-002)
verdict.sniffer = {engine: "fast-path-sniffer", confidence: 0.9, reason: "SIG_MATCH"}
verdict.rf = {engine: "random-forest-level1", confidence: 0.85, reason: "STAT_ANOMALY"}
discrepancy_score = 0.15  // Low (agreement)
```

**Compilation Fixes Applied:**
- ❌ Initial: `-lnetwork_security_proto` (library doesn't exist)
- ✅ Fixed: Compile `network_security.pb.cc` directly
- ❌ Initial: Missing etcd-client symbols
- ✅ Fixed: Added `etcd_client.cpp` + OpenSSL + CURL
- ❌ Initial: Reason codes as enum constants
- ✅ Fixed: Use strings directly ("SIG_MATCH", etc.)

---

### 📋 Day 37 Context (Previous Session)

**ADR-002: Multi-Engine Provenance - COMPLETADO**
- ✅ Protobuf contract extended
- ✅ Sniffer, ml-detector, rag-ingester updated
- ✅ `reason_codes.hpp` created
- ✅ End-to-end provenance pipeline working

**ADR-001: Encryption Mandatory - COMPLETADO**
- ✅ RAGLogger encrypts artifacts (.pb.enc)
- ✅ Pipeline: Serialize → Compress → Encrypt
- ✅ No configuration flags (security hardcoded)

---

## 🎯 DAY 38 - REMAINING TASKS (Tomorrow Morning)

### Overview
**Duración estimada:** 4-5 horas  
**Estado:** Generador compilado ✅ | Ejecución pendiente ⏳

---

## 📋 SESIÓN MAÑANA: Execution + ONNX Embedders (4-5 horas)

### 1. Prerequisite: etcd-server Setup (30 min)

**Verify encryption seed exists:**
```bash
# Check if etcd-server is running
make etcd-server-status

# Check if encryption_seed exists
vagrant ssh
ETCDCTL_API=3 etcdctl get /crypto/ml-detector/tokens/encryption_seed
```

**If not exists, create it:**
```bash
# Generate 32-byte key (64 hex chars)
openssl rand -hex 32 > /tmp/encryption_seed.txt

# Store in etcd
ETCDCTL_API=3 etcdctl put /crypto/ml-detector/tokens/encryption_seed $(cat /tmp/encryption_seed.txt)

# Verify
ETCDCTL_API=3 etcdctl get /crypto/ml-detector/tokens/encryption_seed
```

**Success Criteria:**
- ✅ etcd-server running
- ✅ encryption_seed present (64 hex chars)
- ✅ Same key used by ml-detector

---

### 2. Execute Synthetic Generator (30 min)

**Run generator:**
```bash
cd /vagrant/tools/build

# Generate 100 events (20% malicious)
./generate_synthetic_events 100 0.20

# Or use custom config
./generate_synthetic_events 200 0.25 /vagrant/tools/config/synthetic_generator_config.json
```

**Expected Output:**
```
╔════════════════════════════════════════════════════════════╗
║  Synthetic Event Generator - Via Appia Quality             ║
║  100% Compliance: etcd + RAGLogger + crypto-transport      ║
╚════════════════════════════════════════════════════════════╝

📋 Loading configuration from: /vagrant/tools/config/synthetic_generator_config.json
✅ Configuration loaded

🔗 [etcd] Initializing connection to localhost:2379
✅ [etcd] Connected and registered

🔑 [crypto] Retrieving encryption seed from etcd...
🔑 [crypto] Retrieved encryption seed (64 hex chars)
✅ [crypto] Encryption key converted: 32 bytes
✅ [crypto] CryptoManager initialized (ChaCha20-Poly1305 + LZ4)

🔒 Generating 100 synthetic events...

   [  1/100] DDoS         | disc=0.156
   [  2/100] Benign       | disc=0.089
   ...
   [100/100] Benign       | disc=0.123

═══════════════════════════════════════════════════════════
📊 SYNTHETIC DATASET SUMMARY
═══════════════════════════════════════════════════════════

✅ Total events: 100
   Malicious: 20 (20.0%)
   Benign: 80 (80.0%)

📈 Attack Types:
   DDoS: 12
   Ransomware: 8

🎯 Discrepancy Distribution:
   Low (0.0-0.25): 78 (78%)
   Medium (0.25-0.5): 12 (12%)
   High (0.5-1.0): 10 (10%)

📁 RAGLogger Statistics:
   Events logged: 100
   Current log: /vagrant/logs/rag/synthetic/events/2026-01-15.jsonl

📄 SPEC saved: /vagrant/logs/rag/synthetic/SPEC.json

✅ Synthetic dataset generation complete!
═══════════════════════════════════════════════════════════
```

**Verify files:**
```bash
# JSONL event log
ls -lh /vagrant/logs/rag/synthetic/events/

# Encrypted artifacts
ls -lh /vagrant/logs/rag/synthetic/artifacts/2026-01-15/
# Should see: event_000000.pb.enc, event_000001.pb.enc, ...

# SPEC (ground truth)
cat /vagrant/logs/rag/synthetic/SPEC.json | jq .
```

**Success Criteria:**
- ✅ 100 files generated: `.pb.enc` artifacts
- ✅ JSONL log created with metadata
- ✅ SPEC.json contains feature/provenance definitions
- ✅ No errors during generation
- ✅ Encrypted files are non-plaintext (binary)

---

### 3. Update ONNX Embedders (103 features) (2-3 horas)

**Goal:** Extend embedders from 101 → 103 features

**Changes needed:**
```cpp
// include/embedders/chronos_embedder.hpp
class ChronosEmbedder {
public:
    static constexpr size_t INPUT_DIM = 103;  // Was 101
    static constexpr size_t OUTPUT_DIM = 512;
    
    std::vector<float> embed(const Event& event);
};
```

**Implementation pattern (all 3 embedders):**
```cpp
// src/embedders/chronos_embedder.cpp
std::vector<float> ChronosEmbedder::embed(const Event& event) {
    std::vector<float> input;
    input.reserve(INPUT_DIM);  // 103
    
    // 1. Original 101 features
    input.insert(input.end(), event.features.begin(), event.features.end());
    
    // 2. NEW Meta-features from ADR-002
    input.push_back(event.discrepancy_score);                    // Feature 102
    input.push_back(static_cast<float>(event.verdicts.size()));  // Feature 103
    
    // Validation
    if (input.size() != INPUT_DIM) {
        throw std::runtime_error(
            "Invalid input size: " + std::to_string(input.size()) + 
            " (expected " + std::to_string(INPUT_DIM) + ")"
        );
    }
    
    // ONNX inference (unchanged)
    auto input_tensor = create_tensor(input);
    auto output_tensor = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), 1
    );
    
    float* output_data = output_tensor[0].GetTensorMutableData<float>();
    return std::vector<float>(output_data, output_data + OUTPUT_DIM);
}
```

**Files to modify:**
1. `/vagrant/rag-ingester/include/embedders/chronos_embedder.hpp`
2. `/vagrant/rag-ingester/src/embedders/chronos_embedder.cpp`
3. `/vagrant/rag-ingester/include/embedders/sbert_embedder.hpp` (103 → 384)
4. `/vagrant/rag-ingester/src/embedders/sbert_embedder.cpp`
5. `/vagrant/rag-ingester/include/embedders/attack_embedder.hpp` (103 → 256)
6. `/vagrant/rag-ingester/src/embedders/attack_embedder.cpp`

**Compilation:**
```bash
cd /vagrant/rag-ingester/build
cmake ..
make -j$(nproc)
```

**Success Criteria:**
- ✅ All 3 embedders accept 103 features
- ✅ Input validation passes
- ✅ Output dimensions unchanged (512/384/256)
- ✅ Clean compilation (0 errors)

---

### 4. End-to-End Smoke Test (1 hora)

**Test pipeline:**
```bash
# Terminal 1: Start rag-ingester
cd /vagrant/rag-ingester/build
./rag-ingester ../config/rag-ingester.json

# Watch logs in Terminal 2
tail -f /vagrant/logs/rag-ingester/rag-ingester.log

# Expected log output:
# [INFO] Event loaded: synthetic_000000
# [INFO] Provenance: 2 verdicts, discrepancy=0.156
# [INFO] ChronosEmbedder: 103 → 512-d, norm=23.45
# [INFO] SBERTEmbedder: 103 → 384-d, norm=18.32
# [INFO] AttackEmbedder: 103 → 256-d, norm=15.67
# [INFO] FAISS: Added event to index
```

**Verify:**
```bash
# Check logs for errors
grep ERROR /vagrant/logs/rag-ingester/rag-ingester.log
# Should be empty

# Check provenance parsing
grep "verdicts" /vagrant/logs/rag-ingester/rag-ingester.log
# Should show: "2 verdicts", "discrepancy_score"

# Check embeddings generated
grep "Embedding" /vagrant/logs/rag-ingester/rag-ingester.log | wc -l
# Should be: 100 events * 3 embedders = 300 lines
```

**Success Criteria:**
- ✅ rag-ingester loads 100 synthetic events
- ✅ Provenance parsed correctly (2 verdicts per event)
- ✅ Embeddings generated without errors
- ✅ No crashes during processing
- ✅ FAISS index populated

---

## 🐛 TECHNICAL DEBT (Pending - Day 39+)

### ISSUE-007: Magic Numbers → JSON Config
**File:** `/vagrant/ml-detector/src/zmq_handler.cpp`
**Lines:** 332, 365
**Fix:** Move thresholds to `ml_detector_config.json`
**Priority:** Medium
**Estimación:** 30 min

### ISSUE-006: Log Files Not Persisted
**Impact:** Monitor scripts can't tail logs
**Fix:** Configure spdlog with rotating file sinks
**Priority:** Medium
**Estimación:** 1 hour

### ISSUE-005: RAGLogger Memory Leak (Known)
**Impact:** Restart every 3 days
**Root Cause:** nlohmann/json allocations
**Fix:** Migrate to RapidJSON
**Priority:** Medium
**Estimación:** 2-3 days

### ISSUE-003: Thread-Local FlowManager Bug (Known)
**Impact:** Only 11/102 features captured
**Workaround:** PCA trained on synthetic data
**Fix:** Remove thread_local or add mutex
**Priority:** HIGH
**Estimación:** 1-2 days

---

## ✅ CHECKLIST DEL DÍA 38 (Updated)

### Mañana (Synthetic Data Execution):
- [x] Generador compilado exitosamente
- [x] CMakeLists.txt corregido
- [x] Integración etcd completa
- [ ] etcd-server con encryption_seed verificado
- [ ] Generador ejecutado (100 eventos)
- [ ] Archivos .pb.enc validados

### Tarde (ONNX Embedders):
- [ ] ChronosEmbedder actualizado (103 → 512)
- [ ] SBERTEmbedder actualizado (103 → 384)
- [ ] AttackEmbedder actualizado (103 → 256)
- [ ] rag-ingester compila con 103 features
- [ ] End-to-end smoke test PASS

---

## 🎯 Success Criteria Day 38

**Synthetic Data Generation:**
- ✅ Generator compiled with etcd integration
- ⏳ 100+ eventos .pb.enc generados
- ⏳ Encryption + Compression verificados
- ⏳ Provenance completa en cada evento

**ONNX Embedders:**
- ⏳ 103 features procesadas correctamente
- ⏳ Output dimensions verificadas (512/384/256)
- ⏳ Validation errors capturados

**End-to-End:**
- ⏳ rag-ingester procesa sintéticos sin errors
- ⏳ Provenance parseada correctamente
- ⏳ Embeddings generados con normas razonables

---

## 🔒 CRITICAL REMINDERS

**100% Compliance:**
- ✅ Generator uses same etcd integration as ml-detector
- ✅ Generator uses same RAGLogger (zero drift)
- ✅ Generator uses same crypto_manager
- ✅ Output format identical to production

**Security:**
- ✅ No hardcoded keys (etcd-based)
- ✅ Encryption mandatory (ADR-001)
- ✅ Provenance preserved (ADR-002)

**Quality:**
- ✅ Via Appia: Reuse production code
- ✅ Test tools as robust as production
- ✅ If RAGLogger changes, generator inherits automatically

---

## 🏛️ VIA APPIA REMINDERS

1. **Zero Drift** - Generador usa código de producción ✅
2. **Security by Design** - Clave desde etcd, no hardcoded ✅
3. **Test before Scale** - Sintéticos antes de datos reales ✅
4. **Foundation Complete** - Compilación exitosa antes de ejecución ✅
5. **Measure before Optimize** - End-to-end funcional antes de optimizar

---

**End of Continuation Prompt**

**Ready for Day 38 Completion:** Execute generator → Update embedders → E2E test  
**Dependencies:** etcd-server with encryption_seed  
**Expected Duration:** 4-5 hours  
**Blockers:** None (generator compiled, ready to run)

🏛️ Via Appia: Day 38 parcial complete - Generator compiled with 100% production compliance, ready for execution and ONNX updates.