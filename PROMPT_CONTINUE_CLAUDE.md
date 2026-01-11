# RAG Ingester - Continuation Prompt
**Last Updated:** 11 Enero 2026 - Day 35 Skeleton Complete  
**Phase:** 2A - Foundation  
**Status:** ✅ Structure complete, dependencies verified, ready for Day 36

---

## 📍 CURRENT STATE (11 Enero 2026)

### ✅ Day 35 Achievements

**Estructura Completa:**
```
/vagrant/rag-ingester/
├── src/          # 12 source files (stubs)
├── include/      # 12 header files (interfaces)
├── config/       # rag-ingester.json
├── tests/        # test_config_parser ✅
├── docs/         # BACKLOG.md, design docs
├── models/       # onnx/, pca/ (empty, ready)
└── CMakeLists.txt
```

**Compilación Exitosa:**
- ✅ CMake configuration successful
- ✅ All dependencies found:
    - `libetcd_client.so` → `/usr/local/lib/`
    - `libcrypto_transport.so` → `/usr/local/lib/`
    - `libcommon-rag-ingester.so` → `/vagrant/common-rag-ingester/build/`
    - `libfaiss.so` → `/usr/local/lib/`
    - `libonnxruntime.so` → `/usr/local/lib/`
- ✅ Binary compiles cleanly: `rag-ingester`
- ✅ Test suite passing: `test_config_parser`

**Binario Funcional:**
```bash
vagrant@bookworm:/vagrant/rag-ingester/build$ ./rag-ingester
[INFO] RAG Ingester Starting (Version: 0.1.0)
[INFO] Configuration loaded
[INFO] Service ID: rag-ingester-default
[INFO] Threading mode: single
[INFO] ✅ RAG Ingester ready and waiting for events
```

**Stubs Creados:**
- `FileWatcher` (inotify placeholder)
- `EventLoader` (crypto-transport integration)
- `ChronosEmbedder`, `SBERTEmbedder`, `AttackEmbedder` (ONNX stubs)
- `MultiIndexManager` (4 FAISS indices)
- `IndexHealthMonitor` (CV calculation)
- `ThreadPool` (generic worker pool)
- `ConfigParser` (JSON → Config struct) ✅ FUNCTIONAL

---

## 🎯 DAY 36 OBJECTIVES (Immediate Next)

### FileWatcher Implementation

**Goal:** Watch `/vagrant/logs/rag/events/*.pb` with inotify

**Implementation:**
```cpp
// include/file_watcher.hpp
class FileWatcher {
    int inotify_fd_;
    int watch_descriptor_;
    std::string directory_;
    std::string pattern_;
    Callback callback_;
    
    void start(Callback callback);
    void process_events();  // Main loop
    void stop();
};
```

**Key Points:**
- Use `inotify_init()`, `inotify_add_watch()`
- Watch for `IN_CLOSE_WRITE` events (file complete)
- Pattern matching: `*.pb`
- Non-blocking read with timeout
- Thread-safe callback invocation

**Test:**
```bash
# Terminal 1: Start ingester
./rag-ingester

# Terminal 2: Generate test file
echo "test" > /vagrant/logs/rag/events/test.pb

# Expected: FileWatcher detects and logs
```

---

### EventLoader Implementation

**Goal:** Decrypt + decompress `.pb` files using crypto-transport

**Implementation:**
```cpp
// include/event_loader.hpp
class EventLoader {
    std::unique_ptr<crypto::Decryptor> decryptor_;
    std::unique_ptr<crypto::Decompressor> decompressor_;
    
    std::vector<Event> load(const std::string& filepath);
    
private:
    std::vector<uint8_t> read_file(const std::string& path);
    std::vector<uint8_t> decrypt(const std::vector<uint8_t>& encrypted);
    std::vector<uint8_t> decompress(const std::vector<uint8_t>& compressed);
    std::vector<Event> parse_protobuf(const std::vector<uint8_t>& data);
};
```

**Crypto-Transport Integration:**
```cpp
#include <crypto_transport/stream_decryptor.hpp>
#include <crypto_transport/decompressor.hpp>

// Initialize (once)
decryptor_ = std::make_unique<crypto::StreamDecryptor>(
    crypto::Algorithm::AES_256_GCM,
    key_from_config
);

decompressor_ = std::make_unique<crypto::Decompressor>(
    crypto::CompressionType::GZIP
);

// Decrypt + decompress
auto decrypted = decryptor_->decrypt(encrypted_data);
auto decompressed = decompressor_->decompress(decrypted);
```

**Protobuf Parsing:**
```cpp
#include <network_security.pb.h>

std::vector<Event> parse_protobuf(const std::vector<uint8_t>& data) {
    ml_defender::NetworkEvent proto_event;
    
    if (!proto_event.ParseFromArray(data.data(), data.size())) {
        throw std::runtime_error("Failed to parse protobuf");
    }
    
    Event event;
    event.id = proto_event.event_id();
    event.features = extract_83_features(proto_event);
    event.classification = proto_event.classification().final_class();
    
    return {event};
}
```

**Success Criteria:**
- ✅ Read encrypted `.pb` file
- ✅ Decrypt successfully
- ✅ Decompress successfully
- ✅ Parse protobuf
- ✅ Extract 83 features
- ✅ Handle errors gracefully

---

## 🏗️ ARCHITECTURE CONTEXT

### Symbiosis with ml-detector

**ml-detector** (sniffer):
- Location: `/vagrant/sniffer/`
- Produces: Encrypted `.pb` files → `/vagrant/logs/rag/events/`
- Format: AES-256-GCM + gzip
- Rate: ~100 events/day (idle), 1000+ events/day (active)

**rag-ingester** (this component):
- Location: `/vagrant/rag-ingester/`
- Consumes: `.pb` files from ml-detector
- Processes: Decrypt → Decompress → Embed → Index (FAISS)
- Registers in etcd: `partner_detector: "ml-detector-default"`

**etcd-server Symbiosis:**
```json
// ml-detector registration
{
  "type": "ml-detector",
  "location": "default",
  "partner_ingester": "rag-ingester-default",
  "output_path": "/vagrant/logs/rag/events"
}

// rag-ingester registration
{
  "type": "rag-ingester",
  "location": "default",
  "partner_detector": "ml-detector-default",
  "input_path": "/vagrant/logs/rag/events"
}
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
**Impact:** Only 11/102 features captured in `.pb` files  
**Workaround:** PCA trained for 102-feature schema (synthetic data)  
**Plan:** Fix in Week 6, re-train PCA with real data

**NOT blocking rag-ingester development:**
- Can process 11-feature events now
- Will scale to 102 features when sniffer fixed
- Zero-padding strategy for missing features

### ISSUE-005: RAGLogger Memory Leak

**Status:** Identified, not fixed  
**Impact:** ml-detector requires restart every ~3 days  
**Root Cause:** nlohmann/json allocations  
**Solution:** Replace with RapidJSON (2-3 days work)  
**Priority:** Medium (does NOT block FAISS work)

---

## 📚 KEY DOCUMENTS

### Architecture
- `/vagrant/rag-ingester/docs/BACKLOG.md` - Vision & roadmap
- `/vagrant/rag-ingester/README.md` - Build & run instructions
- `/vagrant/rag-ingester/config/rag-ingester.json` - Configuration

### Related Components
- `/vagrant/sniffer/` - ml-detector (produces `.pb` files)
- `/vagrant/etcd-client/` - Service discovery library
- `/vagrant/crypto-transport/` - Encryption/compression library
- `/vagrant/common-rag-ingester/` - PCA dimensionality reduction
- `/vagrant/protobuf/network_security.proto` - Event schema (THE LAW)

### Bug Reports
- `/vagrant/docs/bugs/2025-01-10_thread_local_flowmanager_bug.md`

---

## 💡 COLLABORATION CONTEXT

**Philosophy:** Via Appia Quality - Build to last 2000 years

**Working with:**
- Claude (AI Co-author) - Architecture & implementation
- DeepSeek, Grok, Qwen, ChatGPT (Peer review)

**Goal:** Democratize enterprise-grade security
- Hospitals: Protect patient data
- Schools: Safe learning environments
- Small businesses: Affordable cybersecurity

**Design Constraints:**
- Must run on Raspberry Pi 4 (4GB RAM)
- Must scale to 64-core servers
- Memory target: <500MB (100K events)
- Latency target: <500ms per event

---

## 🎯 SUCCESS CRITERIA

### Day 36 (FileWatcher + EventLoader)
- [ ] inotify detects `.pb` files within 100ms
- [ ] Decryption successful (crypto-transport)
- [ ] Decompression successful (gzip)
- [ ] Protobuf parsing successful
- [ ] 83 features extracted (or 11 with current ml-detector)
- [ ] Unit tests passing
- [ ] No memory leaks (Valgrind clean)

### Week 5 (Phase 2A)
- [ ] End-to-end pipeline working (Days 35-40)
- [ ] Embedders functional (ONNX Runtime)
- [ ] 4 FAISS indices operational
- [ ] etcd registration & heartbeat
- [ ] <500ms latency per event

---

## 🚀 COMMANDS FOR NEXT SESSION
```bash
# Navigate to project
cd /vagrant/rag-ingester

# Check current status
./build/rag-ingester  # Should show TODO for FileWatcher

# Compile after changes
cd build
make -j$(nproc)

# Run tests
./tests/test_config_parser  # Should pass
# TODO: ./tests/test_file_watcher (create on Day 36)
# TODO: ./tests/test_event_loader (create on Day 36)

# Generate test .pb file (if ml-detector running)
cd /vagrant/sniffer/build
sudo ./sniffer --test-mode
```

---

## 📊 PROGRESS TRACKER
```
Phase 2A: Foundation (Week 5)
├── Day 35: Skeleton        [████] 100% ✅
├── Day 36: FileWatcher     [░░░░]   0% ← NEXT
├── Day 37: Embedders       [░░░░]   0%
├── Day 38: Multi-Index     [░░░░]   0%
├── Day 39: Health Monitor  [░░░░]   0%
└── Day 40: etcd Integration[░░░░]   0%
```

**Overall Phase 2A:** 10% complete (1/6 days)

---

## 🏛️ VIA APPIA REMINDERS

1. **Foundation first, always**
    - Day 35: Structure ✅
    - Day 36-37: I/O & Processing
    - Day 38-40: Integration & Monitoring

2. **Test-driven development**
    - Every component has unit tests
    - Integration tests before moving on

3. **Raspberry Pi as baseline**
    - If it works on Pi, it works anywhere
    - Memory-conscious design from day 1

4. **Document exhaustively**
    - Future maintainers (including future us) thank us
    - ADRs for every architectural decision

5. **Measure before optimize**
    - Single-threaded first (Day 36-40)
    - Multi-threaded only when needed (Week 6)
    - Profile with real data, not assumptions

---

**End of Continuation Prompt**

**Ready for Day 36:** FileWatcher + EventLoader  
**Dependencies:** crypto-transport, protobuf, inotify  
**Expected Duration:** 4-6 hours  
**Blockers:** None (all dependencies available)

🏛️ Via Appia: Foundation complete, now we build functionality brick by brick.
