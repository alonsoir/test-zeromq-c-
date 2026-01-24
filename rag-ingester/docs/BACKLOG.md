# RAG Ingester - Development Backlog

**Last Updated:** 2026-01-11  
**Current Phase:** 2A - Foundation

---

## 🎯 Vision: GAIA System (Sistema Inmunológico Global)

ML Defender no es solo un IDS - es un **sistema inmunológico distribuido** para redes empresariales.

### Architecture Vision
```
                    RAG-MASTER (GAIA)
            "Sistema linfático central - coordina respuesta global"
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   RAG-client          RAG-client          RAG-client
   Planta 1            Planta 2            Planta 3
        │                   │                   │
   ml-detector        ml-detector        ml-detector
   + ingester         + ingester         + ingester
```

**Flujo de "vacunación" (inmunidad de rebaño):**

1. RAG-client (Planta 3) detecta nuevo ransomware variant
2. RAG-master recibe alerta → LLM analiza → Operador valida
3. RAG-master genera "vacuna": nuevo embedding signature
4. Distribuye a TODOS los RAG-clients via etcd (prioridad: CRÍTICA)
5. Red inmunizada en <5 minutos

---

## 📅 Phase 2A - Foundation (Week 5: Days 35-40)

### ✅ Day 35 - Skeleton Complete (2026-01-11)

**Completado:**
- [x] Directory structure
- [x] CMakeLists.txt with all dependencies
- [x] Configuration parser (JSON → Config struct)
- [x] Main loop with signal handling
- [x] All stub files created
- [x] Test suite passing
- [x] Binary compiling and running

**Dependencies verified:**
- ✅ etcd_client: `/usr/local/lib/libetcd_client.so`
- ✅ crypto_transport: `/usr/local/lib/libcrypto_transport.so`
- ✅ common-rag-ingester: `/vagrant/common-rag-ingester/build/`
- ✅ FAISS: `/usr/local/lib/libfaiss.so`
- ✅ ONNX Runtime: `/usr/local/lib/libonnxruntime.so`

---

### 📋 Day 36 - File Watcher & Event Loader

**Goals:**
- [ ] Implement `FileWatcher` with inotify
- [ ] Watch `/vagrant/logs/rag/events/*.pb`
- [ ] Implement `EventLoader` with crypto-transport
- [ ] Decrypt + decompress .pb files
- [ ] Parse protobuf events (83 features)

**Test:**
```bash
# Generate test .pb file from ml-detector
cd /vagrant/sniffer
sudo ./build/sniffer  # Generate events

# Watch ingester consume them
cd /vagrant/rag-ingester/build
./rag-ingester
```

**Success criteria:**
- ✅ inotify detects new .pb files
- ✅ Decryption successful
- ✅ Decompression successful
- ✅ Protobuf parsing successful
- ✅ 83 features extracted

---

### 📋 Day 37 - Embedders (ONNX Runtime)

**Goals:**
- [ ] Download ONNX models (or train stubs)
- [ ] Implement `ChronosEmbedder` (83 → 512-d)
- [ ] Implement `SBERTEmbedder` (83 → 384-d)
- [ ] Implement `AttackEmbedder` (83 → 256-d)
- [ ] ONNX Runtime session initialization
- [ ] Batch inference support

**Models needed:**
```
models/onnx/chronos.onnx      # Time series embedder
models/onnx/sbert.onnx         # Semantic embedder
models/onnx/attack.onnx        # Attack-specific embedder
```

**Test:**
```bash
# Embed single event
Event event = load_test_event();
auto chronos_emb = chronos_embedder->embed(event);
assert(chronos_emb.size() == 512);
```

**Success criteria:**
- ✅ ONNX models loaded
- ✅ Inference <10ms per event
- ✅ Correct output dimensions (512, 384, 256)
- ✅ Batch processing functional

---

### 📋 Day 38 - PCA & Multi-Index Manager

**Goals:**
- [ ] Integrate `common-rag-ingester` PCA library
- [ ] Train PCA models (102→64, 384→96, 256→64)
- [ ] Implement `MultiIndexManager`
- [ ] Create 4 FAISS indices
- [ ] Implement eventual consistency logic

**PCA Training:**
```bash
cd /vagrant/common-rag-ingester
python3 scripts/train_pca.py --events 10000
```

**Index architecture:**
```cpp
chronos_index_         (128-d, IndexFlatL2)
sbert_index_           (96-d, IndexFlatL2)
entity_benign_index_   (64-d, IndexFlatL2, 10% sampling)
entity_malicious_index_(64-d, IndexFlatL2, 100% coverage)
```

**Success criteria:**
- ✅ PCA reduces dimensions correctly
- ✅ Variance retained >95%
- ✅ All 4 indices operational
- ✅ Best-effort commit working
- ✅ Partial failures handled gracefully

---

### 📋 Day 39 - Health Monitoring

**Goals:**
- [ ] Implement `IndexHealthMonitor`
- [ ] CV (Coefficient of Variation) calculation
- [ ] Alert when CV < 0.20 (degradation threshold)
- [ ] etcd health reporting

**Monitoring metrics:**
```cpp
struct HealthMetrics {
    double CV;              // Target: >0.20
    double mean_distance;
    double std_distance;
    size_t num_vectors;
    bool is_healthy() const { return CV > 0.2; }
};
```

**Success criteria:**
- ✅ CV calculated correctly
- ✅ Alerts trigger at thresholds
- ✅ Health reported to etcd every 10s

---

### 📋 Day 40 - etcd Integration & Symbiosis

**Goals:**
- [ ] Register in etcd with `partner_detector`
- [ ] Heartbeat every 10s
- [ ] Subscribe to ml-detector status
- [ ] Alert if partner fails

**etcd Registration:**
```json
PUT /ml-defender/services/rag-ingester-planta-3
{
  "type": "rag-ingester",
  "location": "planta-3",
  "partner_detector": "ml-detector-planta-3",
  "faiss_indices": {
    "chronos": { "vectors": 33000, "cv": 0.352 },
    "sbert": { "vectors": 33000, "cv": 0.42 }
  },
  "health": { "status": "healthy" }
}
```

**Success criteria:**
- ✅ Service visible in etcd
- ✅ Heartbeat maintains TTL
- ✅ Partner detection working
- ✅ Coordinated shutdown tested

---

## 📅 Phase 2B - Optimization (Week 6: Days 41-45)

### Day 41 - Multi-Threading

**Goals:**
- [ ] Enable parallel mode in config
- [ ] ThreadPool for embeddings (3 workers)
- [ ] ThreadPool for indexing (4 workers)
- [ ] Performance benchmarking

### Day 42 - Persistence

**Goals:**
- [ ] FAISS index save/load
- [ ] Checkpoint every 1000 events
- [ ] Graceful shutdown with persistence

### Day 43 - Advanced Strategies

**Goals:**
- [ ] Temporal tiers (hot/warm/cold)
- [ ] Metadata-first search
- [ ] Quantization (int8)

### Day 44 - Integration Testing

**Goals:**
- [ ] End-to-end pipeline test
- [ ] Performance benchmarks
- [ ] Memory profiling (target: <500MB)

### Day 45 - Documentation & Hardening

**Goals:**
- [ ] API documentation
- [ ] Deployment guide
- [ ] Troubleshooting guide

---

## 📅 Phase 3 - GAIA System (Weeks 7-8)

### RAG-Master Development

**Components:**
- [ ] Orchestrator service
- [ ] LLM validator (TinyLlama)
- [ ] Vaccine distributor
- [ ] Multi-client coordination

### Vaccine Distribution

**Workflow:**
1. Anomaly detected by RAG-client
2. LLM validates (reduce false positives)
3. Human operator confirms
4. Vaccine generated (new signature)
5. Distributed to all RAG-clients via etcd

### Features

- [ ] Global threat intelligence sharing
- [ ] Autonomous response (supervised)
- [ ] Federation across plants
- [ ] A/B testing of models

---

## 📅 Phase 4 - Post-Hardening (Future)

### Model Re-training

**Capabilities:**
- [ ] Continual learning from new threats
- [ ] A/B testing of model versions
- [ ] Automatic rollback on degradation

### Advanced Features

- [ ] GPU acceleration (CUDA)
- [ ] Distributed FAISS (cluster)
- [ ] Real-time model updates
- [ ] Threat intelligence APIs

---

## 🎓 Lessons Learned

### Day 35

1. ✅ **Library naming matters**: `libetcd_client.so` not `libetcd-client.so`
2. ✅ **Forward declarations**: Need full headers in `.cpp` for `unique_ptr<T>`
3. ✅ **System vs local libs**: Check `/usr/local/lib` first, then `/vagrant`
4. ✅ **Log permissions**: Use `/tmp` instead of `/var/log` to avoid sudo
5. ✅ **Via Appia principle**: Skeleton first, functionality incremental

---

## 📊 Success Metrics

### Phase 2A (Week 5)
- ✅ Compilation successful
- ✅ All tests passing
- ✅ Dependencies resolved
- [ ] End-to-end pipeline working
- [ ] <500ms latency per event

### Phase 2B (Week 6)
- [ ] Multi-threading operational
- [ ] Memory usage <500MB (100K events)
- [ ] CV metrics stable >0.20
- [ ] 10+ hours continuous operation

### Phase 3 (Weeks 7-8)
- [ ] RAG-master operational
- [ ] Vaccine distribution <5 min
- [ ] Multi-plant coordination
- [ ] Zero false negatives (malicious)

---

**End of Backlog**
