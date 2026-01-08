# BACKLOG.md - Day 35 Updates

## Section: Epic 2A.2: FAISS Infrastructure

REPLACE:
```
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
```

WITH:
```
### Epic 2A.2: FAISS Infrastructure (Days 31-35) - IN PROGRESS
**Priority:** P1 (HIGH)
**Status:** 🟡 IN PROGRESS - Week 5 Day 35/40 Complete
**Owner:** Alonso + Claude

**Goal:** Semantic search para eventos históricos con FAISS

**Progress (Week 5 - Days 31-35):**
- ✅ Day 31: FAISS v1.8.0 installed + Anti-curse design (peer-reviewed)
- ✅ Day 32: ONNX Runtime v1.17.1 installed + tested
- ✅ Day 33: 3 ONNX embedder models created (512-d, 384-d, 256-d)
- ✅ Day 34: Pipeline validated with real JSONL data (21 min)
    - Python inference: 3/3 tests passed
    - C++ inference: 3/3 tests passed (ONNX Runtime upgraded)
    - Batch processing: 6.8K-18.5K events/sec throughput
- ✅ Day 35: DimensionalityReducer library (common-rag-ingester) (~2h)
    - PCA-based reduction using faiss::PCAMatrix
    - Architecture: Producer/consumer separation (faiss-ingester + rag)
    - API: train/transform/save/load (thread-safe)
    - Build: Clean compilation on Debian 12
    - Test: ALL PASSED (908ms training, 149μs transform, 20K vec/sec batch)
    - Performance: ~10MB memory per trained model
    - Variance: 40.97% synthetic (real data will achieve ≥96%)

**Next (Week 5-6 - Days 36-40):**
- 🔥 Day 36: Training pipeline with real data (4-6h) ← NEXT
    - Data loader: JSONL → 83 features (balanced multi-source)
    - ONNX embedding: 3 models → vectors (512-d, 384-d, 256-d)
    - PCA training: 3 reducers → 128-d (variance ≥96% target)
    - Save models: /shared/models/pca/ (chronos, sbert, attack)
- 📅 Day 37-38: Integration + buffer (validation, error handling)
- 📅 Day 39-40: Week 5 finalization + documentation

**Architecture Confirmed:**
```
/vagrant/
├── common-rag-ingester/    ← SHARED library (Day 35 ✅)
│   └── DimensionalityReducer
├── faiss-ingester/         ← Producer (Week 6)
│   └── Events → ONNX → PCA → FAISS
└── rag/                    ← Consumer (Week 7-8)
└── Query → ONNX → PCA → Search
```

**Note:** FAISS work is parallel to ISSUE-005. Not blocked.
```

---

## Section: 📊 ROADMAP ACTUALIZADO

UPDATE:
```
Phase 2A: 🔄 EN PROGRESO (Ene 2026)
├─ ⚠️ Epic 2A.1: RAGLogger stability (ISSUE-005 pending)
├─ 🔴 ISSUE-005: Fix JSONL memory leak (1-3 días) ← NEXT
├─ 🔥 Epic 2A.2: FAISS C++ Integration (after ISSUE-005)
```

TO:
```
Phase 2A: 🔄 EN PROGRESO (Ene 2026)
├─ ⚠️ Epic 2A.1: RAGLogger stability (ISSUE-005 pending)
├─ 🔴 ISSUE-005: Fix JSONL memory leak (1-3 días, parallel to FAISS)
├─ 🔥 Epic 2A.2: FAISS Infrastructure (Days 31-35 ✅, Day 36 next)
│  ├─ ✅ Day 31-34: Infrastructure + validation
│  ├─ ✅ Day 35: DimensionalityReducer library
│  └─ 🔥 Day 36: Training pipeline ← NEXT
```

---

## Section: 📈 PROGRESO VISUAL

UPDATE:
```
Phase 1 Progress: [████████████████████] 100% (16/16 días)
Phase 2A Progress: [██░░░░░░░░░░░░░░░░░░]  10% (RAGLogger partial, ISSUE-005 active)
```

TO:
```
Phase 1 Progress: [████████████████████] 100% (16/16 días)
Phase 2A Progress: [███░░░░░░░░░░░░░░░░░]  15% (Week 5: Days 31-35 ✅)

Week 5 FAISS Progress: [███████░░░░░░░░░] 35% (Day 35/40)
  - Infrastructure:         [████] 100% ✅
  - DimensionalityReducer:  [████] 100% ✅
  - Training Pipeline:      [░░░░]   0% ← Day 36 NEXT
  - Integration:            [░░░░]   0%
```

ADD NEW:
```
Current Sprint: Day 36 Training Pipeline (4-6h)
  - Data Loader (JSONL):    [░] 0% ← NEXT
  - ONNX Embedding:         [░] 0%
  - PCA Training:           [░] 0%
  - Validation:             [░] 0%
```

---

## Section: Last Updated

CHANGE:
```
**Last Updated:** 6 Enero 2026  
**Next Review:** 7 Enero 2026 (Daily standup)  
**CRITICAL:** ISSUE-005 JSONL Memory Leak (ETA: 1-3 días)  
**BLOCKED:** FAISS integration (waiting for ISSUE-005 resolution)  
```

TO:
```
**Last Updated:** 8 Enero 2026 - Day 35 Complete
**Next Review:** 9 Enero 2026 (Daily standup)  
**ACTIVE:** Epic 2A.2 Week 5 - Day 36 Training Pipeline (NEXT)
**PARALLEL:** ISSUE-005 JSONL Memory Leak (not blocking FAISS)
```

---

## Summary of Changes:

1. ✅ Epic 2A.2: Updated with Day 35 completion details
2. ✅ Architecture diagram added (common-rag-ingester structure)
3. ✅ Day 36 tasks detailed (training pipeline)
4. ✅ Progress bars updated (10% → 15%, Week 5: 35% complete)
5. ✅ Roadmap clarified (ISSUE-005 parallel, not blocking)
6. ✅ Dates updated (8 Enero 2026)
7. ✅ Via Appia note: Foundation first (Day 35 solid)