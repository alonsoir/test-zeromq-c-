## 🚨 CRITICAL ITEMS

### 1. Thread-Local FlowManager Bug - ROOT CAUSE IDENTIFIED & DOCUMENTED ✅

**Status:** DOCUMENTED, fix postponed for next week  
**Priority:** P0 - CRITICAL  
**Date Discovered:** 10 Enero 2025  
**Documentation:** Complete root cause analysis available

**Root Cause:**
```
thread_local FlowManager = per-thread instances
Thread A: add_packet() → FlowManager_A (has data)
Thread B: get_flow_stats() → FlowManager_B (EMPTY!)
Result: NULL flow_stats → features not populated → only 11/102 features
```

**Impact:**
- ❌ Real feature capture blocked (11/102 features)
- ⚠️ PCA training: UNBLOCKED (trained with synthetic 102-feature schema)
- ⚠️ FAISS integration: Can proceed with synthetic PCA, will update with real later

**Workaround Implemented (Day 36):**
- ✅ PCA trained for complete 102-feature schema
- ✅ Synthetic data generation pipeline
- ✅ ONNX model ready for C++ inference
- 📋 Will re-train with real data when bug fixed

**Solutions Identified:**

**Option 1: Single-Threaded Processing (2-3h) - RECOMMENDED FIRST**
- Move populate_protobuf_event() to same thread as add_packet()
- Eliminate feature_processor_loop
- Pro: Quick unblock for real data collection
- Con: Temporary, not scalable

**Option 2: Hash Consistent Routing (2-3 days) - PRODUCTION FIX**
- Implement hash_flow() over 5-tuple
- Per-thread queues with flow affinity
- Dedicated processor threads
- Pro: Correct architecture, production-ready
- Con: Requires extensive testing

**Decision:**
- 🏛️ Via Appia Quality: Do it RIGHT, not FAST
- Implement Option 1 next week (unblock real data)
- Then implement Option 2 properly (production architecture)
- No rushing critical infrastructure

**Documentation:** `/vagrant/docs/bugs/2025-01-10_thread_local_flowmanager_bug.md`

**Tasks (Next Week):**
- [ ] Implement single-threaded fix (2-3h)
- [ ] Capture 100K real events with 102 features
- [ ] Re-train PCA with real data (expected 85-95% variance)
- [ ] Compare variance: synthetic (64%) vs real (85-95%)
- [ ] Update FAISS indexes with production PCA

---

### 3. ISSUE-005: RAGLogger Memory Leak - JSON Serialization 🔴

**Status:** IDENTIFIED, not yet fixed  
**Priority:** P0 - CRITICAL (operational impact)  
**Date Discovered:** ~Diciembre 2025  
**Symptoms:** ml-detector requires restart every ~3 days

**Problem:**
```
ml-detector uptime: ~72 hours → memory exhaustion → crash/restart needed
Root cause: Memory leak in RAGLogger JSON serialization
Library: nlohmann/json (current implementation)
Impact: Operational burden, service interruptions
```

**Evidence:**
- ml-detector stable for detection (20+ hours no issues)
- Memory grows continuously during RAG logging
- Restart clears memory, system works again
- Pattern repeats every 3 days

**Root Cause (Suspected):**
```cpp
// Current implementation in RAGLogger
nlohmann::json j;
j["event_id"] = event.id;
j["features"] = event.features;  // ← Suspected leak here
// ... many allocations per event
// No explicit cleanup, relying on destructor
```

**Probable Issues:**
1. nlohmann/json uses exceptions heavily (memory not freed on error paths)
2. Deep copying of protobuf structures
3. String allocations not properly released
4. High allocation rate (1000s events/sec)

**Solutions Evaluated:**

**Option 1: Fix nlohmann/json usage (1-2 days)**
- Audit current RAGLogger code
- Add explicit memory cleanup
- Use move semantics properly
- Profile with Valgrind/ASan
- Pro: Keep current library
- Con: May not fix fundamental library issue

**Option 2: Replace with RapidJSON (2-3 days) - RECOMMENDED**
- SAX-style API (less allocations)
- No exceptions (explicit error handling)
- Faster serialization
- Better memory control
- Pro: Proven in high-throughput systems
- Con: API changes required

**Option 3: Replace with simdjson (3-4 days)**
- SIMD-optimized (fastest)
- Modern C++17 API
- Excellent documentation
- Pro: Best performance
- Con: Newer library, less battle-tested

**Decision Factors:**
- RapidJSON: Used in Ericsson systems (battle-tested)
- simdjson: Excellent performance, growing adoption
- nlohmann/json: Convenient but allocation-heavy

**Recommendation:** Option 2 (RapidJSON)
- Proven reliability in telecom (like Ericsson)
- SAX API fits streaming use case
- No exceptions = predictable memory
- 2-3 days vs weeks of debugging nlohmann

**Impact Assessment:**
- **Does NOT block FAISS work** (can develop in parallel)
- **Does NOT block PCA training** (separate component)
- Blocks: Long-term unattended operation (>3 days)

**Implementation Plan (When Ready):**

**Day 1: Investigation & Preparation (4-6h)**
- [ ] Profile RAGLogger with Valgrind
- [ ] Identify exact leak location
- [ ] Install RapidJSON library
- [ ] Create test harness

**Day 2: Implementation (6-8h)**
- [ ] Replace nlohmann::json with RapidJSON SAX writer
- [ ] Update RAGLogger serialization code
- [ ] Maintain same JSONL output format
- [ ] Unit tests for JSON output correctness

**Day 3: Testing & Validation (4-6h)**
- [ ] Stress test: 72+ hours continuous operation
- [ ] Memory profiling: Verify no leak
- [ ] Performance comparison: Before/after
- [ ] Integration testing with rag component

**Success Criteria:**
- ✅ ml-detector runs >7 days without restart
- ✅ Memory stable (no continuous growth)
- ✅ Same or better throughput
- ✅ JSONL format unchanged (backward compatible)

**Workaround (Current):**
```bash
# Restart ml-detector every 3 days
# Automated with systemd timer (future)
systemctl restart ml-detector.service
```

**Documentation:**
- Issue tracking: ISSUE-005
- Related: RAGLogger component
- Code: `/vagrant/ml_detector/src/rag_logger.cpp`

**Tasks (When Prioritized):**
- [ ] Memory profiling session
- [ ] Choose JSON library (RapidJSON recommended)
- [ ] Implement replacement
- [ ] 72h+ stress testing
- [ ] Deploy and monitor

**Parallel Work:**
- ✅ Can develop FAISS while this is open
- ✅ Can fix thread-local bug independently
- ⚠️ Affects production stability

**Timeline:**
- Investigation: 4-6h
- Implementation: 6-8h
- Testing: 4-6h
- **Total: 2-3 days**

**Via Appia Note:**
> Memory leaks in long-running systems are unacceptable.  
> Better to replace library properly than patch symptoms.  
> RapidJSON proven in Ericsson - reliability over convenience. 🏛️

---

### 2. PCA Embedder Training - COMPLETE ✅

**Status:** Pipeline functional, ready for FAISS integration  
**Priority:** P0 - UNBLOCKED  
**Date Completed:** 10 Enero 2025  
**Effort:** ~3 hours (3 scripts, testing, documentation)

**Deliverables:**
```
✅ /vagrant/contrib/claude/pca_pipeline/
├── generate_training_data.py      # 100K samples × 102 features
├── train_pca_embedder.py           # PCA: 102 → 64 dims
├── convert_pca_to_onnx.py          # ONNX export
├── README.md                       # Documentation
└── models/
    ├── training_data.npz           # 76 MB
    ├── scaler.pkl                  # 2.9 KB
    ├── pca_model.pkl               # 55 KB
    ├── pca_embedder.onnx           # 28 KB ← Production model
    └── training_metrics.json       # Stats
```

**Results:**
- Dimensionality: 102 → 64 (37% reduction)
- Variance explained: 64.0% (synthetic data)
- Transform time: 1.08 μs/sample
- ONNX validation: PASSED
- Model size: 28 KB

**Note:** Variance lower than target (64% vs 90%) because:
- Synthetic data = uniform random (no natural correlations)
- Real data expected: 85-95% variance (feature correlations exist)
- Strategy: Re-train when sniffer bug fixed

**Next Steps:**
- [ ] Integrate pca_embedder.onnx into FAISS pipeline
- [ ] Test semantic search end-to-end
- [ ] Re-train with real data when available
- [ ] Performance comparison

---

## 🏗️ HIGH PRIORITY

### Multi-Threaded Architecture Refactor

**Epic:** "Production-Ready Hash Consistent Routing"  
**Priority:** HIGH  
**Effort:** 2-3 days  
**Target:** Week of 13-17 Enero 2025  
**Depends On:** Single-threaded fix first

**Current Plan:**
1. Week of 10-12 Jan: Single-threaded fix (quick unblock)
2. Week of 13-17 Jan: Hash routing (production architecture)

**Architecture:**
```
Hash Router (5-tuple)
  ↓
Dedicated Threads (flow affinity)
  ↓
thread_local FlowManager (correct usage)
```

**Deliverables:**
- [ ] hash_flow() implementation
- [ ] Routing logic in handle_event()
- [ ] dedicated_processor_loop(thread_id)
- [ ] sniffer.json update (threads: 4-8)
- [ ] Zero race conditions
- [ ] Performance >= single-threaded
- [ ] Documentation & ADR

---

## 📊 SPRINT GOALS

### Current Sprint (10-12 Enero 2025)
**Theme:** "Unblock Phase 2A - PCA Complete, FAISS Ready"

**Completed:**
- ✅ Thread-local bug root cause identified & documented
- ✅ PCA pipeline complete (synthetic data)
- ✅ ONNX model ready for C++ inference
- ✅ Feature contract documented (102 features)

**In Progress:**
- 🔄 FAISS Integration (Days 37-38)

**Must Have:**
- [ ] FAISS semantic search working
- [ ] End-to-end test with synthetic PCA
- [ ] Performance benchmarks

**Should Have:**
- [ ] Single-threaded sniffer fix
- [ ] Real data collection started

**Could Have:**
- [ ] PCA re-training with real data

---

### Next Sprint (13-17 Enero 2025)
**Theme:** "Production Data & Architecture"

**Must Have:**
- [ ] Hash consistent routing implemented
- [ ] Multi-threaded processing working
- [ ] Real PCA models (85-95% variance)
- [ ] FAISS indexes updated

**Should Have:**
- [ ] Load testing (10K+ pps)
- [ ] CPU/memory profiling

---

## 📈 PROGRESO VISUAL

Phase 1 Progress: [████████████████████] 100% (16/16 días)
Phase 2A Progress: [████░░░░░░░░░░░░░░░░]  20% (Week 5: Days 31-36 ✅)

Day 36 PCA Pipeline: [████] 100% ✅
- Data Generation:    [████] 100% ✅
- PCA Training:       [████] 100% ✅
- ONNX Conversion:    [████] 100% ✅
- Documentation:      [████] 100% ✅

Next Steps:
- FAISS Integration:  [░░░░]   0% ← Day 37-38
- Sniffer Fix:        [░░░░]   0% ← Week of 13-17 Jan
- Real PCA Training:  [░░░░]   0% ← After sniffer fix

---

## Last Updated

**Last Updated:** 10 Enero 2025 - Day 36 Complete  
**Next Session:** 11 Enero 2025 - FAISS Integration (Day 37)  
**COMPLETED:** PCA Embedder Pipeline (synthetic data)  
**DOCUMENTED:** Thread-local bug (postponed for proper fix)  
**NEXT:** FAISS semantic search integration

**Via Appia Note:**
> Day 36: Discovered bug, documented thoroughly, built workaround.  
> Pipeline validated with synthetic data.  
> Will re-train with real data when proper fix implemented.  
> Foundation first, expansion properly. 🏛️