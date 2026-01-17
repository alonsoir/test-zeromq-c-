# RAG Ingester - Development Backlog

ML Defender - Founding Principles
Co-authored by: Alonso Isidoro Roman (Human) + AI Collaborators

Purpose:
To democratize enterprise-grade cybersecurity protection for:
- Medical infrastructure (hospitals, clinics, care facilities)
- Educational institutions (schools, universities, research)
- Small businesses (the 99% of economic fabric)
- Critical civil infrastructure (water, power, emergency services)

Prohibited Uses:
- Offensive military operations
- Mass surveillance of civilian populations
- Support for authoritarian regimes
- Any use that prioritizes property over human life

Technical Philosophy:
- Via Appia Quality: Built to last, built to protect
- Zero Trust: Every component independently verifiable
- Explainability: ADR-002 ensures human understanding
- Open Design: Transparency prevents abuse

If this software ever protects:
- A newborn in an NICU from ransomware → We succeeded
- A small family business from bankruptcy → We succeeded
- A water treatment plant from sabotage → We succeeded

If it ever contributes to:
- Civilian casualties → We failed, regardless of legality
- Suppression of human rights → We failed
- Profit over protection → We failed

Signed:
Alonso Isidoro Roman, Lead Architect
Claude (Anthropic), AI Collaborator
[Other AI collaborators if applicable]

Date: [When Phase 2A completes]

**Last Updated:** 2026-01-15 (Day 38 - 75% Complete)  
**Current Phase:** 2A - Foundation + Synthetic Data  
**Next Session:** Day 38 Completion - Steps 4-5 (2.5h)

---

## 📊 Day 38 Status - 75% COMPLETE

### ✅ Completed Today (15 Enero 2026)

**Step 1: etcd-server Bootstrap** ✅
- etcd-server running (verified PID + HTTP 200)
- `/seed` endpoint responding
- Encryption seed retrieval working (64 hex chars)

**Step 2: Synthetic Event Generation** ✅
- 100 events generated successfully
- Distribution: 19% malicious, 81% benign
- Artifacts: 100 `.pb.enc` files encrypted + compressed
- RAGLogger: 0 errors, 100% success rate

**Step 3: Gepeto Validation PASSED** ✅
- Count: 100 `.pb.enc` verified
- **Dispersión Real:** Mean 0.244, StdDev 0.226 (> 0.1) ✅
- Distribution: 75% low, 14% medium, 11% high
- ADR-002 compliance: Full provenance present

### ⏳ Pending Tomorrow (15 Enero 2026 - Evening)

**Step 4: Update Embedders (2h)**
- [ ] chronos_embedder.hpp/cpp (101 → 103)
- [ ] sbert_embedder.hpp/cpp (101 → 103)
- [ ] attack_embedder.hpp/cpp (101 → 103)

**Step 5: Smoke Test (30min)**
- [ ] 100 events loaded
- [ ] 300 embeddings generated
- [ ] Invariant validated (disc > 0.5 ⇒ verdicts ≥ 2)
- [ ] No errors

**Estimated completion:** 2.5-3 hours

---

## 🔒 CRITICAL SECURITY DECISION: Mandatory Encryption

**ADR-001: Encryption is NOT Optional**

**Decision:** Encryption and compression are HARDCODED in the pipeline, NOT configurable.

**Rationale:**
- **Poison Log Prevention:** Attacker could disable encryption to inject malicious events
- **Data Integrity:** Compressed + encrypted data has built-in tamper detection
- **Compliance:** Enterprise security requires encryption at rest
- **No Backdoors:** No "debug mode" that bypasses security

**Implementation:** ✅ COMPLETE (Day 37)

**Validation:** ✅ TESTED (Day 38 - 100 encrypted artifacts)

---

## 🔍 ADR-002: Multi-Engine Provenance & Situational Intelligence

**Date:** 13 Enero 2026  
**Status:** ✅ IMPLEMENTED + VALIDATED  
**Decision:** Extend protobuf contract to capture multiple engine verdicts  
**Validation:** Day 38 - Real dispersion confirmed (StdDev: 0.226)

### Synthetic Data Validation (Day 38)

**Dispersion Metrics:**
```
Mean divergence: 0.244
StdDev: 0.226 (threshold: > 0.1)
Distribution: 75% low, 14% medium, 11% high
```

**Reason Codes Distribution:**
```
SIG_MATCH: 36 events
STAT_ANOMALY: 132 events  
PCA_OUTLIER: 16 events
PROT_VIOLATION: 10 events
ENGINE_CONFLICT: 6 events
```

**Quality Confirmation:**
- ✅ Real variance (not synthetic correlation)
- ✅ Realistic distributions
- ✅ Full ADR-002 compliance
- ✅ Multi-engine provenance present

---

## 🐛 TECHNICAL DEBT REGISTER

### ISSUE-008: etcd-server Bootstrap Idempotency

**Status:** ✅ RESOLVED (Day 38)  
**Solution:** HTTP GET `/seed` with server-side idempotency  
**Result:** Seed persists across restarts, no regeneration

---

### ISSUE-009: RAGLogger Provenance Field Mismatch

**Status:** ✅ RESOLVED (Day 38)  
**Problem:** Reading `decision_metadata.score_divergence` (legacy) instead of `provenance.discrepancy_score` (ADR-002)  
**Solution:** Updated rag_logger.cpp line 192 with backwards compatibility  
**Result:** Real dispersion captured (StdDev: 0.226)

---

### ISSUE-007: Magic Numbers in ml-detector

**Severity:** Medium  
**Status:** Documented, pending  
**Priority:** Day 39  
**Estimated:** 30 minutes

---

### ISSUE-006: Log Files Not Persisted

**Severity:** Medium  
**Status:** Documented, pending  
**Priority:** Day 39  
**Estimated:** 1 hour (all components)

---

### ISSUE-005: RAGLogger Memory Leak

**Status:** Documented, pending  
**Impact:** Restart every 3 days  
**Priority:** Medium  
**Estimated:** 2-3 days

---

### ISSUE-003: Thread-Local FlowManager Bug

**Status:** Documented, pending  
**Impact:** Only 11/102 features captured  
**Workaround:** PCA trained with synthetic data  
**Priority:** HIGH (but not blocking Day 38)  
**Estimated:** 1-2 days

---

## 📅 Phase 2A - Foundation (Week 5: Days 35-40)

### ✅ Day 35 - Skeleton Complete (2026-01-11)

**Completado:**
- [x] Directory structure (18 directories, 12 files)
- [x] CMakeLists.txt with dependency detection
- [x] Configuration parser
- [x] All stub files created
- [x] Binary compiling and running

---

### ✅ Day 36 - Crypto Integration (2026-01-12)

**Completado:**
- [x] Integrate crypto-transport API
- [x] Update event_loader.cpp with crypto
- [x] Successful compilation
- [x] 101-feature extraction implemented

---

### ✅ Day 37 - ADR-002 Provenance (2026-01-13)

**Completado:**
- [x] Protobuf contract extended
- [x] Created reason_codes.hpp
- [x] Sniffer fills fast-path verdict
- [x] ml-detector adds RF verdict + discrepancy
- [x] rag-ingester parses provenance
- [x] RAGLogger encrypts artifacts

---

### 🔄 Day 38 - Synthetic Data + ONNX (2026-01-15) - 75% COMPLETE

**Status:** Steps 1-3 ✅ | Steps 4-5 ⏳ Tomorrow

**Completado HOY:**
- [x] Tools infrastructure (`/vagrant/tools/`)
- [x] `generate_synthetic_events.cpp` (850 lines)
- [x] Generator compiled successfully
- [x] Simplified architecture (HTTP GET `/seed`)
- [x] 100 eventos generados (.pb.enc)
- [x] Gepeto validation PASSED
- [x] Real dispersion confirmed (StdDev: 0.226)
- [x] RAGLogger fix (provenance field)
- [x] Makefile tasks (day38-step1/2/3)

**Pendiente MAÑANA (2.5-3h):**
- [ ] Update ChronosEmbedder (103 features)
- [ ] Update SBERTEmbedder (103 features)
- [ ] Update AttackEmbedder (103 features)
- [ ] End-to-end smoke test
- [ ] Invariant validation
- [ ] **Day 38 COMPLETE**

**Key Achievements:**
- 🏛️ Zero drift: Production RAGLogger reused
- 🔒 Security: etcd integration, no hardcoded keys
- ✅ Quality: Real dispersion (Gepeto validated)
- 📊 Data: 100% ADR-002 compliance

---

### 📋 Day 39 - Technical Debt Cleanup (Pending)

**Goals:**
- [ ] Fix ISSUE-007: Magic numbers → JSON config
- [ ] Fix ISSUE-006: Log files persistence
- [ ] Analysis of ISSUE-003: FlowManager bug
- [ ] Decision on fix strategy

---

### 📋 Day 40 - Integration Testing (Pending)

**Goals:**
- [ ] End-to-end integration tests
- [ ] Performance benchmarking
- [ ] etcd registration
- [ ] Memory profiling

---

## 📅 Phase 2B - Optimization (Week 6: Days 41-45)

### Day 41 - Multi-Threading
- [ ] Enable parallel mode
- [ ] ThreadPool for embeddings
- [ ] Performance: 500 events/sec target

### Day 42 - Persistence
- [ ] FAISS index save/load
- [ ] Checkpointing
- [ ] Crash recovery

### Day 43 - Advanced Strategies
- [ ] Temporal tiers
- [ ] Metadata-first search
- [ ] Quantization (int8)

### Day 44 - Integration Testing
- [ ] End-to-end pipeline test
- [ ] 10K events benchmark
- [ ] 24h load testing

### Day 45 - Documentation
- [ ] API documentation
- [ ] Deployment guide
- [ ] Troubleshooting guide

---

## 📅 Phase 3 - GAIA System (Weeks 7-8)

### RAG-Master Development
- [ ] Local level orchestrator
- [ ] LLM validator (TinyLlama)
- [ ] Vaccine distributor
- [ ] Multi-client coordination
- [ ] Provenance analysis for 0-day

### Campus & Global Levels
- [ ] Campus-level orchestrator
- [ ] Global orchestrator
- [ ] APT detection
- [ ] Global vaccine distribution

---

## 🌍 Vision: GAIA System

Hierarchical immune network:
- **Local (Building):** Immediate response
- **Campus:** Multi-building coordination
- **Global:** Organization-wide intelligence

**Enabled by ADR-002:**
- Multi-engine provenance
- Reason codes for situational intelligence
- 0-day detection (PCA_OUTLIER + ENGINE_CONFLICT)
- Transferable vaccines (embedding signatures)

---

## 📊 Success Metrics

### Phase 2A (Week 5) - Updated

**Day 38 Status:** 75% Complete

- ✅ Compilation successful (Days 35-37)
- ✅ ADR-002 implemented (Day 37)
- ✅ ADR-001 validated (Day 38)
- ✅ Generator compiled (Day 38)
- ✅ Synthetic data generated (Day 38)
- ✅ Gepeto validation passed (Day 38)
- ⏳ ONNX Embedders (Day 38 - tomorrow)
- ⏳ End-to-end pipeline (Day 38 - tomorrow)

---

## 📈 Progress Visual
```
Phase 1:  [████████████████████] 100% COMPLETE
Phase 2A: [████████████░░░░░░░░]  60% (Days 35-38/40)
Phase 2B: [░░░░░░░░░░░░░░░░░░░░]   0%
Phase 3:  [░░░░░░░░░░░░░░░░░░░░]   0%
```

**Day 38 Breakdown:**
```
Structure:    [████] 100% ✅
Compilation:  [████] 100% ✅
Generation:   [████] 100% ✅
Validation:   [████] 100% ✅ (Gepeto)
Embedders:    [░░░░]   0% ← Tomorrow
Integration:  [░░░░]   0% ← Tomorrow
```

---

## 🏛️ Via Appia Quality Checkpoints

**Foundation (Days 35-38):**
- [x] Structure before functionality
- [x] Dependencies verified before code
- [x] Tests from day 1
- [x] Clean compilation before features
- [x] Security by design (encryption mandatory)
- [x] Provenance contract complete (ADR-002)
- [x] Generator compiled with production compliance
- [x] Synthetic data validated (real dispersion)
- [ ] ONNX embedders updated (tomorrow)
- [ ] End-to-end validation (tomorrow)

**Quality Gates Passed:**
- ✅ Zero drift (production RAGLogger)
- ✅ Real dispersion (StdDev: 0.226)
- ✅ ADR-002 compliance (full provenance)
- ✅ Security (etcd integration, no hardcoded keys)
- ✅ Gepeto validation (all critical points)

---

## 📚 KEY DOCUMENTS

### Day 38 Files (Updated)
- `/vagrant/tools/generate_synthetic_events.cpp` - Simplified HTTP version
- `/vagrant/ml-detector/src/rag_logger.cpp` - Fixed provenance field
- `/vagrant/Makefile` - day38-step1/2/3 tasks added
- `/vagrant/logs/rag/synthetic/` - 100 events + artifacts

### Shared Resources
- `/vagrant/common/include/reason_codes.hpp` - 5 reason codes
- `/vagrant/protobuf/network_security.proto` - THE LAW

---

## 🎓 Lessons Learned

### Day 38 (NEW)

1. ✅ **HTTP simplicity wins** - Direct GET better than heavy client
2. ✅ **Protobuf field naming matters** - provenance vs decision_metadata
3. ✅ **Locale matters** - LC_ALL=C for awk calculations
4. ✅ **Real validation critical** - Gepeto caught dispersion importance
5. ✅ **Idempotency from start** - etcd-server handles seed persistence
6. ✅ **Via Appia quality** - Foundation solid before expansion

---

**End of Backlog**

**Last Updated:** 2026-01-15 (Day 38 - 75% Complete)  
**Next Update:** 2026-01-15 Evening (Day 38 - 100% Complete)  
**Vision:** Sistema inmunológico jerárquico global 🌍  
**Security:** Multi-engine provenance + Encryption mandatory 🔒  
**Quality:** Via Appia - 75% done, finish tomorrow 🏛️