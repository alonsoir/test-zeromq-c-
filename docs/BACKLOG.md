# RAG System - Development Backlog

**Last Updated:** 2026-01-22 Morning - Day 40 Producer COMPLETE ✅  
**Current Phase:** 2B - Producer-Consumer RAG (50%)  
**Next Session:** Day 41 - Consumer Implementation

---

## ✅ Day 40 - PRODUCER COMPLETE (22 Enero 2026)

### **Architecture Decision: Producer-Consumer Pattern**

**Alonso's Vision (Correct):**
```
"No entiendo por qué en el RAG tenemos que volver a cargar de manera 
asíncrona los eventos entrantes y volver a indexarlos en el FAISS. 
Es como tener que hacer el trabajo dos veces, no?"
```

**Solution: Classic Big Data Pattern**
```
rag-ingester (Producer):
  └─ Write-once, index-once
  └─ Save FAISS + metadata to disk
  └─ Runs 24/7 in background

RAG (Consumer):  
  └─ Read-only
  └─ Load pre-built indices
  └─ Query without rebuilding
  └─ Zero duplication
```

### **Producer Implementation (COMPLETE):**

**Files Created:**
```
/vagrant/rag-ingester/
├── include/metadata_db.hpp          ✅ NEW
├── src/metadata_db.cpp              ✅ NEW
├── src/main.cpp                     ✅ UPDATED
├── include/indexers/multi_index_manager.hpp ✅ UPDATED
└── CMakeLists.txt                   ✅ UPDATED
```

**Functionality:**
- ✅ MetadataDB writes to SQLite (WAL mode)
- ✅ Schema: events table (faiss_idx, event_id, classification, ...)
- ✅ save_indices_to_disk() every 100 events + shutdown
- ✅ FAISS write_index() for 3 indices
- ✅ Multi_index_manager getters (public access)
- ✅ Compilation successful (100%)

**Output Directory:**
```
/vagrant/shared/indices/
├── chronos.faiss      (Producer writes)
├── sbert.faiss        (Producer writes)
├── attack.faiss       (Producer writes)
└── metadata.db        (Producer writes)
```

---

## 🎯 Day 41 - CONSUMER IMPLEMENTATION (NEXT)

### **Goal:** Complete read-only Consumer in RAG

**Morning (3h):**
- [ ] Create metadata_reader.hpp/cpp (read-only SQLite)
- [ ] Update RAG main.cpp (faiss::read_index)
- [ ] Implement query_similar command
- [ ] Test with real synthetic events

**Tarde (2h):**
- [ ] Add --explain flag (Qwen's feature deltas)
- [ ] Update Vagrantfile (shared/indices directory)
- [ ] Documentation (PRODUCER_CONSUMER_ARCHITECTURE.md)
- [ ] End-to-end testing

**Success Criteria:**
```bash
SECURITY_SYSTEM> query_similar synthetic_000059

🔍 Query: synthetic_000059 (DDoS, discrepancy: 0.82)

📊 Top 5 Similar:
 1. synthetic_000047 (dist: 0.234) - DDoS
 2. synthetic_000082 (dist: 0.312) - DDoS
 3. synthetic_000015 (dist: 0.356) - PortScan
 ...

✅ Same-class clustering: ≥60%
✅ Distances <0.5 for similar
```

---

## 🔧 Vagrantfile Update REQUIRED

```ruby
# Add to provisioning:

config.vm.provision "shell", inline: <<-SHELL
  # Create shared indices directory
  mkdir -p /vagrant/shared/indices
  chown -R vagrant:vagrant /vagrant/shared/indices
  
  # SQLite3 dev headers (if not present)
  apt-get install -y libsqlite3-dev
SHELL
```

---

## 📊 Phase 2B Progress

```
Producer (rag-ingester):  ████████████████████ 100% ✅
Consumer (RAG):          ░░░░░░░░░░░░░░░░░░░░   0% ← Day 41

Overall Phase 2B:        ██████████░░░░░░░░░░  50%
```

**Producer Writes:**
- ✅ FAISS indices (chronos/sbert/attack)
- ✅ Metadata SQLite (events table)
- ✅ Saves every 100 events + shutdown

**Consumer Reads (Pending):**
- ❌ Load FAISS indices (faiss::read_index)
- ❌ Load metadata (SQLite read-only)
- ❌ query_similar implementation
- ❌ --explain flag

---

## 🏛️ Via Appia Quality - Day 40

**Architecture Decisions:**
- ✅ **Producer-Consumer:** Single responsibility principle
- ✅ **No duplication:** Index once, query many
- ✅ **Persistence:** Disk-based indices
- ✅ **Scalability:** Multiple consumers can read
- ✅ **Security:** Config-driven paths, no hardcoding

**Technical Lessons:**
1. ✅ Include order: `<faiss/IndexFlat.h>` for ntotal
2. ✅ Public getters: MultiIndexManager access
3. ✅ WAL mode: SQLite concurrent read/write
4. ✅ Save intervals: Balance I/O vs data loss

---

## 🎓 Key Insights - Day 40

**Alonso's Architecture > Initial Proposal:**

**WRONG (Initial):**
```
rag-ingester: Index in FAISS
RAG:          RE-index same events (duplication!)
```

**RIGHT (Alonso's):**
```
rag-ingester: Index once → Write to disk
RAG:          Read from disk → Query
```

**Why It Matters:**
- Zero duplication (efficiency)
- Producer runs 24/7 (always indexing)
- Consumer can restart anytime (stateless)
- Multiple RAG instances can read same indices

**Big Data Pattern Recognition:**
- Alonso's experience shows ✨
- Classic distributed architecture
- Kafka-style producer/consumer
- Scales naturally

---

## 🐛 Technical Debt

### ISSUE-012: Vagrantfile Missing Provisions (NEW)

**Severity:** Medium  
**Status:** Documented  
**Priority:** Day 41  
**Estimated:** 10 minutes

**Required:**
- Create `/vagrant/shared/indices/` on provision
- Install `libsqlite3-dev` if missing

---

### ISSUE-003: FlowManager Thread-Local Bug

**Status:** Documented, deferred  
**Impact:** Only 11/105 features captured  
**Priority:** HIGH (but workaround exists)  
**Estimated:** 1-2 days  
**Deferral Reason:** RAG pipeline functional with synthetic data

---

## 📅 Roadmap

### Day 41 - Consumer + First Query ⬅️ NEXT
- [ ] metadata_reader.hpp/cpp
- [ ] RAG main.cpp (load indices)
- [ ] query_similar command
- [ ] --explain flag
- [ ] End-to-end test

### Day 42 - ONNX Documentation
- [ ] ONNX_ARCHITECTURE.md
- [ ] Training pipeline spec
- [ ] Decision framework
- [ ] Upgrade triggers

### Day 43 - FlowManager Bug (ISSUE-003)
- [ ] Analyze thread-local issue
- [ ] Design global FlowManager
- [ ] Implement LRU cache
- [ ] Test 105/105 features

### Day 44 - Testing & Hardening
- [ ] 10K events benchmark
- [ ] Memory profiling
- [ ] 24h stability test

### Day 45 - Documentation & Merge
- [ ] README.md update
- [ ] DEPLOYMENT.md
- [ ] USER_GUIDE.md
- [ ] Merge to main (silent)

---

## 🌟 Founding Principles Applied

**"Trabajamos bajo evidencia, no bajo supuestos"**

**Evidence Day 40:**
- ✅ Producer compiles (100%)
- ✅ SQLite schema works
- ✅ FAISS write_index functional
- ⏳ Consumer pending (Day 41)

**Evidence Needed:**
- ⏳ End-to-end query test
- ⏳ Same-class clustering ≥60%
- ⏳ Performance with 1000+ events

---

**End of Backlog Update**

**Status:** Day 40 Producer COMPLETE ✅  
**Next:** Day 41 Consumer Implementation  
**Architecture:** Producer-Consumer (Alonso's vision) 🏗️  
**Quality:** Via Appia maintained 🏛️