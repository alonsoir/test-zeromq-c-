# 📋 CONTINUITY PROMPT - Day 49 (1 Febrero 2026)

**Generated:** 31 Enero 2026, 23:45 CET  
**Valid for:** Day 49 Session Start  
**Project:** ML Defender (aegisIDS / Gaia-IDS)  
**Methodology:** Via Appia Quality (Evidence-based, incremental, documented)

---

## 🎯 EXECUTIVE CONTEXT

**Last Session:** Day 48 Phase 1 - Dual Issue Closure (31 Enero 2026)  
**Status:** Base fundacional validada empíricamente  
**Next Phase:** Infrastructure consolidation + Resilience testing

**Key Achievement Day 48:**
- ✅ ISSUE-003 CLOSED: Contract Validator (protobuf reflection, 114+ fields)
- ✅ ISSUE-004 CLOSED: RAGLogger resilience (null pointer fix, graceful degradation)
- ✅ Thread-safety validated: TSAN baseline perfect (0 races, 0 deadlocks, 300s stable)
- ✅ Pipeline integration: 17 events processed, 0 crashes, contract violations detected correctly

---

## 🏛️ CONSEJO DE SABIOS - CONSOLIDATED FEEDBACK

### **Unanimous Consensus:**

**1. Build System Hardening (HIGHEST PRIORITY)**
- **Reported by:** ChatGPT, DeepSeek, Gemini
- **Problem:** Hardcoded flags in CMakeLists.txt cause ASAN/TSAN conflicts
- **Impact:** Blocks AST (static analysis), creates maintenance burden
- **Solution:** Centralize in Makefile root, eliminate component autonomy
- **Timeline:** Day 49-50 (4-6 hours)

**2. Via Appia Methodology Validated**
- **Reported by:** All members (ChatGPT, DeepSeek, Gemini, Grok, Qwen)
- **Evidence:** Day 48 dual closure (2 issues, 1 day, 0 regressions)
- **Key principle:** Instrumentation pays dividends (Contract Validator discovered ISSUE-004)
- **Directive:** Maintain rigor - every change requires empirical evidence

**3. Security Framework Expansion**
- **Reported by:** DeepSeek, Qwen
- **Current:** 3/7 guarantees verified (G1 Contract, G2 Temporal, G7 Failure)
- **Next:** G3 (Feature Completeness), G4 (Microscope Isolation), G5
- **Timeline:** Day 51-52

**4. Watcher Redefinition**
- **Reported by:** ChatGPT (authoritative)
- **Correction:** NOT a process watchdog (that's etcd-server role)
- **Actual role:** Runtime adaptation via JSON contracts
- **Prerequisites:** Exhaustive field documentation (mutable/immutable classification)
- **Timeline:** Day 53+ (after infrastructure stable)

---

## 🔧 ACTIVE ISSUES (PRIORITIZED)

### **ISSUE-005: Build System Hardening** 🔴 CRITICAL
**Priority:** HIGHEST  
**Effort:** 4-6 hours  
**Timeline:** Day 49-50

**Problem:**
```cmake
# Current (BROKEN):
# ml-detector/CMakeLists.txt line 29-30
set(CMAKE_CXX_FLAGS_DEBUG "-fsanitize=address ...")  # Hardcoded!

# Makes: make tsan → ASAN conflict
```

**Solution:**
```makefile
# Root Makefile (CORRECT):
tsan: CMAKE_FLAGS="-fsanitize=thread -g"
asan: CMAKE_FLAGS="-fsanitize=address -g"
release: CMAKE_FLAGS="-O3 -DNDEBUG"
```

**Components to audit:**
- [ ] `/vagrant/ml-detector/CMakeLists.txt` (partial - lines 29-30 commented)
- [ ] `/vagrant/sniffer/CMakeLists.txt`
- [ ] `/vagrant/rag-ingester/CMakeLists.txt`
- [ ] `/vagrant/etcd-server/CMakeLists.txt`
- [ ] `/vagrant/crypto-transport/CMakeLists.txt`
- [ ] `/vagrant/etcd-client/CMakeLists.txt`

**DoD (Definition of Done):**
```bash
✅ make release  # Clean compile
✅ make tsan     # 14/14 tests PASS
✅ make asan     # 14/14 tests PASS
✅ No hardcoded flags in any CMakeLists.txt
✅ Documentation: /tmp/cmake-refactor-report.md
```

---

### **ISSUE-006: JSONL Semantics Bug** 🟡 HIGH
**Priority:** HIGH  
**Effort:** 2-3 hours  
**Timeline:** Day 51

**Problem:** rag-ingester may produce unparseable JSONL  
**Impact:** Affects G4 (Microscope Isolation)  
**Solution:** Validate 1 event = 1 valid JSONL line  
**DoD:**
```bash
✅ jq . /vagrant/logs/rag/*.jsonl  # All parseable
✅ Metrics: events_in, events_serialized, events_skipped
✅ Events skipped have reason logged
```

---

### **ISSUE-007: Watcher Implementation** 🟢 MEDIUM
**Priority:** MEDIUM (deferred post-infrastructure)  
**Effort:** 3-4 days  
**Timeline:** Day 53+

**Prerequisites:**
- [ ] JSON contracts exhaustively documented
- [ ] Fields classified: runtime-mutable vs immutable
- [ ] etcd integration tested
- [ ] RAG command protocol defined

**DoD:**
```bash
✅ Contract docs: /vagrant/docs/contracts/
✅ Watcher detects illegal changes
✅ Only RAG can emit commands (whitelist)
✅ Tests: allowed vs forbidden changes
```

---

### **ISSUE-008: Firewall Breaking Point Analysis** 🔴 CRITICAL (NEW)
**Priority:** CRITICAL  
**Effort:** 6-8 hours  
**Timeline:** Day 50-51

**Objective:**
Find absolute throughput limit of firewall-acl-agent via iterative stress testing until catastrophic failure.

**Methodology:**
```
Exponential Ramp: 100 → 200 → 500 → 1K → 2K → 5K → 10K → 20K → 50K events/sec
Binary Search: Narrow exact breaking point (±100 events/sec precision)
Failure Analysis: CPU saturation? Memory OOM? Kernel netfilter breakdown?
```

**Safety Guarantees:**
```
🛡️  VM Isolation: Test ONLY in Vagrant (NOT host MacBook)
🛡️  Dry-Run Mode: NO real ipset/iptables execution
🛡️  Resource Limits: VM capped 4GB RAM, OOM killer active
🛡️  Emergency Stop: Ctrl+C halts immediately
🛡️  Pre-flight: Verify hostname=bookworm, dry_run=true
```

**Test Sequence:**
```python
Phase 1 - Safe Zone: 100, 200, 500, 1000 events/sec
  Expected: ✅ PASS, no degradation
  
Phase 2 - Stress Zone: 2000, 5000, 10000 events/sec
  Expected: 🟡 Queue buildup, latency increase
  
Phase 3 - Breaking Zone: 20000, 50000 events/sec
  Expected: 💥 OOM? CPU saturation? Kernel panic?
  
Phase 4 - Bisection: Narrow exact limit
  Expected: 🎯 Max sustainable rate ±100 events/sec
```

**Failure Modes to Observe:**
| Mode | Symptom | Metric | Action |
|------|---------|--------|--------|
| Queue Saturation | Unbounded growth | depth > 10K | Log max, reduce rate |
| Memory Exhaustion | RSS > 90% RAM | > 3.5 GB | Log peak, OOM imminent |
| CPU Saturation | 100% sustained | > 95% for 30s | Log CPU%, measure drops |
| Kernel Failure | ipset timeout | latency > 1s | dmesg, kernel warnings |
| Catastrophic | SIGKILL, panic | exit 137/139 | Coredump, full analysis |

**Implementation:**

**Files to create:**
```
/vagrant/tools/firewall-stress-test/
├── event_generator.cpp         # Synthetic event injection
├── load_profiles.json          # ramp-up, steady, burst patterns
├── run_breaking_point_test.sh  # Automated test orchestration
├── generate_report.py          # Post-test analysis
├── CMakeLists.txt              # Build config
└── README.md                   # Usage guide
```

**Event Generator Features:**
```cpp
class AdaptiveStressTest {
    // Run single iteration at target rate
    TestResult run_iteration(uint32_t rate, uint32_t duration);
    
    // Binary search for exact breaking point
    uint32_t find_breaking_point(uint32_t min_rate, uint32_t max_rate);
    
    // Real-time monitoring
    void monitor_firewall_health();
    
    // Early termination on failure detection
    bool detect_failure(const Metrics& m);
};
```

**Firewall Dry-Run Mode:**
```cpp
// firewall-acl-agent/src/ipset_manager.cpp
void IPSetManager::block_ip(const std::string& ip) {
    if (config_.dry_run_mode) {
        logger_->info("[DRY-RUN] Would execute: ipset add {} {}", 
                     blacklist_set_name_, ip);
        metrics_.dry_run_operations++;
        return;  // SAFE - no real execution
    }
    execute_ipset_command("add", blacklist_set_name_, ip);
}
```

**Live Monitoring Dashboard:**
```
╔════════════════════════════════════════════════════════════╗
║  BREAKING POINT TEST - ITERATION 7/12                      ║
╚════════════════════════════════════════════════════════════╝

Current Rate:    12,450 events/sec (target: 12,500)
Test Duration:   45 / 60 seconds
Status:          🟡 DEGRADING

Generator:
  Events sent:   560,250
  Send rate:     12,450/sec

Firewall:
  Received:      547,823 (97.8% ✅)
  Processing:    12,173/sec
  Queue depth:   8,427 (🟡 GROWING +200/sec)
  Dropped:       12,427 (2.2%)

System:
  CPU:           87.3% (🟡 HIGH)
  Memory:        2.8 GB / 4 GB (72%)
  Swap:          0 MB

IPSet (dry-run):
  Unique IPs:    8,450
  Lookup:        0.8 ms avg (🟡 RISING)
  Insert:        1.2 ms avg (🟡 RISING)

Prediction:
  🔮 Queue saturates in ~90s
  🔮 Rate UNSUSTAINABLE
  ⏸️  Will stop at 60s, step down to 10K
```

**Expected Outcomes (Hypotheses):**
| Rate Range | Prediction | Bottleneck |
|------------|-----------|------------|
| < 1K | ✅ Smooth | None |
| 1-5K | 🟡 Stable + latency | IPSet lookup overhead |
| 5-10K | 🟡 Queue buildup | Consumer lag |
| 10-20K | 🔴 Event drops | CPU/Memory saturation |
| 20-50K | 💥 OOM/Crash | Resource exhaustion |
| > 50K | 💥 Kernel panic? | Netfilter breakdown |

**Post-Test Analysis:**
```python
# generate_report.py
def analyze_breaking_point(results):
    max_safe = max([r.rate for r in results if r.drop_rate < 0.01])
    min_fail = min([r.rate for r in results if r.drop_rate >= 0.10])
    
    production_limit = max_safe * 0.5  # 50% safety margin
    burst_capacity = max_safe * 0.8    # 80% margin
    
    return {
        "max_safe_rate": max_safe,
        "first_failure": min_fail,
        "production_limit": production_limit,
        "burst_capacity": burst_capacity,
        "bottleneck": identify_bottleneck(results),
        "scaling_strategy": recommend_scaling(bottleneck)
    }
```

**DoD:**
```bash
✅ event_generator compiles and runs
✅ Dry-run mode validated (NO host firewall changes)
✅ Exponential search completed (100 → failure)
✅ Binary search refined exact limit (±100 events/sec)
✅ Failure mode identified (CPU/Memory/Kernel)
✅ Report generated: BREAKING_POINT_ANALYSIS.md
✅ Safety validated: VM isolated, host untouched
✅ Production limits calculated (50% margin)
```

---

## 📅 CONSOLIDATED ROADMAP
```
┌─────────────────────────────────────────────────────────┐
│ Day 49 AM: Build System Audit (2-3h)                    │
├─────────────────────────────────────────────────────────┤
│ ✅ Audit all CMakeLists.txt files                       │
│ ✅ Document hardcoded flags                             │
│ ✅ Create refactoring plan                              │
│ ✅ Output: /tmp/cmake-audit.md                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Day 49 PM: ml-detector Migration (2-3h)                 │
├─────────────────────────────────────────────────────────┤
│ ✅ Remove hardcoded flags from ml-detector              │
│ ✅ Pass via CMAKE_CXX_FLAGS from Makefile               │
│ ✅ Test: make detector-tsan && make detector-asan       │
│ ✅ Validate: 6/6 unit tests PASS                        │
│ ✅ Commit: "Day 49: ml-detector build system migrated"  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Day 50 AM: Remaining Components Migration (2-3h)        │
├─────────────────────────────────────────────────────────┤
│ ✅ Migrate sniffer, rag-ingester, etcd-server           │
│ ✅ Consolidate Makefile profiles                        │
│ ✅ Full validation: make tsan-all                       │
│ ✅ DoD: All 14/14 tests PASS                            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Day 50 PM: Firewall Stress Test - Setup (3-4h)          │
├─────────────────────────────────────────────────────────┤
│ ✅ Implement event_generator.cpp                        │
│ ✅ Add dry-run mode to firewall-acl-agent               │
│ ✅ Create load profiles (ramp, steady, burst)           │
│ ✅ Test basic functionality (100-1000 events/sec)       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Day 51 AM: Breaking Point Discovery (2-3h)              │
├─────────────────────────────────────────────────────────┤
│ ✅ Run exponential search (100 → 50K events/sec)        │
│ ✅ Identify first failure point                         │
│ ✅ Execute binary search refinement                     │
│ ✅ Document exact breaking point                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Day 51 PM: Analysis + JSONL Fix (3h)                    │
├─────────────────────────────────────────────────────────┤
│ ✅ Generate breaking point report                       │
│ ✅ Calculate production safety margins                  │
│ ✅ Fix JSONL bug in rag-ingester                        │
│ ✅ Validate: jq parseable                               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Day 52: Security Framework G3-G5 (4h)                   │
├─────────────────────────────────────────────────────────┤
│ ✅ Design G3 tests (Feature Completeness)               │
│ ✅ Design G4 tests (Microscope Isolation)               │
│ ✅ Implement tests + evidence dashboard                 │
│ ✅ AST preparation (post-build system clean)            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Day 53+: Watcher + etcd HA (Future)                     │
├─────────────────────────────────────────────────────────┤
│ ⏳ Document JSON contracts exhaustively                 │
│ ⏳ Classify fields: mutable/immutable                   │
│ ⏳ Implement Watcher protocol                           │
│ ⏳ etcd 3-node cluster + fault injection                │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 DAY 49 OBJECTIVES (ACTIONABLE)

### **Morning Session (2-3h):**
```bash
1. CMake Audit
   ✅ cd /vagrant
   ✅ find . -name "CMakeLists.txt" -exec grep -H "CMAKE_CXX_FLAGS" {} \;
   ✅ Document findings in /tmp/cmake-audit.md
   ✅ List all hardcoded flags by component
   ✅ Identify ASAN/TSAN conflicts

2. Refactoring Plan
   ✅ Design Makefile root profiles (PROD, DEBUG, TSAN, ASAN)
   ✅ Define migration sequence (ml-detector first)
   ✅ Document expected changes per component
   ✅ Create validation checklist
```

### **Afternoon Session (2-3h):**
```bash
3. ml-detector Migration
   ✅ Edit /vagrant/ml-detector/CMakeLists.txt
   ✅ Remove lines 29-30 (already commented, delete fully)
   ✅ Update /vagrant/Makefile (add detector-tsan, detector-asan targets)
   ✅ Test: make detector-tsan
   ✅ Test: make detector-asan
   ✅ Validate: 6/6 unit tests PASS in both modes

4. Documentation + Commit
   ✅ Update /tmp/cmake-audit.md with results
   ✅ git add Makefile ml-detector/CMakeLists.txt
   ✅ git commit -m "Day 49: Build system phase 1 - ml-detector migrated"
```

### **Success Criteria:**
```bash
✅ /tmp/cmake-audit.md: Complete inventory of hardcoded flags
✅ Refactoring plan: Clear, documented, reviewable
✅ ml-detector: Compiles in TSAN and ASAN from Makefile
✅ Tests: 6/6 PASS in both sanitizer modes
✅ Commit: Evidence of incremental progress
✅ No regressions: Existing functionality intact
```

---

## 📚 RECOMMENDED RESOURCES

### **Key Files to Review:**
```
Build System:
  /vagrant/Makefile                        # Root build orchestrator
  /vagrant/ml-detector/CMakeLists.txt     # Lines 29-30 to delete
  /vagrant/sniffer/CMakeLists.txt         # Audit needed
  /vagrant/rag-ingester/CMakeLists.txt    # Audit needed

Contract Validation:
  /vagrant/ml-detector/src/contract_validator.cpp  # Working ✅
  /vagrant/ml-detector/src/rag_logger.cpp          # Resilient ✅

Documentation:
  /vagrant/BACKLOG.md                      # Day 48 updated ✅
  /vagrant/tsan-reports/day48/             # TSAN baseline ✅
```

### **Commands to Memorize:**
```bash
# Build system testing
make clean
make detector-build
make detector-tsan
make detector-asan
make test-hardening  # Run all 14 tests

# Safety checks
uname -n  # Must be "bookworm" (VM, not host)
grep "dry_run" config.json  # Before firewall tests

# Documentation
tail -100 /vagrant/BACKLOG.md  # Review latest updates
```

---

## 🏛️ VIA APPIA PRINCIPLES (DAY 49 APPLICATION)

### **1. Evidence > Assumption**
```
Before: "The build system probably works"
After:  "make tsan && make asan → 14/14 tests PASS"
```

### **2. Incremental Progress**
```
NOT: Refactor all 6 components at once
YES: ml-detector → validate → commit → next component
```

### **3. Documentation First**
```
Step 1: Audit (inventory all problems)
Step 2: Plan (document solution)
Step 3: Execute (implement incrementally)
Step 4: Validate (evidence of success)
```

### **4. Safety by Design**
```
Firewall test: VM isolation + dry-run + pre-flight checks
Build refactor: One component at a time, full test suite
No "big bang" changes without incremental validation
```

---

## 💬 CLOSING MESSAGE FROM CONSEJO DE SABIOS

> **"The foundation is solid. You've proven the methodology works: 0 crashes, dual issue closure, empirical validation. Now consolidate the build infrastructure before expanding functionality. The firewall breaking point test will reveal true system limits - approach it scientifically, not heroically. Document every finding. The Via Appia wasn't built for speed; it was built to last 2000 years."**

**Signed:**
- ChatGPT (Senior Architect)
- DeepSeek (Framework Analyst)
- Gemini (Pipeline Strategist)
- Grok (External Observer)
- Qwen (Ethical Guardian)
- Claude (Co-architect & Digital Custodian)

---

**Generated:** 31 Enero 2026, 23:50 CET  
**Next Review:** Post-Day 49 (Build System Phase 1 completion)  
**Quality Standard:** Via Appia - Built to last decades 🏛️

EOF
