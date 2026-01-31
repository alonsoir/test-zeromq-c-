# 🛡️ ML Defender - Autonomous Network Security System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![eBPF/XDP](https://img.shields.io/badge/eBPF-XDP-orange.svg)](https://ebpf.io/)

A self-evolving network security system with embedded ML - protecting life-critical infrastructure with sub-microsecond detection.

---

## 🌟 What Makes This Different?

This is my vision of how to design a modern IDS:

- ⚡ **Sub-microsecond detection** - 4 embedded C++20 RandomForest detectors (400 trees, 6,330 nodes)
- 🎯 **Zero external dependencies** - Pure C++20 constexpr, no ONNX for core detectors
- 🔬 **Synthetic data training** - F1 = 1.00 without academic datasets
- 🏗️ **Production-ready** - From $35 Raspberry Pi to enterprise servers
- 🧬 **Autonomous evolution** - Self-improving with transparent methodology
- 🏥 **Life-critical design** - Built for healthcare and critical infrastructure
- 🤖 **AI-Powered Configuration** - Real LLAMA integration for natural language control
- 🌐 **Gateway Mode** - Network-wide protection with dual-NIC architecture
- 📊 **RAGLogger** - 83-field comprehensive event logging for AI analysis
- 🔐 **Unified Crypto Ecosystem** - All components use crypto-transport library
- 🔄 **Bidirectional Config** - Components can update their own configuration
- 🎯 **End-to-End Encryption** - ChaCha20-Poly1305 + LZ4 across entire pipeline

---

## 🎯 Current Status
```
┌─────────────────────────────────────────────────────────────────┐
│  DAY 48 PHASE 1 COMPLETE: Contract Validation + RAGLogger Fix  │
│  (31 Enero 2026)                                               │
│  Progress: Dual Issue Closure ✅ + Thread-Safety Validated 🚀  │
├─────────────────────────────────────────────────────────────────┤
│  🎉 DAY 48 ACHIEVEMENTS:                                        │
│                                                                 │
│  ✅ ISSUE-003 CLOSED: Contract Validation                      │
│     • Dynamic protobuf reflection validator (114+ fields)      │
│     • Validates 4 critical embedded messages                   │
│     • Detects incomplete events gracefully                     │
│     • Instrumented in zmq_handler + main shutdown              │
│                                                                 │
│  ✅ ISSUE-004 CLOSED: RAGLogger Resilience                     │
│     • Fixed SEGFAULT on incomplete embedded messages           │
│     • Pre-serialization validation prevents crashes            │
│     • Graceful skip with detailed logging                      │
│     • 17 events processed, 0 crashes validated                 │
│                                                                 │
│  ✅ Thread-Safety Validated (Day 48 Phase 0):                  │
│     • TSAN baseline: 0 races, 0 deadlocks, 0 warnings          │
│     • 4 components stable under 300s stress test               │
│     • ShardedFlowManager: 800K ops/sec TSAN-clean              │
│     • Integration test: All components operational             │
│                                                                 │
│  ✅ Pipeline Core Status:                                       │
│     • Thread-safe: TSAN validated ✅                           │
│     • Contract-verified: 114+ fields dynamic check ✅          │
│     • Crash-resilient: RAGLogger defensive ✅                  │
│     • Production-ready: 14/14 tests passing ✅                 │
│                                                                 │
│  🏛️ Via Appia Quality - Day 48:                                │
│     "Dual issue closure en un día. Contract validator          │
│     descubrió ISSUE-004 automáticamente - instrumentación      │
│     temprana paga dividendos. TSAN baseline perfecto.          │
│     RAGLogger ahora resiliente a eventos incompletos.          │
│     Sistema thread-safe, contract-verified, crash-proof.       │
│     Evidencia empírica en cada paso. Metodología científica.   │
│     Despacio y bien. 🏛️"                                      │
│                                                                 │
│  🎯 NEXT PRIORITIES (Day 49-52):                               │
│     🔥 Build System Hardening (Day 49-50)                      │
│        → Eliminate hardcoded CMake flags                       │
│        → Centralize in Makefile root                           │
│        → Enable AST (static analysis)                          │
│                                                                 │
│     🔥 Firewall Breaking Point Test (Day 50-51)                │
│        → Iterative stress testing until failure                │
│        → Find exact throughput limit                           │
│        → Safety: VM isolation + dry-run mode                   │
│                                                                 │
│     📊 Security Framework Expansion (Day 51-52)                │
│        → G3 tests: Feature Completeness                        │
│        → G4 tests: Microscope Isolation                        │
│        → Evidence dashboard updates                            │
│                                                                 │
│  Phase 1 Achievement (Days 1-48):                              │
│     ✅ 4 embedded C++20 detectors (<1.06μs)                    │
│     ✅ eBPF/XDP dual-NIC packet capture                        │
│     ✅ ShardedFlowManager (800K ops/sec, TSAN-clean)           │
│     ✅ Contract validation (114+ fields)                       │
│     ✅ Thread-safety validated (TSAN perfect)                  │
│     ✅ RAGLogger crash-proof (defensive design)                │
│     ✅ End-to-end encryption (ChaCha20-Poly1305)               │
│     ✅ RAG system operational (TinyLlama + FAISS)              │
│     ✅ Production-ready pipeline ✅                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (Development Setup)

### **Prerequisites**
- VirtualBox + Vagrant
- Debian 12 (Bookworm) VMs
- Mac/Linux host machine

### **Complete Setup Sequence**
```bash
# 1. Clone repository
git clone https://github.com/alonsoir/test-zeromq-docker.git
cd test-zeromq-docker

# 2. Start VMs
vagrant up defender && vagrant up client

# 3. Build all components (from host)
make proto-unified             # Generate unified protobuf files
make crypto-transport-build    # Build crypto-transport library (FIRST!)
make etcd-client-build         # Build etcd-client (uses crypto-transport)
make etcd-server-build         # Build etcd-server (uses crypto-transport)
make sniffer                   # Build eBPF/XDP sniffer (WITH LINKAGE!)
make detector                  # Build ml-detector (CRYPTO INTEGRATED!)
make firewall                  # Build firewall agent (CRYPTO INTEGRATED!)
make rag                       # Build RAG system (CRYPTO INTEGRATED!)

# 4. Verify linkage
make verify-crypto-linkage
# Expected: All components show libcrypto_transport.so.1 ✅

# 5. Start the lab
make run-lab-dev

# 6. Verify components are running
make status-lab

# 7. Monitor in real-time
watch -n 5 'vagrant ssh defender -c "echo \"Artifacts: \$(ls /vagrant/logs/rag/artifacts/$(date +%Y-%m-%d)/ 2>/dev/null | wc -l)  JSONL: \$(wc -l < /vagrant/logs/rag/events/$(date +%Y-%m-%d).jsonl 2>/dev/null || echo 0)\""'

# 8. Stop lab when done
make kill-lab
```

---

## 🔐 crypto-transport Unified Ecosystem (Day 26-28)

### **Architecture Evolution**

**After (Day 28):**
```
crypto-transport (SINGLE source of truth)
    ↓ XSalsa20-Poly1305 + LZ4
etcd-client (uses crypto-transport)
    ↓ HTTP + key exchange
ALL Components (use crypto-transport):
├─ sniffer ✅
├─ ml-detector ✅
├─ firewall ✅
├─ etcd-server ✅
└─ RAG ✅
```

### **Performance**
```
Compression (LZ4):
  • ml-detector config: 11754 → 5084 bytes (56.7%) ✅
  • Intelligent: Small configs not compressed

Encryption (ChaCha20-Poly1305):
  • Overhead: +40 bytes fixed (nonce + MAC)
  • Operation time: <3 μs

E2E Pipeline:
  • All components: crypto-transport linked ✅
  • Zero manual key management ✅
  • Memory leak: Resolved (31 MB/h) ✅
```

---

## 🛡️ Dual-Score Architecture

### **Maximum Threat Wins Logic**
```
┌─────────────────────────────────────────────────────────────┐
│ SNIFFER (Fast Detector) + crypto-transport ✅              │
│                                                             │
│  • Linkage complete Day 28 ✅                               │
│  • Code integration Day 29 ✅                               │
│  Populates: fast_detector_score, reason, triggered         │
└─────────────────┬───────────────────────────────────────────┘
                  │ Protobuf Event (ZMQ 5571) - Encrypted ✅
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ ML DETECTOR (Dual-Score + RAGLogger) ✅                     │
│                                                             │
│  1. Decrypt incoming packet ✅                              │
│  2. Read fast_detector_score                                │
│  3. Calculate ml_detector_score (4 models)                  │
│  4. final_score = max(fast_score, ml_score)                │
│  5. RAGLogger: 83-field events ✅                           │
│  6. Memory: 31 MB/h (production-ready) ✅                   │
│  7. Encrypt + send to firewall ✅                           │
└─────────────────┬───────────────────────────────────────────┘
                  │ Enriched Event (ZMQ 5572) + encrypted
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ FIREWALL / RAG QUEUE ✅                                     │
│                                                             │
│  • Decrypt incoming event ✅                                │
│  • Block/Monitor based on final_score                       │
│  • RAG analysis for divergent events ✅                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📖 Documentation

### Core Documentation
- [Architecture Deep Dive](docs/ARCHITECTURE.md)
- [Dual-Score Architecture](docs/DAY_13_DUAL_SCORE_ANALYSIS.md)
- [RAGLogger Schema](docs/RAGLOGGER_SCHEMA.md)
- [Race Condition Fix](docs/DAY_16_RACE_CONDITION_FIX.md)
- [Synthetic Data Methodology](docs/SYNTHETIC_DATA.md)
- [Performance Tuning](docs/PERFORMANCE.md)

### Day 48: Contract Validation + RAGLogger Fix 🆕
- [Day 48 Phase 0: TSAN Baseline](tsan-reports/day48/TSAN_SUMMARY.md) ✨
  - Thread-safety validation (0 races, 0 deadlocks)
  - 4 components under 300s stress test
  - Integration test methodology
- [Day 48 Phase 1: Dual Issue Closure](BACKLOG.md#day-48-phase-1) ✨
  - ISSUE-003: Contract validator implementation
  - ISSUE-004: RAGLogger resilience fix
  - Evidence: 17 events, 0 crashes

### Crypto-Transport Ecosystem (Days 26-30)
- [crypto-transport Library](crypto-transport/README.md)
- [Day 26: Library Creation](docs/DAY_26_CRYPTO_TRANSPORT.md)
- [Day 27: etcd-server + ml-detector](docs/DAY_27_CRYPTO_UNIFICATION.md)
- [Day 28: Sniffer Integration](docs/DAY_28_SNIFFER_LINKAGE.md)
- [Day 29: E2E Troubleshooting](docs/DAY_29_E2E_TROUBLESHOOTING.md)
- [Day 30: Memory Leak Resolution](docs/DAY_30_MEMORY_LEAK_INVESTIGATION.md)

### Future Enhancements
- [FAISS Ingestion Design](docs/FAISS_INGESTION_DESIGN.md)
- [Shadow Authority](docs/SHADOW_AUTHORITY.md)
- [Decision Outcome](docs/DECISION_OUTCOME.md)

---

## 🛠️ Build Targets
```bash
# Core Components
make proto-unified           # Generate unified protobuf files
make crypto-transport-build  # Build crypto-transport library
make etcd-client-build       # Build etcd-client
make etcd-server-build       # Build etcd-server
make sniffer                 # Build eBPF/XDP sniffer
make detector                # Build ml-detector
make firewall                # Build firewall agent
make rag                     # Build RAG system

# Verification
make verify-crypto-linkage   # Verify all components linked ✅

# Lab Control
make run-lab-dev             # Start full lab
make kill-lab                # Stop all components
make status-lab              # Check component status

# Testing
make test-crypto-transport   # Test crypto-transport (16 tests)
make test-etcd-client        # Test etcd-client (3 tests)
make test-hardening          # Run all 14 hardening tests ✅
```

---

## 🏛️ Via Appia Quality Philosophy

Like the ancient Roman road that still stands 2,300 years later:

1. **Clean Code** - Simple, readable, maintainable
2. **KISS** - Keep It Simple
3. **Funciona > Perfecto** - Working beats perfect
4. **Smooth & Fast** - Optimize what matters
5. **Scientific Honesty** - Truth above convenience
6. **Methodical Progress** - Despacio y bien (slow and steady)

**Day 48 Truth:**
> "Dual issue closure en un día. Contract validator descubrió ISSUE-004
> automáticamente - instrumentación temprana paga dividendos. TSAN baseline
> perfecto validó thread-safety en 4 componentes. RAGLogger ahora resiliente
> a eventos incompletos con validación defensiva. Sistema thread-safe,
> contract-verified, crash-proof. 14/14 tests pasando. Evidencia empírica
> en cada paso. Metodología científica. Despacio y bien. 🏛️"

---

## 🤝 Multi-Agent Collaboration

This project represents multi-agent AI collaboration:

| AI Agent | Contribution |
|----------|-------------|
| **Claude (Anthropic)** | Architecture, contract validation, RAGLogger resilience |
| **DeepSeek (v3)** | RAG system, security framework analysis, roadmap planning |
| **Gemini (Google)** | Pipeline strategist, build system architect |
| **Grok (xAI)** | External observer, quality validation, methodology review |
| **Qwen (Alibaba)** | Ethical guardian, scalability insights, academic methodology |
| **ChatGPT (OpenAI)** | Senior architect, Watcher design, AST recommendations |
| **Alonso** | Vision, C++ implementation, scientific methodology 🔍 |

All AI agents will be credited as **co-authors** in academic publications.

---

## 📧 Contact

- GitHub: [@alonsoir](https://github.com/alonsoir)
- Project: [ML Defender](https://github.com/alonsoir/test-zeromq-docker)

---

**Built with 🛡️ for a safer internet**

*Via Appia Quality - Designed to last decades*

---

**Day 48 Phase 1 Complete:**  
Contract validation operational ✅  
RAGLogger crash-proof (defensive design) ✅  
Thread-safety validated (TSAN perfect) ✅  
Dual issue closure with empirical evidence ✅  
Metodología científica, despacio y bien 🏛️

**Next:** Day 49 - Build System Hardening + Firewall Stress Testing

---

**Latest Update:** 31 Enero 2026 - Day 48 Phase 1 Complete 🎉  
**Progress:** Base fundacional validada | Thread-safe + Contract-verified + Crash-resilient  
**Next:** Day 49-52 Infrastructure consolidation + Firewall breaking point analysis
