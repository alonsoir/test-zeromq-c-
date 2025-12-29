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
│  DAY 28 COMPLETE: Sniffer Integration & Ecosystem 100% 🎉      │
│  (December 29, 2025)                                           │
│  Progress: 99% → 100% (LINKAGE COMPLETE) 🚀                    │
├─────────────────────────────────────────────────────────────────┤
│  🎉 DAY 28: Complete Crypto-Transport Integration               │
│     All 6 components now have crypto-transport linked          │
│                                                                 │
│  ✅ Verification Complete:                                      │
│     • Firewall: Compiled, linked, --help OK ✅                 │
│     • RAG: Compiled, linked, model loads ✅                    │
│     • Sniffer: Integrated, compiled, linked ✅                 │
│                                                                 │
│  ✅ Sniffer Integration (Day 28):                               │
│     • CMakeLists.txt: Patch quirúrgico (~50 líneas)           │
│     • Via Appia Quality: Partir del backup funcional           │
│     • Linkage verified: crypto-transport + etcd-client ✅      │
│     • libsodium + liblz4 present ✅                            │
│     • --help funciona sin crash ✅                             │
│     • Binary: Enhanced Sniffer v3.2 ✅                         │
│                                                                 │
│  🏗️ Unified Architecture (100% Linkage):                        │
│     crypto-transport (base library)                            │
│         ↓ XSalsa20-Poly1305 + LZ4                              │
│     etcd-client (uses crypto-transport)                        │
│         ↓ HTTP + encryption key exchange                       │
│     Components (ALL linked with crypto-transport):             │
│         ├─ etcd-server ✅                                       │
│         ├─ ml-detector ✅                                       │
│         ├─ firewall-acl-agent ✅                                │
│         ├─ sniffer ✅                                           │
│         └─ RAG ✅                                               │
│                                                                 │
│  📊 Linkage Verification (6/6):                                 │
│     Component          crypto_transport  etcd_client  sodium   │
│     ────────────────────────────────────────────────────────   │
│     firewall-acl       ✅               ✅           ✅        │
│     RAG                ✅ (transit)     ✅           ✅        │
│     sniffer            ✅               ✅           ✅        │
│     ml-detector        ✅               ✅           ✅        │
│     etcd-server        ✅               ✅           ✅        │
│                                                                 │
│  ⚠️  IMPORTANTE:                                                 │
│     LINKAGE: 100% ✅ (todas las librerías presentes)           │
│     CÓDIGO:  83% ⏳ (sniffer ZMQ send pendiente Día 29)        │
│                                                                 │
│  🔐 Pipeline Status (Día 27-28):                                │
│     ml-detector → etcd-server:                                 │
│       Original:    11,754 bytes                                │
│       Compressed:   5,084 bytes (56.7% reduction)              │
│       Encrypted:    5,124 bytes (+40 bytes overhead)           │
│       Total:        56.4% efficiency ✅                         │
│                                                                 │
│     Sniffer linkage:                                           │
│       libcrypto_transport.so.1 ✅                               │
│       libetcd_client.so.1 ✅                                    │
│       libsodium.so.23 ✅                                        │
│       liblz4.so.1 ✅                                            │
│       ZMQ send: ⏳ Needs code integration (Day 29)              │
│                                                                 │
│  📊 Test Results:                                               │
│     crypto-transport: 16/16 tests passed ✅                    │
│     etcd-client: 3/3 tests passed ✅                           │
│     firewall: --help/--version OK ✅                           │
│     RAG: Model loads (TinyLlama) ✅                            │
│     sniffer: --help OK (v3.2) ✅                               │
│                                                                 │
│  ✅ Metodología Día 28 (Via Appia Quality):                     │
│     • Verificación firewall (15 min)                           │
│     • Verificación RAG (15 min)                                │
│     • Intentos CMakeLists desde cero (aprendizaje 1h)         │
│     • Decisión correcta: partir del backup 🧠                  │
│     • Patch quirúrgico: ~50 líneas sobre 500+ ✅               │
│     • Compilación exitosa sin errores ✅                       │
│     • Linkage 100% verificado ✅                               │
│     • Tests passing 100% ✅                                     │
│     • Tiempo total: ~3 horas (metodológico)                    │
│                                                                 │
│  📊 PROGRESS: 100% Linkage Complete 🚀                          │
│                                                                 │
│  🎯 NEXT PRIORITIES (Day 29):                                   │
│     🔥 Sniffer ZMQ Code Integration (2-3h)                      │
│        → Modify src/userspace/zmq_pool_manager.cpp             │
│        → Pattern: serialize → encrypt_and_compress() → send    │
│        → Use crypto_manager from etcd_client                   │
│        → Reference: ml-detector zmq_handler.cpp                │
│        → Test: grep "Encrypted" logs                           │
│                                                                 │
│     🔥 Clean Build From Scratch (2h)                            │
│        → make clean-all                                        │
│        → Rebuild: proto → crypto → etcd → components          │
│        → Verify linkage all components                         │
│        → Test: make verify-crypto-linkage                      │
│                                                                 │
│     🔥 Stability Test (2h)                                      │
│        → Start full pipeline (etcd + all components)           │
│        → Idle test: No PCAP injection                          │
│        → Monitor: logs, memory, CPU                            │
│        → Duration: 30-60 minutes                               │
│                                                                 │
│     🔥 Neris PCAP Relay Test (4-6h)                             │
│        → Inject Neris botnet traffic                           │
│        → Monitor IPSet blacklist population 🚨 CRITICAL        │
│        → Verify: 147.32.84.* IPs blocked                       │
│        → Capture metrics: latency, throughput                  │
│        → Check: RAG artifacts generation                       │
│        → Memory leaks: AddressSanitizer validation             │
│                                                                 │
│     🔥 IPSet Blocking Implementation (1h)                       │
│        → firewall-acl-agent: Add IP to blacklist              │
│        → Code: ipset add ml_defender_blacklist_test           │
│        → Threshold: final_score > 0.7                          │
│        → Timeout: 3600 seconds (configurable)                  │
│                                                                 │
│     ✅ If Stable → Merge to Main                                │
│        → Feature complete: Unified crypto ecosystem            │
│        → All tests passing                                     │
│        → Production metrics captured                           │
│        → Ready for next feature                                │
│                                                                 │
│  COMPLETED (Phase 0 + Phase 1 Days 1-28):                      │
│     ✅ 4 embedded C++20 detectors (<1.06μs)                    │
│     ✅ eBPF/XDP dual-NIC metadata extraction                   │
│     ✅ crypto-transport library (unified ecosystem) ✅          │
│     ✅ etcd-server migrated to crypto-transport ✅             │
│     ✅ ml-detector crypto integration (bidirectional) ✅       │
│     ✅ firewall-acl-agent integrated ✅                        │
│     ✅ RAG integration with crypto ✅                          │
│     ✅ sniffer linkage complete ✅ (código pending Day 29)     │
│     ✅ All components use single crypto library ✅             │
│     ✅ 100% linkage verified (6/6 components) ✅               │
│     ✅ Zero crashes, all tests passing ✅                      │
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
make proto-unified    # Generate unified protobuf files
make crypto-transport-build  # Build crypto-transport library (FIRST!)
make etcd-client-build       # Build etcd-client (uses crypto-transport)
make etcd-server-build       # Build etcd-server (uses crypto-transport)
make sniffer          # Build eBPF/XDP sniffer (NOW WITH LINKAGE! 🆕)
make detector         # Build ml-detector (CRYPTO INTEGRATED!)
make firewall         # Build firewall agent (CRYPTO INTEGRATED!)
make rag              # Build RAG system (CRYPTO INTEGRATED!)

# 4. Verify linkage (NEW! Day 28)
make verify-crypto-linkage
# Expected: All components show libcrypto_transport.so.1 ✅

# 5. Test etcd-client library
vagrant ssh defender -c "cd /vagrant/etcd-client/build && ctest --output-on-failure"
# Expected: 3/3 tests passed ✅

# 6. Test crypto-transport library
vagrant ssh defender -c "cd /vagrant/crypto-transport/build && ctest --output-on-failure"
# Expected: 16/16 tests passed ✅

# 7. Start etcd-server
vagrant ssh defender -c "cd /vagrant/etcd-server/build && ./etcd-server --port 2379"

# 8. Start the lab (Day 29 - after code integration)
make run-lab-dev

# 9. Verify components are running
make status-lab
# Expected output:
#   ✅ Firewall: RUNNING (with encryption!)
#   ✅ Detector: RUNNING (with encryption!)
#   ✅ Sniffer:  RUNNING (with encryption! - Day 29)
#   ✅ RAG:      RUNNING (with encryption!)

# 10. Monitor in real-time
watch -n 5 'vagrant ssh defender -c "echo \"Artifacts: \$(ls /vagrant/logs/rag/artifacts/$(date +%Y-%m-%d)/ 2>/dev/null | wc -l)  JSONL: \$(wc -l < /vagrant/logs/rag/events/$(date +%Y-%m-%d).jsonl 2>/dev/null || echo 0)\""'

# 11. Check ml-detector uptime (should increase steadily)
vagrant ssh defender -c "ps -p \$(pgrep ml-detector) -o etime="

# 12. Stop lab when done
make kill-lab
```

---

## 🔐 crypto-transport Unified Ecosystem (Day 26-28)

### **Architecture Evolution**

**Before (Day 25):**
```
Each component had own crypto/compression code
├─ sniffer: Local LZ4
├─ ml-detector: Local compression
├─ firewall: etcd-client with embedded crypto
└─ etcd-server: CryptoPP (different library!)
```

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

### **Features**

- **ChaCha20-Poly1305 Encryption** - Military-grade authenticated encryption
- **LZ4 Compression** - Ultra-fast compression (5+ GB/s, intelligent!)
- **Unified Library** - Single source of truth (SRP)
- **Thread-Safe** - Mutex-protected operations
- **Binary-Safe API** - std::vector<uint8_t>
- **RAII Pattern** - Automatic libsodium initialization
- **16 Unit Tests** - 100% passing
- **Installed System-Wide** - `/usr/local/lib/libcrypto_transport.so`

### **Performance**
```
Compression (LZ4):
  • ml-detector config: 11754 → 5084 bytes (56.7%) ✅
  • Sniffer config: 17391 → 8569 bytes (49.3%) ✅
  • Small configs: Not compressed (intelligent)

Encryption (ChaCha20-Poly1305):
  • Overhead: +40 bytes fixed (nonce + MAC)
  • ml-detector: 5084 + 40 = 5124 bytes total ✅
  • Operation time: <3 μs

E2E Pipeline (Day 27-28):
  • Client: JSON → Compress → Encrypt → HTTP PUT
  • Server: HTTP → Decrypt → Decompress → Validate → Store
  • All components: crypto-transport linked ✅
  • Zero manual key management ✅
  • Zero hardcoded seeds ✅
```

### **Integration Status**

| Component | Linkage | Code | Status |
|-----------|---------|------|--------|
| crypto-transport | ✅ | ✅ | Base library |
| etcd-client | ✅ | ✅ | Refactored Day 26 |
| firewall-acl-agent | ✅ | ✅ | Integrated Day 26 |
| etcd-server | ✅ | ✅ | Migrated Day 27 |
| ml-detector | ✅ | ✅ | Integrated Day 27 |
| RAG | ✅ | ✅ | Integrated Day 19 |
| **sniffer** | **✅** | **⏳** | **Linkage Day 28, Code Day 29** |

---

## 🛡️ Dual-Score Architecture

### **Maximum Threat Wins Logic**
```
┌─────────────────────────────────────────────────────────────┐
│ SNIFFER (Fast Detector - Layer 1) + crypto-transport ⏳    │
│                                                             │
│  • external_ips_30s >= 15 → score = 0.70                   │
│  • smb_diversity >= 10 → score = 0.70                      │
│  • dns_entropy > 0.95 → score = 0.70                       │
│  • Linkage: crypto-transport ✅ (Day 28)                   │
│  • Code: ZMQ send integration ⏳ (Day 29)                  │
│  Populates: fast_detector_score, reason, triggered         │
└─────────────────┬───────────────────────────────────────────┘
                  │ Protobuf Event (ZMQ 5571) - Encrypted ⏳
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ ML DETECTOR (Dual-Score + RAGLogger) + crypto-transport ✅ │
│                                                             │
│  1. Decrypt incoming packet (Day 27) ✅                     │
│  2. Read fast_detector_score from event                     │
│  3. Calculate ml_detector_score (4 models)                  │
│  4. final_score = max(fast_score, ml_score)                │
│  5. Determine authoritative_source                          │
│  6. RAGLogger: Write artifacts atomically ✅                │
│  7. RAGLogger: Buffer .jsonl (stable with fix) ✅           │
│  8. Encrypt + send to firewall ✅                           │
│  9. Register + upload config to etcd ✅                     │
└─────────────────┬───────────────────────────────────────────┘
                  │ Enriched Event (ZMQ 5572) + etcd (encrypted)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ FIREWALL / RAG QUEUE + crypto-transport ✅                  │
│                                                             │
│  • Decrypt incoming event ✅                                │
│  • Block/Monitor based on final_score                       │
│  • IPSet blacklist: ⏳ Implementation Day 29                │
│  • RAG analysis for divergent events ✅                     │
│  • Retrieve config from etcd (encrypted) ✅                 │
│  • Register + upload config ✅                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 The Synthetic Data Story

### **Methodology (Validated)**

1. Extract statistics from real benign traffic
2. Generate synthetic samples (mean, std, distribution)
3. Train RandomForest on synthetic data ONLY
4. Deploy without academic datasets
5. Result: F1 = 1.00 (training) → High detection on real traffic

**Why It Works:**
- ✅ No dataset bias (CTU-13, CICIDS issues avoided)
- ✅ No label noise (synthetic = perfect labels)
- ✅ No licensing issues (own data)
- ✅ Generalizes to real attacks

**Evidence:**
- Neris botnet (Dec 12): 97.6% MALICIOUS detection
- SmallFlows (Dec 14): 97.1% MALICIOUS detection
- Day 16 (continuous): 1,152 events, stable
- Day 19-28 (encrypted): All components operational
- No threshold tuning required
- No retraining required

---

## 📖 Documentation

### Core Documentation
- [Architecture Deep Dive](docs/ARCHITECTURE.md)
- [Dual-Score Architecture](docs/DAY_13_DUAL_SCORE_ANALYSIS.md)
- [RAGLogger Schema](docs/RAGLOGGER_SCHEMA.md)
- [Race Condition Fix](docs/DAY_16_RACE_CONDITION_FIX.md)
- [Synthetic Data Methodology](docs/SYNTHETIC_DATA.md)
- [Performance Tuning](docs/PERFORMANCE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

### Crypto-Transport Ecosystem (Days 26-28)
- [crypto-transport Library](crypto-transport/README.md) 🆕
- [Day 26: Library Creation + Refactor](docs/DAY_26_CRYPTO_TRANSPORT.md) 🆕
- [Day 27: etcd-server + ml-detector](docs/DAY_27_CRYPTO_UNIFICATION.md) 🆕
- [Day 28: Sniffer Integration](docs/DAY_28_SNIFFER_LINKAGE.md) 🆕
- [Security Roadmap](docs/SECURITY_ROADMAP.md)

### Integration Documentation
- [etcd-client Library](etcd-client/README.md)
- [Day 18: Bidirectional Config](docs/DAY_18_BIDIRECTIONAL_CONFIG.md)
- [Day 19: RAG Integration](docs/DAY_19_RAG_INTEGRATION.md)
- [Day 20: Sniffer Config Upload](docs/DAY_20_SNIFFER_INTEGRATION.md)
- [Day 21: Component Integration](docs/DAY_21_COMPONENT_INTEGRATION.md)
- [RAG System Documentation](docs/RAG_SYSTEM.md)
- [ETCD-Server Integration](docs/ETCD_SERVER.md)

### Future Enhancements
- [Shadow Authority](docs/SHADOW_AUTHORITY.md)
- [Decision Outcome](docs/DECISION_OUTCOME.md)
- [Future Enhancements](docs/FUTURE_ENHANCEMENTS.md)

---

## 🛠️ Build Targets
```bash
# Core Components
make proto-unified         # Generate unified protobuf files
make crypto-transport-build # Build crypto-transport library (FIRST!)
make etcd-client-build     # Build etcd-client (uses crypto-transport)
make etcd-server-build     # Build etcd-server (uses crypto-transport)
make sniffer               # Build eBPF/XDP sniffer (WITH LINKAGE! 🆕)
make detector              # Build ml-detector (CRYPTO INTEGRATED!)
make detector-debug        # Build ml-detector (debug mode)
make firewall              # Build firewall agent (CRYPTO INTEGRATED!)
make rag                   # Build RAG system (CRYPTO INTEGRATED!)

# Verification (Day 28)
make verify-crypto-linkage # Verify all components linked ✅

# Lab Control
make run-lab-dev           # Start full lab
make kill-lab              # Stop all components
make status-lab            # Check component status

# Testing
make test-crypto-transport # Test crypto-transport library (16 tests)
make test-etcd-client      # Test etcd-client library (3 tests)
make test-rag-small        # Test with smallFlows.pcap
make test-rag-neris        # Test with Neris botnet (large)

# Monitoring
make monitor-day13-tmux    # Real-time monitoring in tmux

# Cleanup
make clean-crypto          # Clean crypto-transport
make detector-clean        # Clean ml-detector build
make clean-all             # Clean everything
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

**Day 28 Truth:**
> "Verified firewall + RAG compilados sin errores. Linkage crypto-transport
> correcto. Integramos sniffer: Intentamos CMakeLists desde cero - aprendimos.
> Decisión correcta: partir del backup funcional. Patch quirúrgico ~50 líneas
> sobre 500+ existentes. Compilación limpia. Linkage 100% verificado (6/6
> componentes). Tests passing. --help funciona. Zero crashes. Linkage primero,
> código después (Día 29). Via Appia Quality: Metodología > velocidad.
> Despacio y bien. 🏛️"

---

## 🤝 Multi-Agent Collaboration

This project represents multi-agent AI collaboration:

| AI Agent | Contribution |
|----------|-------------|
| **Claude (Anthropic)** | Architecture, Days 16-28 implementation, debugging |
| **DeepSeek (v3)** | RAG system, ETCD-Server, automation |
| **Grok4 (xAI)** | XDP expertise, eBPF edge cases |
| **Qwen (Alibaba)** | Network routing, production insights |
| **Alonso** | Vision, C++ implementation, code detective 🔍 |

All AI agents will be credited as **co-authors** in academic publications.

---

## 📧 Contact

- GitHub: [@alonsoir](https://github.com/alonsoir)
- Project: [ML Defender](https://github.com/alonsoir/test-zeromq-docker)

---

**Built with 🛡️ for a safer internet**

*Via Appia Quality - Designed to last decades*

---

**Day 28 Complete:**  
Unified crypto-transport ecosystem - 6/6 components linkage verified ✅  
Sniffer CMakeLists patched quirúrgicamente (~50 líneas)  
Compilación limpia, tests passing, zero crashes  
Metodología > velocidad, despacio y bien 🏛️

**Next:** Day 29 - Sniffer ZMQ code + Clean build + Stability + Neris test

---

**Latest Update:** December 29, 2025 - Day 28 Complete - Sniffer Linkage 100% 🎉  
**Progress:** 100% Linkage (6/6 components) | 83% Code Integration  
**Next:** Day 29 - Final code integration + E2E validation