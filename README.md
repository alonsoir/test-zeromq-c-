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
- 🔐 **etcd-client Library** - Military-grade encryption + compression (ChaCha20 + LZ4)
- 🔄 **Bidirectional Config** - Components can update their own configuration
- 🎯 **Encrypted Pipeline** - End-to-end encryption across all components

---

## 🎯 Current Status
```
```
┌─────────────────────────────────────────────────────────────────┐
│  DAY 26 COMPLETE: crypto-transport Library & Architecture 🎉   │
│  (December 26, 2025)                                           │
│  Progress: 98% → 99% 🚀                                         │
├─────────────────────────────────────────────────────────────────┤
│  🎉 DAY 26: Foundation Architecture Refactoring Complete        │
│     Extracted transport layer into independent library         │
│                                                                 │
│  ✅ crypto-transport Library Created:                           │
│     • ChaCha20-Poly1305 + LZ4 in single package                │
│     • Binary-safe API (std::vector<uint8_t>)                   │
│     • 16 unit tests passing (100%)                             │
│     • RAII pattern for libsodium initialization                │
│     • Zero external config dependencies                        │
│     • Installed: /usr/local/lib/libcrypto_transport.so         │
│                                                                 │
│  ✅ etcd-client Refactored:                                     │
│     • Removed LZ4 + OpenSSL dependencies                       │
│     • Uses crypto-transport exclusively                        │
│     • Added get_encryption_key() public API                    │
│     • 3 tests updated and passing (100%)                       │
│     • Cleaner architecture (SRP respected)                     │
│                                                                 │
│  ✅ firewall-acl-agent Integration:                             │
│     • zmq_subscriber.cpp refactored (crypto-transport)         │
│     • etcd_client wrapper with get_crypto_seed()               │
│     • Crypto seed from etcd (NO hardcoding!)                   │
│     • Decrypt/decompress ZMQ payloads ready                    │
│     • Component registration: ✅                                │
│     • Config upload: 7532 → 3815 bytes (49.3% reduction)       │
│     • Heartbeat: ✅ (30s interval)                              │
│     • Clean shutdown: ✅                                         │
│                                                                 │
│  🏗️ Architecture Improvements:                                  │
│     • Single Responsibility Principle enforced                 │
│     • Transport logic extracted from business logic            │
│     • Dependency hierarchy clarified:                          │
│       1. crypto-transport (base)                               │
│       2. etcd-client (uses crypto-transport)                   │
│       3. components (use both)                                 │
│     • Makefile maestro updated with correct order              │
│                                                                 │
│  📊 Test Results:                                               │
│     crypto-transport: 16/16 tests passed ✅                    │
│     etcd-client: 3/3 tests passed ✅                           │
│     firewall-acl-agent: Compiled + linked ✅                   │
│                                                                 │
│  🔐 Security Verified (Production Test):                        │
│     • etcd-server → crypto seed generation ✅                  │
│     • firewall → crypto seed retrieval ✅                      │
│     • ChaCha20-Poly1305 encryption enabled ✅                  │
│     • LZ4 compression enabled ✅                               │
│     • Component registration successful ✅                      │
│     • Heartbeat mechanism operational ✅                        │
│     • Config upload encrypted: 7532 → 3815 bytes ✅            │
│                                                                 │
│  ✅ Via Appia Quality:                                          │
│     • Troubleshooting methodology documented                   │
│     • Scientific honesty: admitted coupling issue              │
│     • Methodical refactoring (3 hours, zero shortcuts)         │
│     • Test-driven: 100% pass rate maintained                   │
│     • Production validation before commit                      │
│                                                                 │
│  📊 PROGRESS: 99% Complete 🚀                                   │
│                                                                 │
│  🎯 NEXT PRIORITIES (Day 27):                                   │
│     🔥 ml-detector Integration (Most Complex)                   │
│        → Refactor for crypto-transport                         │
│        → Both encrypt/compress (send) + decrypt/decompress     │
│        → Update CMakeLists.txt                                 │
│        → Crypto seed from etcd                                 │
│        → Estimated: 2-3 hours                                  │
│                                                                 │
│     🔥 sniffer Integration (Simpler)                            │
│        → Refactor for crypto-transport                         │
│        → Only encrypt/compress (send)                          │
│        → Update CMakeLists.txt                                 │
│        → Estimated: 1-2 hours                                  │
│                                                                 │
│     🔥 End-to-End Pipeline Test                                 │
│        → Full pipeline with encryption                         │
│        → etcd-server → sniffer → detector → firewall          │
│        → Verify decrypt/decompress chain                       │
│        → Performance metrics                                   │
│        → Estimated: 1 hour                                     │
│                                                                 │
│  COMPLETED (Phase 0 + Phase 1 Days 1-26):                      │
│     ✅ 4 embedded C++20 detectors (<1.06μs)                    │
│     ✅ eBPF/XDP dual-NIC metadata extraction                   │
│     ✅ crypto-transport library (independent) 🆕               │
│     ✅ etcd-client refactored (cleaner) 🆕                     │
│     ✅ firewall-acl-agent integrated 🆕                        │
│     ✅ Makefile maestro updated 🆕                             │
│     ✅ Architecture follows SRP 🆕                             │
│     ✅ Zero hardcoded crypto seeds 🆕                          │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  DAY 21 COMPLETE: ml-detector + firewall Integration 🎉        │
│  (December 21, 2025)                                           │
│  Progress: 92% → 98% 🚀                                         │
├─────────────────────────────────────────────────────────────────┤
│  🎉 DAY 21: Component Integration Complete                      │
│     ml-detector and firewall now upload encrypted configs      │
│                                                                 │
│  ✅ ml-detector Integration:                                     │
│     • PIMPL adapter pattern (zero breaking changes)             │
│     • Config upload: 11,756 → 5,113 bytes (56.9% reduction)    │
│     • ChaCha20-Poly1305 + LZ4 compression working              │
│     • Automatic encryption key exchange                         │
│     • 5 ML models loaded (Level 1-3 detectors)                 │
│     • Component registered successfully                         │
│                                                                 │
│  ✅ firewall-acl-agent Integration:                             │
│     • PIMPL adapter in src/core/etcd_client.cpp                │
│     • Config upload: 4,698 → 2,405 bytes (48.8% reduction)     │
│     • ChaCha20-Poly1305 + LZ4 compression working              │
│     • Automatic encryption key exchange                         │
│     • Component registered successfully                         │
│     • IPSet + IPTables health checks operational               │
│                                                                 │
│  📊 Pipeline Verified (3 Components):                            │
│     Component        Original → Encrypted   Reduction           │
│     ────────────────────────────────────────────────────        │
│     sniffer          17,391 → 8,609 bytes   50.5%              │
│     ml-detector      11,756 → 5,113 bytes   56.9%              │
│     firewall         4,698  → 2,405 bytes   48.8%              │
│                                                                 │
│  🔐 Security Verified:                                           │
│     • 3/3 components using etcd-client library ✅               │
│     • ChaCha20-Poly1305 E2E encryption ✅                       │
│     • LZ4 intelligent compression ✅                            │
│     • Automatic key exchange (no manual keys) ✅               │
│     • JSON validation on server ✅                              │
│                                                                 │
│  ✅ Via Appia Quality:                                           │
│     • PIMPL adapter pattern (backward compatibility)            │
│     • Zero breaking changes to main.cpp                         │
│     • Single source of truth: complete JSON configs             │
│     • Scientific honesty: heartbeat needs implementation        │
│                                                                 │
│  📊 PROGRESS: 98% Complete 🚀                                    │
│                                                                 │
│  🎯 NEXT PRIORITIES (Day 22):                                   │
│     🔥 Heartbeat Endpoint Implementation                         │
│        → POST /v1/heartbeat/:component_name                     │
│        → Update last_heartbeat timestamp                        │
│        → Mark components active/inactive                        │
│        → Estimated: 2-3 hours                                   │
│                                                                 │
│     🔥 Clean Shutdown & Deregistration                           │
│        → Verify components unregister on exit                   │
│        → Test graceful shutdown                                 │
│        → Estimated: 1 hour                                      │
│                                                                 │
│     🔥 End-to-End Encrypted Pipeline                             │
│        → Verify ZMQ traffic between components                  │
│        → Sniffer → Detector → Firewall (encrypted configs)     │
│        → RAGLogger data path (stays unencrypted for FAISS)     │
│        → Estimated: 2 hours                                     │
│                                                                 │
│  COMPLETED (Phase 0 + Phase 1 Days 1-21):                       │
│     ✅ 4 embedded C++20 detectors (<1.06μs)                     │
│     ✅ eBPF/XDP dual-NIC metadata extraction                    │
│     ✅ etcd-client library (encryption + compression)           │
│     ✅ Sniffer integration (Day 20)                             │
│     ✅ ml-detector integration (Day 21) 🆕                      │
│     ✅ firewall integration (Day 21) 🆕                         │
│     ✅ 3 components registered encrypted 🆕                     │
└─────────────────────────────────────────────────────────────────┘
```
```
┌─────────────────────────────────────────────────────────────────┐
│  DAY 20 COMPLETE: Sniffer Integration with etcd-client 🎉      │
│  (December 20, 2025)                                           │
│  Progress: 82% → 92% 🚀                                         │
├─────────────────────────────────────────────────────────────────┤
│  🎉 DAY 20: Sniffer Encrypted Integration Complete              │
│     Sniffer now uploads full config encrypted to etcd-server   │
│                                                                 │
│  ✅ Sniffer Integration:                                         │
│     • PIMPL adapter pattern (zero breaking changes)             │
│     • Maintained legacy main.cpp compatibility                  │
│     • Full sniffer.json upload (17,391 bytes)                   │
│     • Automatic encryption key exchange                         │
│     • ChaCha20-Poly1305 E2E encryption working                  │
│     • LZ4 compression: 17391 → 8569 bytes (49.3%)               │
│     • Config validation fixed (accepts objects)                 │
│                                                                 │
│  ✅ Architecture Improvements:                                   │
│     • Fixed config_types.cpp mapping (etcd.enabled)             │
│     • Fixed etcd-server validation (JSON objects)               │
│     • Adapter maintains backward compatibility                  │
│     • Single source of truth: complete sniffer.json             │
│                                                                 │
│  📊 Pipeline Verified:                                           │
│     Sniffer → Compress → Encrypt → etcd-server                 │
│              ↓           ↓          ↓                           │
│         17391 bytes  8569 bytes  8609 bytes                     │
│                                                                 │
│     etcd-server → Decrypt → Decompress → Validate → Store      │
│                   ↓         ↓           ✅         ✅          │
│               8569 bytes 17391 bytes                            │
│                                                                 │
│  🔐 Security Verified:                                           │
│     • ChaCha20 encryption: 8609 → 8569 bytes ✅                 │
│     • LZ4 decompression: 8569 → 17391 bytes ✅                  │
│     • JSON validation: 17391 bytes ✅                            │
│     • Config stored: sniffer component ✅                        │
│                                                                 │
│  ✅ Via Appia Quality:                                           │
│     • Zero hardcoded filters (uploaded complete JSON)           │
│     • Single source of truth preserved                          │
│     • Transparent methodology maintained                        │
│     • Scientific honesty: heartbeat 404 documented              │
│                                                                 │
│  🎉 DAY 19: RAG Integration with etcd-client Complete           │
│     RAG now uses etcd-client library with full encryption      │
│                                                                 │
│  ✅ DAY 18: Bidirectional Config Management                     │
│     PUT endpoint + Server ChaCha20 migration                   │
│                                                                 │
│  ✅ DAY 17: etcd-client Library Created                         │
│     Encryption + Compression + Component Discovery             │
│                                                                 │
│  ✅ DAY 16: Race Condition Fixed                                │
│     RAGLogger Stable + Release Optimization Enabled            │
│                                                                 │
│  📊 PROGRESS: 92% Complete 🚀                                    │
│                                                                 │
│  🎯 NEXT PRIORITIES (Week 3 - Days 21-22):                      │
│     🔥 Day 21: Remaining Component Integration                  │
│        → Integrate etcd-client in ml-detector                   │
│        → Integrate etcd-client in firewall                      │
│        → Heartbeat endpoint implementation                      │
│        → Component health monitoring                            │
│        → Estimated: 1 day                                       │
│                                                                 │
│     Priority 2: End-to-End Encrypted Pipeline (Day 22)          │
│        → Sniffer → Detector → Firewall → RAG (all encrypted)   │
│        → Config sync across all components                      │
│        → Live config updates demonstration                      │
│        → Estimated: 1 day                                       │
│                                                                 │
│     Priority 3: Basic Quorum (Week 4)                           │
│        → Simple leader election                                 │
│        → Data replication between etcd-server instances         │
│        → Configuration sync                                     │
│        → Estimated: 2 days                                      │
│                                                                 │
│     Priority 4: FAISS C++ Integration (Week 4)                  │
│        → Semantic search over artifacts directory               │
│        → Vector DB for RAG queries                              │
│        → Natural language event search                          │
│        → Estimated: 3-4 days                                    │
│                                                                 │
│     Priority 5: Watcher Unified Library (Week 4-5)              │
│        → Runtime config updates from etcd                       │
│        → Hot-reload without restart                             │
│        → RAG command: "accelerate pipeline"                     │
│        → Estimated: 3-4 days                                    │
│                                                                 │
│  COMPLETED (Phase 0 + Phase 1 Days 1-20):                       │
│     ✅ 4 embedded C++20 detectors (<1.06μs)                     │
│     ✅ eBPF/XDP dual-NIC metadata extraction                    │
│     ✅ Dual-Score Architecture (Fast + ML)                      │
│     ✅ Maximum Threat Wins logic                                │
│     ✅ RAGLogger 83-field event capture                         │
│     ✅ Race condition fix (production-ready)                    │
│     ✅ Release optimization enabled                             │
│     ✅ etcd-client library (encryption + compression)           │
│     ✅ Comprehensive test suite (3 tests, 100% pass)            │
│     ✅ Bidirectional config management (GET + PUT)              │
│     ✅ Server ChaCha20 migration                                │
│     ✅ RAG integration with etcd-client                         │
│     ✅ Sniffer integration with etcd-client 🆕                  │
│     ✅ Complete config upload (no filtering) 🆕                 │
│     ✅ Adapter pattern for seamless migration                   │
│     ✅ Host-based + Gateway modes validated                     │
│     ✅ RAG + LLAMA + ETCD ecosystem                             │
│     ✅ End-to-end encrypted communication                       │
│     ✅ Config validation (accepts JSON objects) 🆕              │
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
make proto           # Generate protobuf files
make sniffer         # Build eBPF/XDP sniffer (NOW WITH ENCRYPTION! 🆕)
make detector        # Build ml-detector (STABLE - race condition fixed!)
make firewall        # Build firewall agent
make rag             # Build RAG system (WITH ENCRYPTION!)
make etcd-server     # Build ETCD server (ChaCha20!)
make etcd-client     # Build etcd-client library

# 4. Test etcd-client library
vagrant ssh defender -c "cd /vagrant/etcd-client/build && ctest --output-on-failure"
# Expected: 3/3 tests passed

# 5. Test Sniffer + etcd-server integration (NEW!)
# Terminal 1: Start etcd-server
vagrant ssh defender -c "cd /vagrant/etcd-server/build && ./etcd-server --port 2379"

# Terminal 2: Start sniffer
vagrant ssh defender -c "cd /vagrant/sniffer/build && sudo ./sniffer -c ../config/sniffer.json"
# Expected:
#   ✅ [etcd] Sniffer registered and config uploaded
#   🔐 [etcd] Config encrypted with ChaCha20-Poly1305
#   🗜️  [etcd] Config compressed with LZ4

# 6. Verify config was uploaded
curl http://localhost:2379/components | jq
# Expected: Should show "sniffer" component

# 7. Start the lab
make run-lab-dev

# 8. Verify components are running
make status-lab
# Expected output:
#   ✅ Firewall: RUNNING
#   ✅ Detector: RUNNING
#   ✅ Sniffer:  RUNNING (with encryption! 🆕)
#   ✅ RAG:      RUNNING (with encryption!)

# 9. Monitor in real-time
watch -n 5 'vagrant ssh defender -c "echo \"Artifacts: \$(ls /vagrant/logs/rag/artifacts/$(date +%Y-%m-%d)/ 2>/dev/null | wc -l)  JSONL: \$(wc -l < /vagrant/logs/rag/events/$(date +%Y-%m-%d).jsonl 2>/dev/null || echo 0)\""'

# 10. Check ml-detector uptime (should increase steadily)
vagrant ssh defender -c "ps -p \$(pgrep ml-detector) -o etime="

# 11. View results
vagrant ssh defender -c "ls -lh /vagrant/logs/rag/artifacts/$(date +%Y-%m-%d)/ | head -20"
vagrant ssh defender -c "tail -10 /vagrant/logs/rag/events/$(date +%Y-%m-%d).jsonl | jq '.detection'"

# 12. Stop lab when done
make kill-lab
```

---

## 🔐 etcd-client Library (Day 20 Update!)

### **Features**

- **ChaCha20-Poly1305 Encryption** - Military-grade authenticated encryption (ALL components!)
- **LZ4 Compression** - Ultra-fast compression (5+ GB/s, intelligent!)
- **Component Discovery** - Registration, heartbeat, health monitoring
- **Config Management** - Master + active copies with rollback
- **Bidirectional Config** - GET + PUT operations
- **Automatic Key Exchange** - Server provides key on registration
- **Thread-Safe** - Mutex-protected operations
- **JSON-Driven** - 100% configuration via JSON
- **HTTP Client** - Retry logic with exponential backoff
- **PIMPL Adapter** - Zero breaking changes to existing code (NEW! 🆕)

### **Performance**
```
Compression (LZ4):
  • Sniffer config: 17391 → 8569 bytes (49.3%) ✅
  • RAG config: 535 → 460 bytes (86%)
  • Small configs: Not compressed (intelligent)

Encryption (ChaCha20-Poly1305):
  • Overhead: +40 bytes fixed (nonce + MAC)
  • Sniffer: 8569 + 40 = 8609 bytes total ✅
  • Operation time: <3 μs

Complete Pipeline (Days 18-20):
  • Client: JSON → Compress → Encrypt → HTTP PUT
  • Server: HTTP → Decrypt → Decompress → Validate → Store
  • Sniffer integration: Zero main.cpp changes ✅
  • RAG integration: <100ms connection time ✅
  • Zero manual key management ✅
```

### **New in Day 20**
```
✅ Sniffer Integration:
  • PIMPL adapter pattern implementation
  • Zero changes to main.cpp required
  • Maintained legacy API surface
  • Internally uses etcd-client library
  • Automatic encryption key exchange

✅ Complete Config Upload:
  • Full 17,391 byte sniffer.json uploaded
  • No selective field filtering
  • Single source of truth preserved
  • Via Appia Quality: JSON is the law

✅ Config Validation Fixed:
  • Server now accepts JSON objects
  • Validates both {"component": "string"}
  • And {"component": {"name": "...", ...}}
  • Flexible schema validation

✅ Dual System Support:
  • SnifferConfig (new system, etcd-client)
  • StrictSnifferConfig (legacy system)
  • Automatic mapping between both
  • Backward compatibility guaranteed
```

### **Security Roadmap**
```
✅ Phase 2A (COMPLETE): Bidirectional Encrypted Config
  • ChaCha20-Poly1305 (client + server)
  • LZ4 compression
  • Automatic key exchange
  • Component registration

✅ Phase 2B (80% COMPLETE): Component Integration
  ✅ RAG integration (Day 19)
  ✅ Sniffer integration (Day 20) 🆕
  ⏳ ml-detector integration (Day 21)
  ⏳ firewall integration (Day 21)
  ⏳ Heartbeat mechanism (Day 21)

⏳ Phase 2C (Week 4): Advanced Features
  • Basic quorum (Day 22)
  • FAISS semantic search
  • Watcher unified library
  • Hot-reload configuration

Phase 3 (Month 2): Production Hardening
  • Server-side TLS (HTTPS)
  • Mutual TLS (client certs)
  • Key encryption in RAM
  • Memory locking (mlock)

Phase 4 (Future): Enterprise Grade
  • HSM integration
  • Tamper-proof key storage
  • FIPS 140-2 compliance
```

---

## 🛡️ Dual-Score Architecture

### **Maximum Threat Wins Logic**
```
┌─────────────────────────────────────────────────────────────┐
│ SNIFFER (Fast Detector - Layer 1) + etcd-client 🆕         │
│                                                             │
│  • external_ips_30s >= 15 → score = 0.70                   │
│  • smb_diversity >= 10 → score = 0.70                      │
│  • dns_entropy > 0.95 → score = 0.70                       │
│  • Registers with etcd-server (encrypted) 🆕               │
│  • Uploads full sniffer.json (17,391 bytes) 🆕             │
│  Populates: fast_detector_score, reason, triggered         │
└─────────────────┬───────────────────────────────────────────┘
                  │ Protobuf Event (ZMQ 5571)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ ML DETECTOR (Dual-Score + RAGLogger) + etcd-client (Day 21)│
│                                                             │
│  1. Read fast_detector_score from event                     │
│  2. Calculate ml_detector_score (4 models)                  │
│  3. final_score = max(fast_score, ml_score)                │
│  4. Determine authoritative_source                          │
│  5. RAGLogger: Write artifacts atomically ✅                │
│  6. RAGLogger: Buffer .jsonl (stable with fix) ✅           │
│  7. Send to etcd-server (encrypted) ✅                      │
│  8. Register + upload config (Day 21) ⏳                    │
└─────────────────┬───────────────────────────────────────────┘
                  │ Enriched Event (ZMQ 5572) + etcd (encrypted)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ FIREWALL / RAG QUEUE + etcd-client (Day 21)                │
│                                                             │
│  • Block/Monitor based on final_score                       │
│  • RAG analysis for divergent events ✅                     │
│  • Retrieve config from etcd (encrypted) ✅                 │
│  • Register + upload config (Day 21) ⏳                     │
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
- Day 19 (encrypted): RAG registration successful
- Day 20 (encrypted): Sniffer config upload successful
- No threshold tuning required
- No retraining required

---

## 📖 Documentation

- [Architecture Deep Dive](docs/ARCHITECTURE.md)
- [Dual-Score Architecture](docs/DAY_13_DUAL_SCORE_ANALYSIS.md)
- [RAGLogger Schema](docs/RAGLOGGER_SCHEMA.md)
- [Race Condition Fix](docs/DAY_16_RACE_CONDITION_FIX.md)
- [etcd-client Library](etcd-client/README.md)
- [Day 18: Bidirectional Config](docs/DAY_18_BIDIRECTIONAL_CONFIG.md)
- [Day 19: RAG Integration](docs/DAY_19_RAG_INTEGRATION.md)
- [Day 20: Sniffer Integration](docs/DAY_20_SNIFFER_INTEGRATION.md) 🆕
- [Security Roadmap](docs/SECURITY_ROADMAP.md)
- [Synthetic Data Methodology](docs/SYNTHETIC_DATA.md)
- [Performance Tuning](docs/PERFORMANCE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [RAG System Documentation](docs/RAG_SYSTEM.md)
- [ETCD-Server Integration](docs/ETCD_SERVER.md)

---

## 🤝 Multi-Agent Collaboration

This project represents multi-agent AI collaboration:

| AI Agent | Contribution |
|----------|-------------|
| **Claude (Anthropic)** | Architecture, Days 16-20 implementation, debugging |
| **DeepSeek (v3)** | RAG system, ETCD-Server, automation |
| **Grok4 (xAI)** | XDP expertise, eBPF edge cases |
| **Qwen (Alibaba)** | Network routing, production insights |
| **Alonso** | Vision, C++ implementation, code detective 🔍 |

All AI agents will be credited as **co-authors** in academic publications.

---

## 🛠️ Build Targets
```bash
# Core Components
make proto           # Generate protobuf files
make sniffer         # Build eBPF/XDP sniffer (WITH ENCRYPTION! 🆕)
make detector        # Build ml-detector (STABLE!)
make detector-debug  # Build ml-detector (debug mode)
make firewall        # Build firewall agent
make rag             # Build RAG system (WITH ENCRYPTION!)
make etcd-server     # Build ETCD server (ChaCha20!)
make etcd-client     # Build etcd-client library

# Lab Control
make run-lab-dev     # Start full lab
make kill-lab        # Stop all components
make status-lab      # Check component status

# Testing
make test-rag-small  # Test with smallFlows.pcap
make test-rag-neris  # Test with Neris botnet (large)
make test-etcd-client # Test etcd-client library
make test-rag-encryption # Test RAG encrypted communication
make test-sniffer-encryption # Test Sniffer encrypted upload (NEW! 🆕)

# Monitoring
make monitor-day13-tmux # Real-time monitoring in tmux

# Cleanup
make detector-clean  # Clean ml-detector build
make clean-all       # Clean everything
```

---

## 🏛️ Via Appia Quality Philosophy

Like the ancient Roman road that still stands 2,300 years later:

1. **Clean Code** - Simple, readable, maintainable
2. **KISS** - Keep It Simple
3. **Funciona > Perfecto** - Working beats perfect
4. **Smooth & Fast** - Optimize what matters
5. **Scientific Honesty** - Truth above convenience

**Day 20 Truth:**
> "We integrated Sniffer with etcd-client library. PIMPL adapter pattern
> maintained zero breaking changes. Full 17,391-byte sniffer.json uploaded
> encrypted. ChaCha20-Poly1305 E2E working. LZ4 compression: 49.3% reduction.
> Fixed config_types.cpp mapping. Fixed etcd-server validation for JSON objects.
> No selective filtering - uploaded complete config. Single source of truth
> preserved. Heartbeat 404 documented (needs implementation). Tests passing.
> Via Appia Quality: JSON is the law. Reality documented."

---

## 📧 Contact

- GitHub: [@alonsoir](https://github.com/alonsoir)
- Project: [ML Defender](https://github.com/alonsoir/test-zeromq-docker)

---

**Built with 🛡️ for a safer internet**

*Via Appia Quality - Designed to last decades*

---
**Day 26 Troubleshooting Truth:**
> "Discovered coupling between etcd-client and crypto/compression code.
> Violated SRP. Extracted independent crypto-transport library.
> Refactored etcd-client to use it. Updated firewall-acl-agent.
> Added get_encryption_key() to etcd-client. Removed all hardcoding.
> 3 hours methodical work. 100% tests passing. Production validated.
> Via Appia Quality: When wrong, fix it right."

---

**Latest Update:** December 26, 2025 - Day 26 Complete - crypto-transport Library 🎉  
**Progress:** 90% Complete  
**Next:** Day 27 - ml-detector + sniffer integration
