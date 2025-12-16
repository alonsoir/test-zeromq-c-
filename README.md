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

---

## 🎯 Current Status
```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 COMPLETE + DAY 17 etcd-client Library 🎉              │
│  (December 16, 2025 - 08:45 AM)                                │
├─────────────────────────────────────────────────────────────────┤
│  ✅ DAY 17 COMPLETE: etcd-client Library Created               │
│     Encryption + Compression + Component Discovery             │
│                                                                 │
│  🎉 NEW LIBRARY: etcd-client (1,238 lines C++20)                │
│     • ChaCha20-Poly1305 encryption (libsodium)                  │
│     • LZ4 compression (ultra-fast)                              │
│     • Component registration/discovery                          │
│     • Automatic heartbeat mechanism                             │
│     • Config versioning (master + active)                       │
│     • Thread-safe operations                                    │
│     • HTTP client with retry logic                              │
│     • 100% JSON-driven configuration                            │
│                                                                 │
│  Day 17 Achievements:                                           │
│     Structure & API Design:                                     │
│       ✅ Directory structure created                            │
│       ✅ CMakeLists.txt with libsodium/lz4 detection            │
│       ✅ Complete API designed (etcd_client.hpp)                │
│       ✅ Example configuration JSON                             │
│       ✅ README with design principles                          │
│                                                                 │
│     Core Implementation (6 modules, 1,238 lines):               │
│       ✅ config_loader.cpp (110 lines)                          │
│       ✅ compression_lz4.cpp (82 lines)                         │
│       ✅ crypto_chacha20.cpp (142 lines)                        │
│       ✅ http_client.cpp (178 lines)                            │
│       ✅ component_registration.cpp (119 lines)                 │
│       ✅ etcd_client.cpp (607 lines - PIMPL)                    │
│                                                                 │
│     Compilation:                                                │
│       ✅ libetcd_client.so.1.0.0 (1.1 MB)                       │
│       ✅ Zero warnings, zero errors                             │
│       ✅ g++ 12.2.0 with -std=c++20                             │
│       ✅ Dependencies: libsodium 1.0.18, liblz4 1.9.4           │
│                                                                 │
│     Comprehensive Tests (515 lines, 3 tests):                   │
│       ✅ test_compression.cpp (136 lines)                       │
│          • 10KB repetitive → 59 bytes (0.59% compression!)      │
│          • Random data compression validated                    │
│          • Threshold logic tested                               │
│          • Empty data edge cases covered                        │
│                                                                 │
│       ✅ test_encryption.cpp (202 lines)                        │
│          • ChaCha20-Poly1305 validated                          │
│          • Overhead: +40 bytes fixed (24 nonce + 16 MAC)        │
│          • Wrong key rejection tested                           │
│          • Corrupted data detection verified                    │
│          • Nonce randomness confirmed                           │
│                                                                 │
│       ✅ test_pipeline.cpp (177 lines)                          │
│          • Complete pipeline: Compress → Encrypt → Decrypt      │
│          • 100KB data → 452 bytes (0.452% total!)               │
│          • JSON config: 535 → 460 bytes (86% efficiency)        │
│          • Production use case validated                        │
│                                                                 │
│     Test Results (CTest):                                       │
│       • 3/3 tests passed (0.05 seconds)                         │
│       • Compression ratio: 99.41% reduction (repetitive data)   │
│       • Encryption overhead: 0.39% (large data)                 │
│       • Pipeline validated: Data integrity preserved            │
│                                                                 │
│     Security Design:                                            │
│       ✅ ChaCha20-Poly1305 (TLS 1.3 standard)                   │
│       ✅ Authenticated encryption (MAC verification)            │
│       ✅ Random nonces (prevents replay attacks)                │
│       ✅ Key management designed (etcd-server generates)        │
│       ✅ mTLS roadmap documented (Phase 2B)                     │
│       ✅ HSM integration planned (Phase 3)                      │
│                                                                 │
│  Performance Metrics:                                           │
│     ✅ Encryption: <1-3 μs per operation                        │
│     ✅ Compression: <1-2 μs per operation                       │
│     ✅ Total overhead: ~8 μs for config reload (amortized)      │
│     ✅ Per-packet impact: 0 μs (config cached)                  │
│     ✅ Storage efficiency: 0.4-0.5% of original size            │
│                                                                 │
│  ✅ DAY 16 COMPLETE: Race Condition Fixed (Previous)           │
│     RAGLogger Stable + Release Optimization Enabled            │
│                                                                 │
│  📊 PHASE 1 PROGRESS: 17/17 days complete (100%) 🎉             │
│                                                                 │
│  🎯 PHASE 2A PRIORITIES (Week 3 - Next Steps):                  │
│     🔥 Priority 1: RAG Integration with etcd-client (Day 18)    │
│        → Update rag/CMakeLists.txt                              │
│        → Replace rag/src/etcd_client.cpp with library           │
│        → Update rag-config.json format                          │
│        → Test registration/heartbeat                            │
│        → Estimated: 1 day                                       │
│                                                                 │
│     Priority 2: Component Integration (Day 19-20)               │
│        → ml-detector, sniffer, firewall integration             │
│        → End-to-end encrypted communication                     │
│        → Config distribution via etcd                           │
│        → Estimated: 2 days                                      │
│                                                                 │
│     Priority 3: FAISS C++ Integration                           │
│        → Semantic search over artifacts directory               │
│        → Vector DB for RAG queries                              │
│        → Natural language event search                          │
│        → Estimated: 3-4 days                                    │
│                                                                 │
│     Priority 4: Watcher Unified Library                         │
│        → Runtime config updates from etcd                       │
│        → Hot-reload without restart                             │
│        → RAG command: "accelerate pipeline"                     │
│        → Estimated: 3-4 days                                    │
│                                                                 │
│     Priority 5: Server-side TLS (Phase 2B)                      │
│        → HTTPS with server certificates                         │
│        → Key encrypted in transit                               │
│        → Estimated: 2 days                                      │
│                                                                 │
│     Priority 6: Mutual TLS (Phase 2B)                           │
│        → Client certificates                                    │
│        → Bidirectional authentication                           │
│        → Per-component authorization                            │
│        → Estimated: 3 days                                      │
│                                                                 │
│  COMPLETED (Phase 0 + Phase 1 Days 1-17):                       │
│     ✅ 4 embedded C++20 detectors (<1.06μs)                     │
│     ✅ eBPF/XDP dual-NIC metadata extraction                    │
│     ✅ Dual-Score Architecture (Fast + ML)                      │
│     ✅ Maximum Threat Wins logic                                │
│     ✅ RAGLogger 83-field event capture                         │
│     ✅ Race condition fix (production-ready)                    │
│     ✅ Release optimization enabled                             │
│     ✅ etcd-client library (encryption + compression) 🆕        │
│     ✅ Comprehensive test suite (3 tests, 100% pass) 🆕         │
│     ✅ Host-based + Gateway modes validated                     │
│     ✅ RAG + LLAMA + ETCD ecosystem                             │
│     ✅ End-to-end test validated                                │
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
make sniffer         # Build eBPF/XDP sniffer
make detector        # Build ml-detector (STABLE - race condition fixed!)
make firewall        # Build firewall agent
make rag             # Build RAG system
make etcd-server     # Build ETCD server
make etcd-client     # Build etcd-client library (NEW!)

# 4. Test etcd-client library (NEW!)
vagrant ssh defender -c "cd /vagrant/etcd-client/build && ctest --output-on-failure"
# Expected: 3/3 tests passed

# 5. Verify RAGLogger configuration
vagrant ssh defender -c "jq '.rag_logger' /vagrant/ml-detector/config/ml_detector_config.json"
# Should show: enabled=true, flush_interval_seconds=5

# 6. Start the lab
make run-lab-dev

# 7. Verify components are running
make status-lab
# Expected output:
#   ✅ Firewall: RUNNING
#   ✅ Detector: RUNNING
#   ✅ Sniffer:  RUNNING

# 8. Monitor in real-time
watch -n 5 'vagrant ssh defender -c "echo \"Artifacts: \$(ls /vagrant/logs/rag/artifacts/$(date +%Y-%m-%d)/ 2>/dev/null | wc -l)  JSONL: \$(wc -l < /vagrant/logs/rag/events/$(date +%Y-%m-%d).jsonl 2>/dev/null || echo 0)\""'

# 9. Check ml-detector uptime (should increase steadily)
vagrant ssh defender -c "ps -p \$(pgrep ml-detector) -o etime="

# 10. View results
vagrant ssh defender -c "ls -lh /vagrant/logs/rag/artifacts/$(date +%Y-%m-%d)/ | head -20"
vagrant ssh defender -c "tail -10 /vagrant/logs/rag/events/$(date +%Y-%m-%d).jsonl | jq '.detection'"

# 11. Stop lab when done
make kill-lab
```

---

## 🔐 etcd-client Library (NEW!)

### **Features**

- **ChaCha20-Poly1305 Encryption** - Military-grade authenticated encryption
- **LZ4 Compression** - Ultra-fast compression (5+ GB/s)
- **Component Discovery** - Registration, heartbeat, health monitoring
- **Config Management** - Master + active copies with rollback
- **Thread-Safe** - Mutex-protected operations
- **JSON-Driven** - 100% configuration via JSON
- **HTTP Client** - Retry logic with exponential backoff

### **Performance**
```
Compression (LZ4):
  • 10KB repetitive → 59 bytes (0.59%)
  • 100KB repetitive → 452 bytes (0.452%)
  • JSON config: 535 → 460 bytes (86%)

Encryption (ChaCha20-Poly1305):
  • Overhead: +40 bytes fixed (nonce + MAC)
  • Large data: +0.39% overhead
  • Operation time: <3 μs

Pipeline (Compress → Encrypt):
  • 100KB → 452 bytes total
  • 221x size reduction
  • Data integrity verified
```

### **Security Roadmap**
```
Phase 2A (Week 3): Server-side TLS
  • HTTPS with server certificates
  • Key encrypted in transit
  
Phase 2B (Week 4-5): Mutual TLS
  • Client certificates
  • Bidirectional authentication
  • Per-component authorization

Phase 2C (Month 2+): Key Protection
  • Key encrypted in RAM
  • Memory locking (mlock)
  • Secure wiping (sodium_memzero)

Phase 3 (Future): HSM Integration
  • Hardware Security Module
  • Tamper-proof key storage
  • FIPS 140-2 compliance
```

---

## 🛡️ Dual-Score Architecture

### **Maximum Threat Wins Logic**
```
┌─────────────────────────────────────────────────────────────┐
│ SNIFFER (Fast Detector - Layer 1)                          │
│                                                             │
│  • external_ips_30s >= 15 → score = 0.70                   │
│  • smb_diversity >= 10 → score = 0.70                      │
│  • dns_entropy > 0.95 → score = 0.70                       │
│  Populates: fast_detector_score, reason, triggered         │
└─────────────────┬───────────────────────────────────────────┘
                  │ Protobuf Event (ZMQ 5571)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ ML DETECTOR (Dual-Score + RAGLogger)                        │
│                                                             │
│  1. Read fast_detector_score from event                     │
│  2. Calculate ml_detector_score (4 models)                  │
│  3. final_score = max(fast_score, ml_score)                │
│  4. Determine authoritative_source                          │
│  5. RAGLogger: Write artifacts atomically ✅                │
│  6. RAGLogger: Buffer .jsonl (stable with fix) ✅           │
└─────────────────┬───────────────────────────────────────────┘
                  │ Enriched Event (ZMQ 5572)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ FIREWALL / RAG QUEUE                                        │
│                                                             │
│  • Block/Monitor based on final_score                       │
│  • RAG analysis for divergent events                       │
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
- No threshold tuning required
- No retraining required

---

## 📖 Documentation

- [Architecture Deep Dive](docs/ARCHITECTURE.md)
- [Dual-Score Architecture](docs/DAY_13_DUAL_SCORE_ANALYSIS.md)
- [RAGLogger Schema](docs/RAGLOGGER_SCHEMA.md)
- [Race Condition Fix](docs/DAY_16_RACE_CONDITION_FIX.md)
- [etcd-client Library](etcd-client/README.md) 🆕
- [Security Roadmap](docs/SECURITY_ROADMAP.md) 🆕
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
| **Claude (Anthropic)** | Architecture, Day 16-17 fixes, validation |
| **DeepSeek (v3)** | RAG system, ETCD-Server, automation |
| **Grok4 (xAI)** | XDP expertise, eBPF edge cases |
| **Qwen (Alibaba)** | Network routing, production insights |
| **Alonso** | Vision, C++ implementation, leadership |

All AI agents will be credited as **co-authors** in academic publications.

---

## 🛠️ Build Targets
```bash
# Core Components
make proto           # Generate protobuf files
make sniffer         # Build eBPF/XDP sniffer
make detector        # Build ml-detector (STABLE!)
make detector-debug  # Build ml-detector (debug mode)
make firewall        # Build firewall agent
make rag             # Build RAG system
make etcd-server     # Build ETCD server
make etcd-client     # Build etcd-client library (NEW!)

# Lab Control
make run-lab-dev     # Start full lab
make kill-lab        # Stop all components
make status-lab      # Check component status

# Testing
make test-rag-small  # Test with smallFlows.pcap
make test-rag-neris  # Test with Neris botnet (large)
make test-etcd-client # Test etcd-client library (NEW!)

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

**Day 17 Truth:**
> "We created etcd-client library from scratch. 1,238 lines C++20 with
> ChaCha20 + LZ4. Compiled successfully. 3 tests, all pass. 100KB data →
> 452 bytes (0.452%). Security roadmap designed. Tomorrow: RAG integration.
> Reality documented, not narratives."

---

## 📧 Contact

- GitHub: [@alonsoir](https://github.com/alonsoir)
- Project: [ML Defender](https://github.com/alonsoir/test-zeromq-docker)

---

**Built with 🛡️ for a safer internet**

*Via Appia Quality - Designed to last decades*

---

**Latest Update:** December 16, 2025 - Day 17 Complete - etcd-client Library 🎉  
**Next:** Day 18 - RAG Integration with etcd-client Library
