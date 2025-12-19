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

---

## 🎯 Current Status
```
┌─────────────────────────────────────────────────────────────────┐
│  DAY 19 COMPLETE: RAG Integration with etcd-client 🎉          │
│  (December 19, 2025)                                           │
│  Progress: 72% → 82% 🚀                                         │
├─────────────────────────────────────────────────────────────────┤
│  🎉 DAY 19: RAG Integration Complete                            │
│     RAG now uses etcd-client library with full encryption      │
│                                                                 │
│  ✅ Adapter Pattern Implementation:                             │
│     • Zero changes to main.cpp                                  │
│     • Maintained legacy API for compatibility                   │
│     • Internally uses new etcd-client library                   │
│     • Automatic encryption key exchange                         │
│     • ChaCha20 encryption end-to-end                            │
│     • Intelligent compression (only when beneficial)            │
│                                                                 │
│  ✅ RAG Capabilities:                                            │
│     • Component registration with etcd-server                   │
│     • Config upload with encryption                             │
│     • Config retrieval from other components                    │
│     • LLM-powered natural language commands                     │
│     • Semantic search ready for FAISS integration               │
│                                                                 │
│  📊 Performance Metrics:                                         │
│     • Connection time: <100ms                                   │
│     • Config upload: <50ms                                      │
│     • Encryption overhead: +40 bytes (nonce + MAC)              │
│     • Small configs: No compression (intelligent)               │
│     • Large configs: 40% compression with LZ4                   │
│                                                                 │
│  ✅ DAY 18: Bidirectional Config Management                     │
│     PUT endpoint + Server ChaCha20 migration                   │
│                                                                 │
│  Day 18 Achievements:                                           │
│     PUT Endpoint Implementation:                                │
│       ✅ http_client.cpp: put() function with retry             │
│       ✅ etcd_client.cpp: put_config() method                   │
│       ✅ etcd_server.cpp: PUT /v1/config/:id endpoint           │
│       ✅ X-Original-Size header for decompression               │
│                                                                 │
│     Server Migration to ChaCha20:                               │
│       ✅ Migrated from AES-CBC to ChaCha20-Poly1305             │
│       ✅ Same algorithm as client (compatibility)               │
│       ✅ Added LZ4 decompression to server                      │
│       ✅ Intelligent compression detection                      │
│       ✅ Server-side compression_lz4.cpp created                │
│                                                                 │
│     Automatic Key Exchange:                                     │
│       ✅ Server returns encryption_key on /register             │
│       ✅ Client receives and uses key automatically             │
│       ✅ Hex-to-binary conversion for proper key format         │
│       ✅ No manual key management required                      │
│                                                                 │
│     End-to-End Testing:                                         │
│       ✅ Client: 362B → 217B (compress) → 257B (encrypt)        │
│       ✅ Server: 257B → 217B (decrypt) → 362B (decompress)      │
│       ✅ JSON integrity verified                                │
│       ✅ All tests passing                                      │
│                                                                 │
│  Security Architecture:                                         │
│     ✅ ChaCha20-Poly1305 (client + server)                      │
│     ✅ Automatic key derivation with HKDF                       │
│     ✅ Per-session nonces (replay attack prevention)            │
│     ✅ Authenticated encryption (MAC verification)              │
│     ✅ LZ4 compression (when beneficial)                        │
│     ✅ Thread-safe operations (mutex-protected)                 │
│                                                                 │
│  ✅ DAY 17: etcd-client Library Created                         │
│     Encryption + Compression + Component Discovery             │
│                                                                 │
│  ✅ DAY 16: Race Condition Fixed                                │
│     RAGLogger Stable + Release Optimization Enabled            │
│                                                                 │
│  📊 PROGRESS: 82% Complete 🚀                                    │
│                                                                 │
│  🎯 NEXT PRIORITIES (Week 3 - Days 20-22):                      │
│     🔥 Day 20: Component Integration (ml-detector, sniffer)     │
│        → Integrate etcd-client in ml-detector                   │
│        → Integrate etcd-client in sniffer                       │
│        → Integrate etcd-client in firewall                      │
│        → End-to-end encrypted pipeline                          │
│        → Estimated: 1 day                                       │
│                                                                 │
│     Priority 2: Heartbeat Implementation (Day 21)               │
│        → POST /heartbeat endpoint in etcd-server                │
│        → Health monitoring                                      │
│        → Component status tracking                              │
│        → Estimated: 0.5 days                                    │
│                                                                 │
│     Priority 3: Basic Quorum (Day 22)                           │
│        → Simple leader election                                 │
│        → Data replication between etcd-server instances         │
│        → Configuration sync                                     │
│        → Estimated: 1 day                                       │
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
│  COMPLETED (Phase 0 + Phase 1 Days 1-19):                       │
│     ✅ 4 embedded C++20 detectors (<1.06μs)                     │
│     ✅ eBPF/XDP dual-NIC metadata extraction                    │
│     ✅ Dual-Score Architecture (Fast + ML)                      │
│     ✅ Maximum Threat Wins logic                                │
│     ✅ RAGLogger 83-field event capture                         │
│     ✅ Race condition fix (production-ready)                    │
│     ✅ Release optimization enabled                             │
│     ✅ etcd-client library (encryption + compression)           │
│     ✅ Comprehensive test suite (3 tests, 100% pass)            │
│     ✅ Bidirectional config management (GET + PUT) 🆕          │
│     ✅ Server ChaCha20 migration 🆕                             │
│     ✅ RAG integration with etcd-client 🆕                      │
│     ✅ Adapter pattern for seamless migration 🆕                │
│     ✅ Host-based + Gateway modes validated                     │
│     ✅ RAG + LLAMA + ETCD ecosystem                             │
│     ✅ End-to-end encrypted communication 🆕                    │
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
make rag             # Build RAG system (NOW WITH ENCRYPTION! 🆕)
make etcd-server     # Build ETCD server (ChaCha20! 🆕)
make etcd-client     # Build etcd-client library

# 4. Test etcd-client library
vagrant ssh defender -c "cd /vagrant/etcd-client/build && ctest --output-on-failure"
# Expected: 3/3 tests passed

# 5. Test RAG integration (NEW!)
# Terminal 1: Start etcd-server
vagrant ssh defender -c "cd /vagrant/etcd-server/build && ./etcd-server --port 2379"

# Terminal 2: Start RAG with encryption
vagrant ssh defender -c "cd /vagrant/rag/build && export LD_LIBRARY_PATH=/vagrant/etcd-client/build:\$LD_LIBRARY_PATH && ./rag-security"
# Expected: ✅ Service registered successfully

# 6. Start the lab
make run-lab-dev

# 7. Verify components are running
make status-lab
# Expected output:
#   ✅ Firewall: RUNNING
#   ✅ Detector: RUNNING
#   ✅ Sniffer:  RUNNING
#   ✅ RAG:      RUNNING (with encryption! 🆕)

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

## 🔐 etcd-client Library (Updated!)

### **Features**

- **ChaCha20-Poly1305 Encryption** - Military-grade authenticated encryption (client + server! 🆕)
- **LZ4 Compression** - Ultra-fast compression (5+ GB/s, intelligent! 🆕)
- **Component Discovery** - Registration, heartbeat, health monitoring
- **Config Management** - Master + active copies with rollback
- **Bidirectional Config** - GET + PUT operations (NEW! 🆕)
- **Automatic Key Exchange** - Server provides key on registration (NEW! 🆕)
- **Thread-Safe** - Mutex-protected operations
- **JSON-Driven** - 100% configuration via JSON
- **HTTP Client** - Retry logic with exponential backoff

### **Performance**
```
Compression (LZ4):
  • 10KB repetitive → 59 bytes (0.59%)
  • 100KB repetitive → 452 bytes (0.452%)
  • JSON config: 535 → 460 bytes (86%)
  • Small configs: Not compressed (intelligent)

Encryption (ChaCha20-Poly1305):
  • Overhead: +40 bytes fixed (nonce + MAC)
  • Large data: +0.39% overhead
  • Operation time: <3 μs

Complete Pipeline (Day 18-19):
  • Client: JSON → Compress → Encrypt → HTTP PUT
  • Server: HTTP → Decrypt → Decompress → Validate → Store
  • RAG integration: <100ms connection time
  • Zero manual key management
```

### **New in Day 18-19**
```
✅ Bidirectional Config:
  • PUT /v1/config/:id endpoint
  • Automatic compression (when beneficial)
  • Intelligent size detection
  • X-Original-Size header protocol

✅ Server ChaCha20 Migration:
  • Migrated from AES-CBC to ChaCha20-Poly1305
  • Algorithm parity with client
  • LZ4 decompression support
  • Authenticated encryption with MAC

✅ RAG Integration:
  • Adapter pattern (zero breaking changes)
  • Automatic encryption key exchange
  • Config upload/retrieval working
  • End-to-end encrypted communication

✅ Security Improvements:
  • HKDF key derivation
  • Per-session random nonces
  • Replay attack prevention
  • Thread-safe key management
```

### **Security Roadmap**
```
✅ Phase 2A (COMPLETE): Bidirectional Encrypted Config
  • ChaCha20-Poly1305 (client + server)
  • LZ4 compression
  • Automatic key exchange
  • Component registration

⏳ Phase 2B (Week 3-4): Component Integration
  • ml-detector integration (Day 20)
  • sniffer integration (Day 20)
  • firewall integration (Day 20)
  • Heartbeat mechanism (Day 21)
  • Basic quorum (Day 22)

Phase 2C (Week 4-5): Advanced Features
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
│  7. Send to etcd-server (encrypted) 🆕                     │
└─────────────────┬───────────────────────────────────────────┘
                  │ Enriched Event (ZMQ 5572) + etcd (encrypted)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ FIREWALL / RAG QUEUE                                        │
│                                                             │
│  • Block/Monitor based on final_score                       │
│  • RAG analysis for divergent events                       │
│  • Retrieve config from etcd (encrypted) 🆕                │
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
- No threshold tuning required
- No retraining required

---

## 📖 Documentation

- [Architecture Deep Dive](docs/ARCHITECTURE.md)
- [Dual-Score Architecture](docs/DAY_13_DUAL_SCORE_ANALYSIS.md)
- [RAGLogger Schema](docs/RAGLOGGER_SCHEMA.md)
- [Race Condition Fix](docs/DAY_16_RACE_CONDITION_FIX.md)
- [etcd-client Library](etcd-client/README.md)
- [Day 18: Bidirectional Config](docs/DAY_18_BIDIRECTIONAL_CONFIG.md) 🆕
- [Day 19: RAG Integration](docs/DAY_19_RAG_INTEGRATION.md) 🆕
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
| **Claude (Anthropic)** | Architecture, Days 16-19 implementation, validation |
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
make rag             # Build RAG system (WITH ENCRYPTION! 🆕)
make etcd-server     # Build ETCD server (ChaCha20! 🆕)
make etcd-client     # Build etcd-client library

# Lab Control
make run-lab-dev     # Start full lab
make kill-lab        # Stop all components
make status-lab      # Check component status

# Testing
make test-rag-small  # Test with smallFlows.pcap
make test-rag-neris  # Test with Neris botnet (large)
make test-etcd-client # Test etcd-client library
make test-rag-encryption # Test RAG encrypted communication (NEW! 🆕)

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

**Day 19 Truth:**
> "We integrated RAG with etcd-client library. Adapter pattern maintained
> backward compatibility. Zero changes to main.cpp. Automatic encryption
> key exchange working. ChaCha20 end-to-end. RAG registers, uploads config,
> retrieves data - all encrypted. Connection: <100ms. Smart compression:
> only when beneficial. Tests passing. Reality documented."

---

## 📧 Contact

- GitHub: [@alonsoir](https://github.com/alonsoir)
- Project: [ML Defender](https://github.com/alonsoir/test-zeromq-docker)

---

**Built with 🛡️ for a safer internet**

*Via Appia Quality - Designed to last decades*

---

**Latest Update:** December 19, 2025 - Day 19 Complete - RAG Integration 🎉  
**Progress:** 82% Complete  
**Next:** Day 20 - Component Integration (ml-detector, sniffer, firewall)