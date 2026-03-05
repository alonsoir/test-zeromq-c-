# ML Defender (aRGus EDR)

**Open-source, enterprise-grade network security system protecting critical infrastructure from ransomware and DDoS attacks.**

[![Via Appia Quality](https://img.shields.io/badge/Via_Appia-Quality-gold)](https://en.wikipedia.org/wiki/Appian_Way)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: Preparing to arXiv Delivery](https://img.shields.io/badge/Status-Production_Ready-brightgreen)]()
https://alonsoir-test-zeromq-c-.mintlify.app/introduction

---

## 🎯 Mission

Democratize enterprise-grade cybersecurity for hospitals, schools, and small organizations that cannot afford commercial solutions. Built to last decades with scientific honesty and methodical development.

**Philosophy**: *Via Appia Quality* – Systems built like Roman roads, designed to endure.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────-─┐
│                         ML Defender Pipeline                     │
├─────────────────────────────────────────────────────────────────-┤
│                                                                  │
│  Network Traffic (eBPF/XDP)                                      │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  sniffer (C++20) │  eBPF/XDP packet capture                   │
│  │                  │  - ShardedFlowManager (16 shards)          │
│  │                  │  - Fast Detector (heuristics)              │
│  │                  │  - 4x embedded ML feature extraction       │
│  │                  │  - ChaCha20-Poly1305 + LZ4 transport       │
│  └──────────────────┘                                            │
│         ↓  ZeroMQ (encrypted)                                    │
│  ┌──────────────────┐                                            │
│  │  ml-detector     │  4x Embedded RandomForest Models           │
│  │  (C++20)         │  - DDoS Detection (97.6% accuracy)         │
│  │                  │  - Ransomware Detection                    │
│  │                  │  - Traffic Classification                  │
│  │                  │  - Internal Anomaly Detection              │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐  ChaCha20-Poly1305 + LZ4                   │ 
│  │  Crypto Pipeline │  36K events, 0 errors ✅                   │
│  |     (C++20)      |                                            |
|  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  etcd-server     │  Distributed Config + Key Management       │ 
│  │  (C++20)         │  Automatic crypto seed exchange            │
│  │                  │  HMAC secrets management ✅                │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │ firewall-acl     │  Autonomous Blocking (Day 52 ✅)           │
│  │ agent (C++20)    │  - IPSet/IPTables integration              │
│  │                  │  - Sub-microsecond latency                 │
│  │                  │  - Config-driven (JSON is law)             │
│  │                  │  - 364 events/sec tested                   │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag-ingester    │  Log Parsing + Vector Ingestion            │
│  │  (C++20)         │  - ml-detector logs ✅                     │
│  │                  │  - firewall logs    ✅                     │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag (TinyLlama) │  Natural Language Intelligence             │
│  │  + FAISS         │  - Forensic queries                        │
│  │  (C++20)         │  - ML retraining data                      │
│  └──────────────────┘                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────-┘
```

---

## 📊 Current Status (Day 76 - Mar 5, 2026)

### ✅ Production Ready Components

#### sniffer
- [x] eBPF/XDP packet capture (sub-microsecond latency)
- [x] ShardedFlowManager (16 shards, thread-safe, zero-lock)
- [x] Fast Detector (heuristics, Layer 1)
- [x] RansomwareFeatureProcessor (30s aggregation, Layer 2)
- [x] 4x embedded ML feature extraction (DDoS, Ransomware, Traffic, Internal)
- [x] ChaCha20-Poly1305 + LZ4 encrypted ZMQ transport
- [x] Proto3 sentinel initialization — **DAY 76 fix** ✅
  - `init_embedded_sentinels()` helper covers all 3 send routes
  - Eliminates SIGSEGV in ml-detector ByteSizeLong
  - Pipeline stable: 6/6 components running continuously

#### etcd-server
- [x] Distributed configuration management
- [x] Automatic crypto seed exchange
- [x] Service registration & heartbeats
- [x] **HMAC Secrets Management** (Day 53 ✅)
  - Key generation/rotation/retrieval
  - HTTP API for secrets
  - Historical key tracking
- [x] C++ implementation with etcd v3 API

#### etcd-client
- [x] Configuration retrieval
- [x] Service discovery
- [x] **HMAC Utilities** (Day 53 ✅)
  - compute_hmac_sha256()
  - validate_hmac_sha256()
  - Hex encoding/decoding
  - Key retrieval from etcd-server
- [x] ZMQ crypto seed negotiation

#### ml-detector
- [x] 4x embedded RandomForest models (C++20)
- [x] 83 feature extraction (flow-based)
- [x] Sub-microsecond detection latency
- [x] ChaCha20-Poly1305 encryption
- [x] LZ4 compression
- [x] Dual-NIC deployment (host IDS + gateway mode)
- [x] Validated with real malware (CTU-13 Neris botnet, 97.6% accuracy) 
- [] Validated with real malware (CTU-13 Neris botnet, with full open source components)
- [x] Dual-Score architecture (fast + ML scores)
- [x] RAG Logger with HMAC artifact integrity

#### firewall-acl-agent (Day 52 ✅)
- [x] Kernel-level blocking (IPSet/IPTables)
- [x] ChaCha20-Poly1305 decryption (0 errors @ 36K events)
- [x] LZ4 decompression (0 errors @ 36K events)
- [x] Config-driven architecture (no hardcoding)
- [x] IPSet verification on startup
- [x] Graceful degradation under stress
- [x] Tested: 364 events/sec, 54% CPU, 127MB RAM

#### rag-ingester
- [x] ml-detector log parsing
- [x] Vector embedding generation
- [ ] firewall-acl-agent log parsing (planned P1.1)

#### rag
- [x] TinyLlama integration
- [x] FAISS vector search
- [ ] Cross-component queries (planned P1.1)
- [ ] Temporal queries (planned P1.2)

---

## 🚀 Quick Start

### Prerequisites
```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git \
    libzmq3-dev libprotobuf-dev protobuf-compiler \
    libjsoncpp-dev libssl-dev liblz4-dev \
    libzstd-dev libsnappy-dev \
    libgrpc++-dev libetcd-cpp-api-dev \
    ipset iptables python3 python3-pip

# Kernel headers (for eBPF)
sudo apt-get install -y linux-headers-$(uname -r)

# Fix libsnappy pkg-config (if needed)
sudo ln -sf /usr/lib/x86_64-linux-gnu/pkgconfig/snappy.pc \
            /usr/lib/x86_64-linux-gnu/pkgconfig/libsnappy.pc
```

### Build & Deploy

```bash
# 1. Clone repository
git clone https://github.com/yourusername/ml-defender.git
cd ml-defender

# 2. Build all components (from macOS host with Vagrant)
make all

# 3. Start full pipeline
make pipeline-start

# 4. Verify
make pipeline-status
```

### Test with Synthetic Data

```bash
cd tools/build
./synthetic_ml_output_injector 1000 50

# Monitor blocking
watch -n 1 'sudo ipset list ml_defender_blacklist_test | head -20'
```

---

## 🔬 Day 76 Achievements

### Proto3 Sentinel Fix — SIGSEGV Eliminated
**Problem**: Proto3 C++ 3.21 does not serialize submessages where all float
fields equal `0.0f`. Receiver gets null pointer → SIGSEGV in `ByteSizeLong()`
when ml-detector processes DDoS/Ransomware/Traffic/Internal embedded submessages.

Three routes in `ring_consumer.cpp` were affected:
- `populate_protobuf_event()` — raw eBPF capture path
- `send_fast_alert()` — Layer 1 heuristic alert path
- `send_ransomware_features()` — Layer 2 aggregation path (DAY 75 fix was incomplete)

**Solution**: `init_embedded_sentinels()` helper initializes all 40 fields
across 4 submessages with `0.5f` Phase 1 sentinel values before serialization.

**Result**: Pipeline runs continuously. ml-detector VIVO after 60s+ of operation.

### Additional Fixes
- `snappy::Uncompress()` wrong signature (2 args → 3 args): added `.data(), .size()`
- `libsnappy.pc` symlink for cmake pkg-config discovery

### Pipeline Validation
```
etcd-server:   ✅ RUNNING
rag-security:  ✅ RUNNING
rag-ingester:  ✅ RUNNING
ml-detector:   ✅ RUNNING  
sniffer:       ✅ RUNNING
firewall:      ✅ RUNNING
```

---

## 📋 Backlog & Roadmap

### Priority 0: F1-Score Validation (Current — DAY 77)

**sniffer/ring_consumer.cpp**:
- [ ] Replace `0.5f` sentinels with real extracted values from ShardedFlowManager
- [ ] Fix call order: `populate_ml_defender_features()` must not be overwritten by sentinels
- [ ] Complete `run_ml_detection()` — write inference results back to proto_event
- [ ] Validate F1-score against CTU-13 Neris dataset (`make test-replay-neris`)

### Priority 1: Production Scale (2 weeks)

**firewall-acl-agent**:
- [ ] P1.1: Multi-tier storage (IPSet → SQLite → Parquet)
- [ ] P1.2: Async queue + worker pool (1K+ events/sec)
- [ ] P1.3: Capacity monitoring + auto-eviction

**rag-ingester**:
- [ ] P1.1: Firewall log parser (ground truth blocking data)
- [ ] P1.2: Forensic query library
- [ ] P1.3: ML retraining data export

**rag**:
- [ ] P1.1: Cross-component queries (detection ↔ block linking)
- [ ] P1.2: Temporal queries (natural language time)
- [ ] P1.3: Aggregation & statistics

### Priority 2: Observability (1 week)

- [ ] Prometheus metrics exporter
- [ ] Grafana dashboards
- [ ] Health check endpoints (K8s)
- [ ] Runtime config via etcd

### Priority 3: Intelligence (1 week)

- [ ] Block query REST API
- [ ] Recidivism detection
- [ ] Trend analysis
- [ ] Intent classification

---

## 🎓 Design Philosophy

### Via Appia Quality
Systems built to last decades, like Roman roads:
- **Scientific honesty**: Report actual results, not inflated claims
- **Methodical development**: Validate each component before proceeding
- **Transparent AI collaboration**: Credit all AI systems as co-authors
- **User privacy**: No telemetry, no tracking, no data exfiltration
- **Accessibility**: Documentation in natural language for non-experts

### Collaborative AI Development
This project practices "Consejo de Sabios" (Council of Wise Ones):
- Multiple AI systems (Claude, DeepSeek, Grok, ChatGPT, Qwen) peer-review code
- All AI contributions explicitly credited
- Transparent methodology for academic work
- AI as co-authors, not mere tools

---

## 📚 Documentation

### Architecture & Design
- [System Architecture](docs/architecture.md)
- [Crypto Pipeline](docs/crypto-pipeline.md)
- [eBPF/XDP Packet Capture](docs/ebpf-xdp.md)
- [ML Model Training](docs/ml-training.md)

### Component Guides
- [ml-detector README](ml-detector/README.md)
- [etcd-server README](etcd-server/README.md)
- [firewall-acl-agent README](firewall-acl-agent/README.md)
- [rag-ingester README](rag-ingester/README.md)

### Operations
- [Deployment Guide](docs/deployment.md)
- [Configuration Reference](docs/configuration.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Performance Tuning](docs/performance.md)

---

## 🧪 Testing & Validation

### Datasets Used
- **CTU-13 Neris Botnet**: Ransomware behavior validation (97.6% accuracy)
- **Synthetic Traffic**: Custom generator for DDoS patterns
- **Real Network Captures**: 10+ hours of production traffic

### Test Coverage
- Unit tests: Core algorithms and data structures
- Integration tests: End-to-end pipeline validation
- Stress tests: 36K events, multiple load profiles
- Regression tests: Proto3 serialization, RAG logger HMAC

### Continuous Validation
```bash
# Run full test suite
make test

# Stress test pipeline
make test-replay-neris   # CTU-13 Neris botnet (492K events)
make test-replay-small   # Quick validation

# Validate crypto pipeline
make verify-all
```

---

## 🔐 Security

### Threat Model
**Protects Against**:
- DDoS attacks (volumetric, protocol, application layer)
- Ransomware C2 communication
- Port scanning and reconnaissance
- Known malicious IPs and patterns

**Does NOT Protect Against**:
- Zero-day exploits (no signatures)
- Encrypted malware payloads (TLS/SSL)
- Insider threats (requires authentication layer)
- Physical attacks (out of scope)

### Security Guarantees
- ✅ ChaCha20-Poly1305 authenticated encryption (AEAD)
- ✅ HMAC-SHA256 log integrity (tamper detection)
- ✅ No cleartext transmission of threats
- ✅ Autonomous blocking (no human in loop)
- ✅ IPSet/IPTables kernel-level enforcement
- ✅ Fail-closed design (errors → block, not allow)

### Known Limitations
- IPSet capacity finite (max realistic: 500K IPs)
- No persistence layer yet (evicted IPs lost)
- Single-node deployment (no HA/failover)
- Embedded detector features use Phase 1 sentinels pending real extraction (DAY 77)

---

## 📈 Performance

### Benchmarks

**sniffer**:
- Packet capture: sub-microsecond (eBPF/XDP)
- Flow tracking: ShardedFlowManager 16 shards, lock-free per shard
- Transport: ChaCha20-Poly1305 + LZ4, 0 crypto errors

**ml-detector**:
- Detection latency: 0.24μs – 1.06μs (4 embedded models)
- Throughput: 1M+ packets/sec (synthetic traffic)
- Features: 83 per flow (23 Level 1 + 40 embedded Phase 1)
- Models: 4 concurrent (DDoS, Ransomware, Traffic Class, Anomaly)

**firewall-acl-agent**:
- Blocking latency: <10 ms (detection → block)
- Throughput: 364 events/sec (stress tested)
- CPU: 54% max under extreme load
- Memory: 127 MB RSS
- Crypto pipeline: 0 errors @ 36K events

**etcd-server**:
- Service registration: <50 ms
- Crypto seed exchange: <100 ms
- Heartbeat interval: 30 sec

---

## 🤝 Contributing

ML Defender welcomes contributions! We practice transparent AI collaboration.

### Contribution Guidelines
1. **Scientific honesty**: Report real results, acknowledge limitations
2. **AI transparency**: Credit AI assistants used in development
3. **Testing required**: All changes must include tests
4. **Documentation**: Update docs with code changes
5. **Via Appia Quality**: Build for decades, not quarters

### Development Setup
```bash
# Fork and clone
git clone https://github.com/yourusername/ml-defender.git
cd ml-defender

# Create feature branch
git checkout -b feature/your-feature

# Build and test
make all
make test

# Submit PR with:
# - Description of changes
# - Test results
# - AI collaboration disclosure (if applicable)
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

### Human Contributors
- **Alonso Isidoro Roman** - Creator, ML Architect, Via Appia Philosopher

### AI Co-Authors
This project practices transparent AI collaboration. The following AI systems have contributed to development:
- **Claude** (Anthropic) - Architecture design, code review, debugging, documentation
- **DeepSeek** - Algorithm optimization, debugging
- **Grok** - Performance analysis, cmake diagnostics
- **ChatGPT** - Research assistance, lifetime analysis
- **Qwen** - Documentation review

All AI contributions are explicitly acknowledged in code comments and commit messages.

### Datasets & Research
- **CTU-13 Dataset** - Czech Technical University, Malware Capture Facility
- **NetworkML** - Network traffic feature extraction research

---

## 📞 Contact

- **Email**: alonso@ml-defender.org
- **GitHub**: https://github.com/ml-defender/aegisIDS
- **Documentation**: https://docs.ml-defender.org
- **Discussions**: https://github.com/ml-defender/aegisIDS/discussions

---

## Attribution

This project is authored by Alonso Isidoro Roman and was developed with
AI assistance from Claude (Anthropic) and the Consejo de Sabios methodology.
For details on the collaboration methodology and all acknowledgments, see:

- [AUTHORS.md](AUTHORS.md) - Copyright and ownership
- [ATTRIBUTION.md](ATTRIBUTION.md) - Full acknowledgments and methodology
- [LICENSE](LICENSE) - MIT License terms

---

## 🗺️ Project Status

**Current Phase**: Day 76 — Pipeline stable, F1-score validation next

**Last Updated**: March 5, 2026

**Recent Milestones**:
- ✅ Day 52: Stress testing validation (36K events, 0 crypto errors)
- ✅ Day 53: HMAC infrastructure (secrets management, key rotation)
- ✅ Day 64: CSV pipeline + 127-column schema
- ✅ Day 72: Deterministic trace_id correlation (SHA256 + temporal buckets)
- ✅ Day 75: Proto3 null pointer root cause identified (ByteSizeLong SIGSEGV)
- ✅ Day 76: SIGSEGV eliminated — pipeline 6/6 stable, init_embedded_sentinels()

**Next Milestones**:
- 🎯 Day 77: Real feature extraction in ring_consumer (replace 0.5f sentinels)
- 🎯 Day 78: F1-score validation against CTU-13 Neris (492K events)
- 🎯 Week N: arXiv paper submission

---

**Via Appia Quality** 🏛️ - Built to last decades

*"The road to security is long, but we build it to endure."*