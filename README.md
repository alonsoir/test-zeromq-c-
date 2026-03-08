# ML Defender (aRGus EDR)

**Open-source, enterprise-grade network security system protecting critical infrastructure from ransomware and DDoS attacks.**

[![Via Appia Quality](https://img.shields.io/badge/Via_Appia-Quality-gold)](https://en.wikipedia.org/wiki/Appian_Way)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: F1=0.9921 Validated](https://img.shields.io/badge/Status-F1%3D0.9921_Validated-brightgreen)]()
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
│  │  (C++20)         │  - DDoS Detection                          │
│  │                  │  - Ransomware Detection                    │
│  │                  │  - Traffic Classification                  │
│  │                  │  - Internal Anomaly Detection              │
│  └──────────────────┘                                            │
│         ↓  ChaCha20-Poly1305 + LZ4                               │
│  ┌──────────────────┐                                            │
│  │  etcd-server     │  Distributed Config + Key Management       │
│  │  (C++20)         │  - Automatic crypto seed exchange          │
│  │                  │  - HMAC secrets management ✅              │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │ firewall-acl     │  Autonomous Blocking                       │
│  │ agent (C++20)    │  - IPSet/IPTables integration              │
│  │                  │  - Sub-microsecond latency                 │
│  │                  │  - Config-driven (JSON is law)             │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag-ingester    │  Log Parsing + Vector Ingestion            │
│  │  (C++20)         │  - ml-detector CSV logs ✅                 │
│  │                  │  - firewall logs ✅                        │
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

## 📊 Current Status (Day 79 — March 8, 2026)

### ✅ Validated Results

| Metric | Value |
|---|---|
| **F1-score (CTU-13 Neris)** | **0.9921** |
| Recall | 1.0000 (zero missed attacks) |
| Precision | 0.9844 |
| Dataset | 492K packets, 19,135 flows, 6,810 ML events |
| Features active | 28/40 real (21 sentinel, 1 semantic) |
| Pipeline components | 6/6 RUNNING |

### ✅ Production Ready Components

#### sniffer
- [x] eBPF/XDP packet capture (sub-microsecond latency)
- [x] ShardedFlowManager (16 shards, thread-safe, zero-lock)
- [x] Fast Detector (heuristics, Layer 1)
- [x] RansomwareFeatureProcessor (30s aggregation, Layer 2)
- [x] 4x embedded ML feature extraction (28/40 real features)
- [x] ChaCha20-Poly1305 + LZ4 encrypted ZMQ transport
- [x] Proto3 sentinel initialization — DAY 76 fix ✅
- [x] Placeholder `0.5f` → `MISSING_FEATURE_SENTINEL` — DAY 79 fix ✅

#### etcd-server
- [x] Distributed configuration management
- [x] Automatic crypto seed exchange
- [x] Service registration & heartbeats
- [x] HMAC Secrets Management (Day 53 ✅)

#### ml-detector
- [x] 4x embedded RandomForest models (C++20)
- [x] Dual-Score architecture (fast + ML scores)
- [x] ChaCha20-Poly1305 + LZ4 pipeline
- [x] RAG Logger with HMAC artifact integrity
- [x] **F1=0.9921 validated — CTU-13 Neris botnet (DAY 79)** ✅

#### firewall-acl-agent
- [x] Kernel-level blocking (IPSet/IPTables)
- [x] ChaCha20-Poly1305 decryption (0 errors @ 36K events)
- [x] Config-driven architecture (no hardcoding)
- [x] Tested: 364 events/sec, 54% CPU, 127MB RAM

#### rag-ingester
- [x] ml-detector CSV log parsing + HMAC integrity
- [x] Vector embedding generation (ONNX + FAISS)
- [ ] firewall-acl-agent log parsing (planned)

#### rag-security
- [x] TinyLlama integration
- [x] FAISS vector search
- [ ] Cross-component queries (planned)

---

## 🚀 Quick Start

### Prerequisites
```bash
# Vagrant + VirtualBox required (multi-VM setup)
vagrant --version   # 2.3+
vboxmanage --version  # 7.x
```

### Build & Deploy

```bash
# 1. Clone repository
git clone https://github.com/yourusername/ml-defender.git
cd ml-defender

# 2. Start full environment (defender VM)
vagrant up defender

# 3. Build all components
make all

# 4. Start full pipeline
make pipeline-start

# 5. Verify
make pipeline-status
```

### Run CTU-13 Neris Validation

```bash
# Requires client VM
vagrant up client

# Start pipeline and replay
make pipeline-start && sleep 15
make test-replay-neris

# Monitor all logs simultaneously
make logs-all
```

---

## 🔬 DAY 79 Achievements

### Sentinel Value Fix — F1 Baseline Established

**Problem identified**: 8 feature extractor functions returned `0.5f` as
"neutral" placeholder for unimplemented Phase 2 features. Since `0.5f` falls
within the RandomForest split domain [0.0, 5.1], these values could activate
different tree branches non-deterministically, introducing spurious variance
in the ensemble.

**Key distinction documented**:
- `MISSING_FEATURE_SENTINEL = -9999.0f` → 3 orders of magnitude outside split
  domain → deterministic, auditable routing (always `left_child`) ✅
- `0.5f` placeholder → inside split domain → non-deterministic spurious variance ❌
- `0.5f` semantic (TCP established-not-closed in `flow_completion_rate`) → valid ✅

**Fix**: 8 placeholders → `MISSING_FEATURE_SENTINEL`. 2 semantic values protected
with explicit comments.

**Result**: F1 = 0.9921, Recall = 1.0000 (zero missed attacks on CTU-13 Neris).

Full analysis: [`docs/engineering_decisions/DAY79_sentinel_analysis.md`](docs/engineering_decisions/DAY79_sentinel_analysis.md)

### Standardized Logging — All 6 Components

All pipeline components now write to `/vagrant/logs/lab/`:

```
/vagrant/logs/lab/
├── etcd-server.log
├── rag-security.log
├── rag-ingester.log
├── ml-detector.log
├── firewall-agent.log
└── sniffer.log
```

New Makefile targets:
- `make logs-all` — tail -f all 6 logs simultaneously
- `make logs-lab-clean` — rotate logs to archive

---

## 📋 Backlog & Roadmap

### Priority 0: Thresholds from JSON (DAY 80)

**ring_consumer.cpp** has hardcoded thresholds:
- DDoS: `0.7f`, Ransomware: `0.75f`, Traffic: `0.7f`, Internal: `0.00000000065f`

Must be read from `ml_detector_config.json`. "JSON is the law."

### Priority 1: Remaining Features (DAY 80-81)

Features still pending real extraction:
- `tcp_udp_ratio` — requires protocol field in FlowStatistics
- `flow_duration_std` / `connection_duration_std` — requires multi-flow aggregation
- `protocol_variety` — requires multi-flow aggregation

### Priority 2: Production Scale

- Multi-tier storage (IPSet → SQLite → Parquet)
- Async queue + worker pool (1K+ events/sec)
- Prometheus metrics exporter
- Grafana dashboards

### Priority 3: Intelligence

- Cross-component RAG queries (detection ↔ block linking)
- Temporal forensic queries
- Recidivism detection

---

## 🎓 Design Philosophy

### Via Appia Quality
Systems built to last decades, like Roman roads:
- **Scientific honesty**: Report actual results, acknowledge limitations
- **Methodical development**: Validate each component before proceeding
- **Transparent AI collaboration**: Credit all AI systems as co-authors
- **User privacy**: No telemetry, no tracking, no data exfiltration
- **Accessibility**: Documentation in natural language for non-experts

### Collaborative AI Development — Consejo de Sabios
This project practices multi-agent peer review:
- Multiple AI systems (Claude, DeepSeek, Grok, ChatGPT, Qwen) review code
- All AI contributions explicitly credited
- Transparent methodology documented for academic work
- AI as co-authors, not mere tools

---

## 🧪 Testing & Validation

### Datasets Used
- **CTU-13 Neris Botnet**: F1=0.9921, Recall=1.0000 (DAY 79) ✅
- **Synthetic Traffic**: Custom generator for DDoS patterns
- **Real Network Captures**: 10+ hours of production traffic

### Test Commands
```bash
# Full pipeline validation with CTU-13 Neris
vagrant up client
make pipeline-start && sleep 15
make test-replay-neris

# Quick validation
make test-replay-small

# All logs live
make logs-all

# Crypto pipeline
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
- ✅ Autonomous blocking (no human in loop)
- ✅ IPSet/IPTables kernel-level enforcement
- ✅ Fail-closed design (errors → block, not allow)

### Known Limitations
- IPSet capacity finite (max realistic: 500K IPs)
- No persistence layer yet (evicted IPs lost on restart)
- Single-node deployment (no HA/failover)
- 12/40 ML features use sentinel values (Phase 2 pending)
- Thresholds hardcoded pending JSON migration (DAY 80)
- High FPR on heavily imbalanced datasets (documented in DAY79_sentinel_analysis.md)

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
- F1: 0.9921 on CTU-13 Neris (DAY 79)

**firewall-acl-agent**:
- Blocking latency: <10 ms (detection → block)
- Throughput: 364 events/sec (stress tested)
- CPU: 54% max under extreme load
- Memory: 127 MB RSS
- Crypto pipeline: 0 errors @ 36K events

---

## 🤝 Contributing

ML Defender welcomes contributions. We practice transparent AI collaboration.

### Contribution Guidelines
1. **Scientific honesty**: Report real results, acknowledge limitations
2. **AI transparency**: Credit AI assistants used in development
3. **Testing required**: All changes must include tests
4. **Documentation**: Update docs with code changes
5. **Via Appia Quality**: Build for decades, not quarters

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

### Human Contributors
- **Alonso Isidoro Roman** — Creator, ML Architect, Via Appia Philosopher

### AI Co-Authors (Consejo de Sabios)
- **Claude** (Anthropic) — Architecture design, code review, debugging, documentation
- **DeepSeek** — Algorithm optimization, debugging
- **Grok** — Performance analysis, feature analysis
- **ChatGPT** — Research assistance, academic references
- **Qwen** — Documentation review

### Datasets & Research
- **CTU-13 Dataset** — Czech Technical University, Malware Capture Facility
- **Kitsune** — Mirsky et al., NDSS 2018 (network-only host behavior inference)
- **CIC-IDS2017** — University of New Brunswick (feature reference)

---

## 📞 Contact

- **GitHub**: https://github.com/ml-defender/aegisIDS
- **Documentation**: https://alonsoir-test-zeromq-c-.mintlify.app/introduction

---

## Attribution

Authored by Alonso Isidoro Roman, developed with the Consejo de Sabios methodology.
See [AUTHORS.md](AUTHORS.md), [ATTRIBUTION.md](ATTRIBUTION.md), [LICENSE](LICENSE).

---

## 🗺️ Project Status

**Current Phase**: Day 79 — F1=0.9921 validated, thresholds JSON migration next

**Last Updated**: March 8, 2026

**Recent Milestones**:
- ✅ Day 52: Stress testing (36K events, 0 crypto errors)
- ✅ Day 53: HMAC infrastructure (secrets management, key rotation)
- ✅ Day 64: CSV pipeline + 127-column schema
- ✅ Day 72: Deterministic trace_id correlation (SHA256)
- ✅ Day 76: SIGSEGV eliminated — pipeline 6/6 stable
- ✅ Day 79: F1=0.9921 — placeholder sentinel fix + logging standard

**Next Milestones**:
- 🎯 Day 80: Thresholds from JSON + remaining features + F1 post-fix
- 🎯 Week N+1: arXiv paper submission

---

**Via Appia Quality** 🏛️ — Built to last decades

*"The road to security is long, but we build it to endure."*