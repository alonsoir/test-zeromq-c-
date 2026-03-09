# ML Defender (aRGus EDR)

**Open-source, enterprise-grade network security system protecting critical infrastructure from ransomware and DDoS attacks.**

[![Via Appia Quality](https://img.shields.io/badge/Via_Appia-Quality-gold)](https://en.wikipedia.org/wiki/Appian_Way)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: F1=0.9934 Validated](https://img.shields.io/badge/Status-F1%3D0.9934_Validated-brightgreen)]()
https://alonsoir-test-zeromq-c-.mintlify.app/introduction

---

## 🎯 Mission

Democratize enterprise-grade cybersecurity for hospitals, schools, and small organizations that cannot afford commercial solutions. Built to last decades with scientific honesty and methodical development.

**Philosophy**: *Via Appia Quality* – Systems built like Roman roads, designed to endure.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         ML Defender Pipeline                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Network Traffic (eBPF/XDP)                                      │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  sniffer (C++20) │  eBPF/XDP packet capture                   │
│  │                  │  - ShardedFlowManager (16 shards)          │
│  │                  │  - Fast Detector (heuristics)              │
│  │                  │  - 4x embedded ML feature extraction       │
│  │                  │  - ChaCha20-Poly1305 + LZ4 transport       │
│  │                  │  - Thresholds desde JSON ✅ DAY 80         │
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
│  │                  │  - Config-driven (JSON is law ✅)          │
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
└──────────────────────────────────────────────────────────────────┘
```

---

## 📊 Current Status (Day 80 — March 9, 2026)

### ✅ Validated Results

| Metric | Value |
|---|---|
| **F1-score (CTU-13 Neris)** | **0.9934** |
| Recall | 1.0000 (zero missed attacks) |
| Precision | 0.9869 |
| Dataset | 492K packets, 19,135 flows |
| Features active | 28/40 real (11 sentinel, 1 semantic) |
| Pipeline components | 6/6 RUNNING |
| Thresholds | From JSON ✅ (Phase1-Day4-CRITICAL closed) |

### Threshold Configuration (sniffer.json)

```json
"ml_defender": {
  "thresholds": {
    "ddos": 0.85,
    "ransomware": 0.90,
    "traffic": 0.80,
    "internal": 0.85
  }
}
```

No recompilation needed to tune precision/recall trade-off.

### DAY 79 vs DAY 80 Comparison

| Metric | DAY 79 (hardcoded) | DAY 80 (JSON) |
|---|---|---|
| F1 | 0.9921 | **0.9934** ✅ |
| Precision | 0.9844 | **0.9869** ✅ |
| Recall | 1.0000 | **1.0000** ✅ |
| False Negatives | 0 | **0** ✅ |
| False Positives (abs) | 106 | **79** ✅ |

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
- [x] **Thresholds from JSON — DAY 80 fix ✅** (Phase1-Day4-CRITICAL closed)

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
- [x] **F1=0.9934 validated — CTU-13 Neris botnet (DAY 80)** ✅

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

### Tune Thresholds (no recompilation needed)

Edit `sniffer/build-debug/config/sniffer.json`:

```json
"ml_defender": {
  "thresholds": {
    "ddos": 0.85,
    "ransomware": 0.90,
    "traffic": 0.80,
    "internal": 0.85
  }
}
```

Restart sniffer to apply. Startup log confirms loaded values:
```
[ML Defender] Thresholds (JSON): DDoS=0.85 Ransomware=0.9 Traffic=0.8 Internal=0.85
```

---

## 🔬 Engineering Decisions

### Sentinel Value Taxonomy (DAY 79)

Three categories of special values in ML feature extraction:

1. **Domain-valid sentinel** (`-9999.0f`) — mathematically unreachable (split
   domain [0.0, 5.1]). Deterministic, auditable routing. Always `left_child`.
2. **Semantic value** (e.g. `0.5f` for TCP established-not-closed) — valid domain
   value with explicit meaning. Must be preserved with protective comments.
3. **Placeholder within domain** — strictly worse than category 1. Introduces
   non-deterministic spurious variance in the RandomForest ensemble.

Full analysis: [`docs/engineering_decisions/DAY79_sentinel_analysis.md`](docs/engineering_decisions/DAY79_sentinel_analysis_CLAUDE.md)

### JSON is the LAW (DAY 80)

All configuration values — including ML thresholds — must come from JSON.
No hardcoded constants in production code. Fallbacks must be explicit, logged,
and never silent.

Four-layer bug resolved: literals in `ring_consumer.cpp` → missing mapping in
`main.cpp` → missing struct in `StrictSnifferConfig` → struct layout mismatch
causing NaN. Fix: explicit field-by-field mapping between `StrictSnifferConfig`
and `SnifferConfig`.

### Standardized Logging (DAY 79)

All pipeline components write to `/vagrant/logs/lab/`:

```
/vagrant/logs/lab/
├── etcd-server.log
├── rag-security.log
├── rag-ingester.log
├── ml-detector.log
├── detector.log      ← spdlog internal (ADR pending: unify)
├── firewall-agent.log
└── sniffer.log
```

New Makefile targets:
- `make logs-all` — tail -f all logs simultaneously
- `make logs-lab-clean` — rotate logs to archive

---

## 📋 Roadmap

### Immediate (DAY 81)
- Inspect `FlowStatistics` → implement `tcp_udp_ratio`, `protocol_variety`
- Clean F1 comparison (same replay, both threshold configs)
- Balanced dataset validation (P0 for paper — CTU-13 Neris is 98% malicious)

### Short Term (DAY 82-85)
- CSV Pipeline E2E validation with real traffic
- Fix 2 pre-existing trace_id test failures (DAY 72)
- Unify ml-detector dual logs (ADR)
- arXiv paper preparation

### Medium Term
- Multi-tier storage (IPSet → SQLite → Parquet)
- Async queue + worker pool (1K+ events/sec)
- Prometheus metrics + Grafana dashboards

### Enterprise
- Federated Threat Intelligence (opt-in, local anonymization)
- Attack Graph Generation (GraphML + STIX 2.1)
- P2P Seed Distribution (eliminate etcd as crypto authority)
- Hot-reload configuration (no downtime threshold tuning)

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
- **CTU-13 Neris Botnet**: F1=0.9934, Recall=1.0000 (DAY 80) ✅
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
- ✅ JSON-driven thresholds (no hardcoded security parameters)

### Known Limitations
- IPSet capacity finite (max realistic: 500K IPs)
- No persistence layer yet (evicted IPs lost on restart)
- Single-node deployment (no HA/failover)
- 11/40 ML features use sentinel values (Phase 2 pending)
- High FPR on heavily imbalanced datasets — documented honestly
- Balanced dataset validation pending (CTU-13 Neris is 98% malicious traffic)

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
- F1: 0.9934 on CTU-13 Neris (DAY 80)

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

**Current Phase**: Day 80 — F1=0.9934 validated, JSON thresholds live

**Last Updated**: March 9, 2026

**Recent Milestones**:
- ✅ Day 52: Stress testing (36K events, 0 crypto errors)
- ✅ Day 53: HMAC infrastructure (secrets management, key rotation)
- ✅ Day 64: CSV pipeline + 127-column schema
- ✅ Day 72: Deterministic trace_id correlation (SHA256)
- ✅ Day 76: SIGSEGV eliminated — pipeline 6/6 stable
- ✅ Day 79: F1=0.9921 — sentinel fix + logging standard
- ✅ Day 80: F1=0.9934 — **Phase1-Day4-CRITICAL closed — JSON is the LAW** 🦅

**Next Milestones**:
- 🎯 Day 81: FlowStatistics inspection + balanced dataset validation
- 🎯 Week N+1: arXiv paper submission

---

**Via Appia Quality** 🏛️ — Built to last decades

*"The road to security is long, but we build it to endure."*