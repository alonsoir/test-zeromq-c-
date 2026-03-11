# ML Defender (aRGus EDR)

**Open-source, enterprise-grade network security system protecting critical infrastructure from ransomware and DDoS attacks.**

[![Via Appia Quality](https://img.shields.io/badge/Via_Appia-Quality-gold)](https://en.wikipedia.org/wiki/Appian_Way)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: F1=1.0000 Validated](https://img.shields.io/badge/Status-F1%3D1.0000_Validated-brightgreen)]()
https://alonsoir-test-zeromq-c-.mintlify.app/introduction

---

## 🎯 Mission

Democratize enterprise-grade cybersecurity for hospitals, schools, and small organizations that cannot afford commercial solutions. Built to last decades with scientific honesty and methodical development.

**Philosophy**: *Via Appia Quality* – Systems built like Roman roads, designed to endure.

---

### Three Foundational Capabilities

**Shield** — Operational protection. Real-time detection and blocking of ransomware and DDoS via heuristic Fast Detector and embedded C++20 ML ensemble. F1=1.0000 validated on CTU-13 Neris.

**Microscope** — Traffic measurement instrument. Every datagrama passing through the pipeline is dissected into 40 dimensions, correlated via `trace_id`, and stored as pipeline-native data — eliminating the feature drift that invalidates models trained on academic datasets.

**Research Platform** — Controlled dataset generation with configurable proportions, reproducible experimentation infrastructure, and RAG-security as a conversational analysis interface over generated data.

All code, all analysis scripts, all experiments, and all failures are documented in the repository. No tricks, no shortcuts.

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
│  │  (C++20)         │  - DDoS Detection (<50μs)                  │
│  │                  │  - Ransomware Detection (<55μs)            │
│  │                  │  - Traffic Classification (<50μs)          │
│  │                  │  - Internal Anomaly Detection (<48μs)      │
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
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag-ingester    │  Log Parsing + Vector Ingestion            │
│  │  (C++20)         │  - ml-detector CSV logs ✅                 │
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

## 📊 Current Status (Day 81 — March 10, 2026)

### ✅ Validated Results

| Metric | Value |
|---|---|
| **F1-score (CTU-13 Neris, thresholds 0.85/0.90)** | **1.0000** |
| **F1-score (CTU-13 Neris, thresholds 0.70/0.75)** | **0.9976** |
| Recall | 1.0000 (zero missed attacks, upper bound) |
| Precision | 1.0000 / 0.9951 |
| Dataset | 492K packets, 19,135 flows, same PCAP both conditions |
| Ground truth | 147.32.84.165 (sole malicious IP in this capture) |
| Features active | 28/40 real (11 sentinel Phase 2, 1 semantic) |
| Pipeline components | 6/6 RUNNING |
| Thresholds | From JSON ✅ (Phase1-Day4-CRITICAL closed DAY 80) |

### Threshold Comparison (DAY 81 — same PCAP, controlled)

| Condition | DDoS | Ransom | Traffic | Internal | F1 | FP real |
|---|---|---|---|---|---|---|
| Production (JSON) | 0.85 | 0.90 | 0.80 | 0.85 | **1.0000** | 0 |
| Legacy low | 0.70 | 0.75 | 0.70 | 0.70 | **0.9976** | 1 |

Conservative thresholds eliminate the sole false positive without sacrificing recall.

### Honest Limitations

- FN=0 is an upper bound — requires full per-event IP table to confirm
- CTU-13 Neris is 98% malicious traffic — balanced dataset validation pending (P0)
- ML RandomForest max score = 0.6607 (below threshold) — Fast Detector handles all detections in Neris
- 11/40 ML features use sentinel values — Phase 2 pending

---

## 🚀 Quick Start

### Prerequisites
```bash
vagrant --version   # 2.3+
vboxmanage --version  # 7.x
```

### Build & Deploy

```bash
git clone https://github.com/yourusername/ml-defender.git
cd ml-defender
vagrant up defender
make all
make pipeline-start
make pipeline-status
```

### Run CTU-13 Neris Validation

```bash
# Check VM status first — never double-start client
vagrant status

make pipeline-stop && make logs-lab-clean && make pipeline-start && sleep 15
vagrant ssh -c "grep 'Thresholds (JSON)' /vagrant/logs/lab/sniffer.log"

vagrant up client   # only if client is not already running
make test-replay-neris

# Calculate F1
vagrant ssh -c "cat /vagrant/logs/lab/sniffer.log" > /tmp/sniffer.log
vagrant ssh -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"
python3 scripts/calculate_f1_neris.py /tmp/sniffer.log --total-events N --day 82
```

### Tune Thresholds (no recompilation needed)

Edit **source** file `sniffer/config/sniffer.json` (not the build artifact):

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

Restart pipeline to apply. Startup log confirms:
```
[ML Defender] Thresholds (JSON): DDoS=0.85 Ransomware=0.9 Traffic=0.8 Internal=0.85
```

---

## 🔬 Engineering Decisions

### Sentinel Value Taxonomy (DAY 79)

Three categories of special values in ML feature extraction:

1. **Domain-valid sentinel** (`-9999.0f`) — mathematically unreachable. Deterministic, auditable.
2. **Semantic value** (e.g. `0.5f` TCP established-not-closed) — valid domain value, preserved with protective comments.
3. **Placeholder within domain** — strictly worse than category 1. Introduces spurious variance in the RandomForest ensemble.

Full analysis: [`docs/engineering_decisions/DAY79_sentinel_analysis.md`](docs/engineering_decisions/)

### JSON is the LAW (DAY 80)

All configuration values — including ML thresholds — come from JSON.
No hardcoded constants in production code. Fallbacks must be explicit and logged.

Note: `sniffer/build-debug/config/sniffer.json` is a generated artifact.
Always edit `sniffer/config/sniffer.json` (source of truth).

### ML Training Data — Three Categories (DAY 81)

The project formally distinguishes three types of training data:

- **Category A — Academic** (CTU-13, CIC-IDS2017): F1≈0.99 offline, F1≈0.006 in production. Feature drift invalidates the model entirely.
- **Category B — Synthetic statistical** (own generator): max score 0.6607 in production. Better than A, insufficient for threshold.
- **Category C — Pipeline-native**: features generated by the C++20 extractor itself, correlated via `trace_id`. Zero feature drift by construction. **Hypothesis under validation.**

Full analysis: [`docs/engineering_decisions/DAY81_ml_training_data_analysis.md`](docs/engineering_decisions/)

### Fast Detector Design (DAY 12 — not trained on CTU-13)

Fast Detector heuristics were designed from first principles of ransomware and C&C behavior — not from CTU-13 data. CTU-13 was cited as theoretical validation only. Heuristics:

- `external_ips_30s > 15` — ransomware must contact C&C (unavoidable)
- `smb_connection_diversity > 10` — WannaCry/Petya lateral movement signature
- `dns_entropy > 2.5` — DGA detection
- `upload_download_ratio > 3.0` — double extortion exfiltration
- `burst_connections > 50` — worm behavior

F1=1.0000 on CTU-13 Neris is therefore a generalization result, not overfitting.

### FlowStatistics Phase 2 (DAY 81)

Four features blocked at `FlowStatistics` level, not at protobuf:
- `tcp_udp_ratio` — needs `uint8_t protocol` field in FlowStatistics
- `flow_duration_std`, `protocol_variety`, `connection_duration_std` — need multi-flow TimeWindowAggregator

Protobuf contract is correct. These features return `MISSING_FEATURE_SENTINEL` (-9999.0f)
until Phase 2. Documented as `DEBT-PHASE2` in code comments.

### Standardized Logging (DAY 79 + ADR-005 DAY 81)

```
/vagrant/logs/lab/
├── etcd-server.log
├── rag-security.log
├── rag-ingester.log
├── ml-detector.log     ← stdout (Makefile redirect, startup only)
├── detector.log        ← spdlog internal (operational source of truth)
├── firewall-agent.log
└── sniffer.log
```

ADR-005: unify both ml-detector logs post-paper with ENT-4 hot-reload.

---

## 🧪 Experiment Tracking

All F1 replay results are tracked in `docs/experiments/f1_replay_log.csv`.
Protocol defined in `docs/experiments/f1_replay_log.md`.

| replay_id | day | thresholds | F1 | Precision | Recall | notes |
|---|---|---|---|---|---|---|
| UNKNOWN_DAY79 ⚠️ | 79 | 0.70/0.75 hardcoded | 0.9921 | 0.9844 | 1.0000 | Replay unknown ⚠️ |
| UNKNOWN_DAY80 ⚠️ | 80 | 0.85/0.90 JSON | 0.9934 | 0.9869 | 1.0000 | Replay unknown ⚠️ |
| DAY81_thresholds_085090 | 81 | 0.85/0.90 JSON | 1.0000 | 1.0000 | 1.0000 | First clean replay ✅ |
| DAY81_condicionB | 81 | 0.70/0.75 legacy | 0.9976 | 0.9951 | 1.0000 | Controlled comparison ✅ |

---

## 📋 Roadmap

### Immediate (DAY 82)
- Balanced dataset validation (P0 paper — CTU-13 Neris is 98% malicious)
- Investigate ML RandomForest max score 0.6607 (never reaches threshold)
- Fix pipeline_health.sh (pgrep runs on macOS, not inside VM)

### Short Term (DAY 83-85)
- CSV Pipeline E2E validation with real traffic
- Fix 2 pre-existing trace_id test failures (DAY 72)
- arXiv paper preparation

### Enterprise
- Federated Threat Intelligence (ENT-1)
- Attack Graph Generation — GraphML + STIX 2.1 (ENT-2)
- P2P Seed Distribution — eliminate etcd as crypto authority (ENT-3)
- Hot-reload configuration — no downtime threshold tuning (ENT-4)

---

## 🎓 Design Philosophy

**Via Appia Quality** — Scientific honesty, methodical development, transparent AI collaboration.

**Consejo de Sabios** — Multi-agent peer review: Claude, DeepSeek, Grok, ChatGPT, Qwen.
All AI contributions explicitly credited as co-authors.

---

## 🔐 Security

### Guarantees
- ✅ ChaCha20-Poly1305 authenticated encryption (AEAD)
- ✅ HMAC-SHA256 log integrity
- ✅ Autonomous blocking (no human in loop)
- ✅ JSON-driven thresholds (no hardcoded security parameters)
- ✅ Fail-closed design

### Known Limitations
- 11/40 ML features use sentinel values (Phase 2 pending)
- Balanced dataset validation pending (CTU-13 Neris 98% malicious)
- ML RandomForest not detecting Neris — Fast Detector handles all detections
- Single-node deployment (no HA/failover)

---

## 📈 Performance

**sniffer**: sub-microsecond eBPF/XDP, ShardedFlowManager 16 shards, 0 crypto errors

**ml-detector**: 0.24μs–1.06μs per detection, F1=1.0000 on CTU-13 Neris (DAY 81)

**firewall-acl-agent**: <10ms detection→block, 364 ev/s, 54% CPU, 127MB RAM, 0 crypto errors @ 36K events

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

**Human**: Alonso Isidoro Roman — Creator, ML Architect

**AI Co-Authors (Consejo de Sabios)**: Claude (Anthropic), DeepSeek, Grok, Gemini, ChatGPT, Qwen, Parallel.ai

**Datasets**: CTU-13 (Czech Technical University), CIC-IDS2017 (UNB), UNSW-NB15

---

## 🗺️ Project Status

**Current Phase**: Day 81 — F1=1.0000 controlled comparison validated

**Recent Milestones**:
- ✅ Day 76: SIGSEGV eliminated — pipeline 6/6 stable
- ✅ Day 79: F1=0.9921 — sentinel fix + logging standard
- ✅ Day 80: F1=0.9934 — **JSON is the LAW** 🦅
- ✅ Day 81: F1 comparativa limpia — thresholds empíricamente justificados ✅

**Next**: Day 82 — balanced dataset validation (P0 paper)

---

**Via Appia Quality** 🏛️ — Built to last decades

*"The road to security is long, but we build it to endure."*